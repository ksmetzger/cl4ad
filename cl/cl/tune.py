# adapted from https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from functools import partial
import os
import tempfile
from ray import tune #hyperparameter tuning library
import ray.cloudpickle as pickle
from ray.train import Checkpoint, session, get_checkpoint, report
from ray import train
from ray.tune.schedulers import ASHAScheduler
from types import SimpleNamespace
from models import SimpleDense
import losses
from train_with_signal import LARS, exclude_bias_and_norm, TorchCLDataset, adjust_learning_rate
import augmentations
from linear_evaluation import accuracy, test_accuracy, AverageMeter

#Config with hyper params which should be tuned
config = {
    #"base_lr": tune.loguniform(1e-2, 1),
    "base_lr": 0.025,
    "batch_size": 1024,
    #"naive_mask_p": tune.choice([(i+1)/10 for i in range(10)]),
    #"mask_particle": tune.choice([True, False]),
    "naive_mask_p": 0.4,
    "mask_particle": True,
    "inv_weight": tune.grid_search([1,10,25]),
    "var_weight": tune.grid_search([1,10,25]),
    "cov_weight": tune.grid_search([1,10,25]),
}

def train_tune_params(config, data_dir):
    #Namespace with training/model params
    args = SimpleNamespace(
    #Training params
    epochs = 10,
    epochs_eval = 2,
    weight_decay_eval = 1e-6,
    batch_size = config["batch_size"],
    base_lr = config["base_lr"],
    #Model params
    num_classes = 8,
    embed_dim = 48,
    #Loss params
    inv_weight = config["inv_weight"],
    var_weight = config["var_weight"],
    cov_weight = config["cov_weight"],
    #Augmentations
    naive_mask_p = config["naive_mask_p"],
    mask_particle = config["mask_particle"],
)
    
    train_tune(args, data_dir)

def load_data(data_dir=''):
    dataset = np.load(data_dir)
    x_train, labels_train = dataset['x_train'], dataset['labels_train']
    x_test, labels_test = dataset['x_test'], dataset['labels_test']
    x_val, labels_val = dataset['x_val'], dataset['labels_val']
    return x_train, x_test, x_val, labels_train, labels_test, labels_val

def train_tune(args, data_dir=None):
    #Define the models
    model = SimpleDense() #Import model (might change layers/layer size later with config[l1], ...)
    backbone = SimpleDense()
    head = nn.Linear(args.embed_dim, args.num_classes)

    model.requires_grad_(True)
    backbone.requires_grad_(False)
    head.requires_grad_(True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    model.to(device)
    backbone.to(device)
    head.to(device)

    #criterion = losses.SimCLRloss_nolabels_fast()
    criterion = losses.VICRegLoss(var_weight=args.var_weight, inv_weight=args.inv_weight, cov_weight=args.cov_weight)
    optimizer = LARS(model.parameters(), lr=0, weight_decay=1e-6, weight_decay_filter=exclude_bias_and_norm, lars_adaptation_filter=exclude_bias_and_norm)
    criterion_eval = nn.CrossEntropyLoss().to(device=device)
    optimizer_eval = torch.optim.SGD(head.parameters(), lr=1e-3, weight_decay=args.weight_decay_eval)
    scheduler_eval = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_eval, args.epochs_eval)

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["model_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    x_train, x_test, x_val, labels_train, labels_test, labels_val = load_data(data_dir)

    train_data_loader = DataLoader(
        TorchCLDataset(x_train, labels_train, device),
        batch_size=args.batch_size,
        shuffle=True)

    test_data_loader = DataLoader(
        TorchCLDataset(x_test, labels_test, device),
        batch_size=args.batch_size,
        shuffle=False)

    val_data_loader = DataLoader(
        TorchCLDataset(x_val, labels_val, device),
        batch_size=args.batch_size,
        shuffle=False)

    def train_one_epoch(args, epoch_index):
        running_sim_loss = 0.
        last_sim_loss = 0.

        for idx, val in enumerate(train_data_loader, 1):
            val = val[0]
            # only applicable to the final batch
            if val.shape[0] != args.batch_size:
                continue

            # embed entire batch with first value of the batch repeated
            #first_val_repeated = val[0].repeat(args.batch_size, 1)
            #For DeepSets needs input shape (bsz, 19 , 3)
            embedded_values_orig = model(augmentations.naive_masking(val,device=device, rand_number=0, p=args.naive_mask_p, mask_full_particle=args.mask_particle))
            #embedded_values_orig = model(augmentations.permutation(augmentations.rot_around_beamline(augmentations.gaussian_resampling_pT(augmentations.naive_masking(val, device=device, rand_number=0), device=device, rand_number=0), device=device, rand_number=0), device=device, rand_number=0))
            #embedded_values_aug = model(first_val_repeated)
            #embedded_values_aug = model((augmentations.permutation(augmentations.rot_around_beamline(val, device=device), device=device)).reshape(-1,19,3))
            embedded_values_aug = model(augmentations.naive_masking(val,device=device, rand_number=42, p=args.naive_mask_p, mask_full_particle=args.mask_particle))
            #embedded_values_aug = model(augmentations.permutation(augmentations.rot_around_beamline(augmentations.gaussian_resampling_pT(augmentations.naive_masking(val, device=device, rand_number=42), device=device, rand_number=42), device=device, rand_number=42), device=device, rand_number=42))
            #feature = torch.cat([embedded_values_orig.unsqueeze(dim=1),embedded_values_aug.unsqueeze(dim=1)],dim=1)
            similar_embedding_loss = criterion(embedded_values_orig.reshape((-1,96)), embedded_values_aug.reshape((-1,96)))
            #similar_embedding_loss = criterion(feature)

            optimizer.zero_grad()
            similar_embedding_loss.backward()
            optimizer.step()
            # Gather data and report
            running_sim_loss += similar_embedding_loss.item()
            if idx % 500 == 0:
                last_sim_loss = running_sim_loss / 500
                running_sim_loss = 0.
        return last_sim_loss
    def val_one_epoch(args, epoch_index):
        running_sim_loss = 0.
        last_sim_loss = 0.

        with torch.no_grad():
            for idx, val in enumerate(val_data_loader, 1):
                val = val[0]
                if val.shape[0] != args.batch_size:
                    continue

                #first_val_repeated = val[0].repeat(args.batch_size, 1)

                embedded_values_orig = model(augmentations.naive_masking(val,device=device, rand_number=0, p=args.naive_mask_p, mask_full_particle=args.mask_particle))
                #embedded_values_aug = model(first_val_repeated)
                #embedded_values_orig = model(augmentations.permutation(augmentations.rot_around_beamline(augmentations.gaussian_resampling_pT(augmentations.naive_masking(val, device=device, rand_number=0), device=device, rand_number=0), device=device, rand_number=0), device=device, rand_number=0))
                #embedded_values_aug = model((augmentations.permutation(augmentations.rot_around_beamline(val, device=device), device=device)).reshape(-1,19,3))
                embedded_values_aug = model(augmentations.naive_masking(val,device=device, rand_number=42, p=args.naive_mask_p, mask_full_particle=args.mask_particle))
                #embedded_values_aug = model(augmentations.permutation(augmentations.rot_around_beamline(augmentations.gaussian_resampling_pT(augmentations.naive_masking(val, device=device, rand_number=42), device=device, rand_number=42), device=device, rand_number=42), device=device, rand_number=42))
                #feature = torch.cat([embedded_values_orig.unsqueeze(dim=1),embedded_values_aug.unsqueeze(dim=1)],dim=1)
                similar_embedding_loss = criterion(embedded_values_orig.reshape((-1,96)), embedded_values_aug.reshape((-1,96)))
                #similar_embedding_loss = criterion(feature)

                running_sim_loss += similar_embedding_loss.item()
                if idx % 50 == 0:
                    last_sim_loss = running_sim_loss / 50
                    running_sim_loss = 0.
        return last_sim_loss
    
    train_losses = []
    val_losses = []
    for epoch in range(start_epoch, args.epochs):  # loop over the dataset multiple times
        print(f'EPOCH {epoch}')
        lr = adjust_learning_rate(args, 10, epoch+1 , optimizer, args.base_lr)
        print(f"current Learning rate: {lr:.4f}")
        # Gradient tracking
        model.train(True)
        avg_train_loss = train_one_epoch(args, epoch)
        train_losses.append(avg_train_loss)

        # no gradient tracking, for validation
        model.train(False)
        avg_val_loss = val_one_epoch(args, epoch)
        val_losses.append(avg_val_loss)

        print(f"Train/Val Sim Loss after epoch: {avg_train_loss:.4f}/{avg_val_loss:.4f}")


        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)

            train.report(
                {"loss": avg_val_loss, "accuracy": get_accuracy(args, model.state_dict(), backbone, head, train_data_loader, val_data_loader, test_data_loader,
                                                                optimizer_eval, criterion_eval, scheduler_eval)},
                checkpoint=checkpoint,
            )
    #Confirm the validation accuracy at the end of a try on the test dataset
    test_accuracy(test_data_loader, backbone, head)
    print("Finished Training")

def get_accuracy(args, state_dict, backbone, head, train_data_loader, val_data_loader, test_data_loader, 
                 optimizer, criterion, scheduler):
    #Load weights for backbone and initialize head
    backbone.load_state_dict(state_dict=state_dict, strict=False)
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()
    #Set model to eval
    backbone.eval()
    head.eval()
    #Namespace for the best acc that is reported
    best_acc = Namespace(top1=0)
    for epoch in range(args.epochs_eval):
        #Train
        for step, (data, target) in enumerate(train_data_loader, start = epoch*len(train_data_loader)):
            target = target.long()
            output = head(backbone.representation(data))
            loss = criterion(output, target.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #Evaluate
        top1 = AverageMeter("Acc@1")
        with torch.no_grad():
            for data, target in val_data_loader:
                output = head(backbone.representation(data))
                acc1, acc5 = accuracy(output, target, topk=(1,5))
                top1.update(acc1[0].item(), data.size(0))
            
        best_acc.top1 = max(best_acc.top1, top1.avg)

        scheduler.step()
    #Return the best top-1 accuracy as the metric for the tuner
    return best_acc.top1


def main(config, num_samples=20, cpus_per_trial=4, gpus_per_trial=1, device='cpu', restore=False):
    data_dir = os.path.abspath('dataset_background_signal.npz')
    trainable_with_resources = tune.with_resources(partial(train_tune_params, data_dir=data_dir), {"cpu": cpus_per_trial, "gpu": gpus_per_trial})
    if restore==False:
        scheduler = ASHAScheduler(
            #metric="accuracy",
            #mode="max",
            max_t=10,
            grace_period=5,
            reduction_factor=2,
        )
        # result = tune.run(
        #     partial(train_tune_params, data_dir=data_dir),
        #     resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        #     config=config,
        #     num_samples=num_samples,
        #     scheduler=scheduler,
        # )

        tuneConfig = tune.TuneConfig(metric="accuracy", mode="max", scheduler=scheduler, num_samples=num_samples)
        tuner = tune.Tuner(trainable_with_resources, param_space=config, tune_config=tuneConfig)
    elif restore:
        tuner = tune.Tuner.restore(path='C:\\Users\\Kyle\\ray_results\\train_tune_params_2024-05-17_17-03-53', trainable=trainable_with_resources, resume_unfinished=True, restart_errored=True)
    
    result = tuner.fit()

    best_trial = result.get_best_result("accuracy", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.metrics['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.metrics['accuracy']}")

    best_trained_model = SimpleDense()
    best_trained_model.to(device)

    best_checkpoint = best_trial.checkpoint
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["model_state_dict"])
        torch.save(best_trained_model.state_dict(), 'best_model_state_dict.pth')


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if str(device) == 'cpu':
        gpus_per_trial = 0
        cpus_per_trial = 8
    elif str(device) == 'cuda:0':
        gpus_per_trial = 1
        cpus_per_trial = 4

    main(config, num_samples=1, cpus_per_trial=cpus_per_trial ,gpus_per_trial=gpus_per_trial, device=device, restore=True)

