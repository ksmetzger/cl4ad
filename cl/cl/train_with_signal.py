import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import losses
from dataset import TorchCLDataset, CLBackgroundDataset, CLSignalDataset, CLBackgroundSignalDataset
from models import CVAE, SimpleDense, DeepSets
import augmentations
import math

def main(args):
    '''
    Infastructure for training CVAE (background specific and with anomalies)
    '''

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    #Dataset with signals and original divisions=[0.592,0.338,0.067,0.003]
    dataset = CLBackgroundSignalDataset(args.background_dataset, args.background_ids, args.anomaly_dataset,
        preprocess=args.scaling_filename, n_events=args.sample_size,
        divisions=[0.592,0.338,0.067,0.003],
        device=device
    )
    dataset.report_specs()

    train_data_loader = DataLoader(
        TorchCLDataset(dataset.x_train, dataset.labels_train, device),
        batch_size=args.batch_size,
        shuffle=True)

    test_data_loader = DataLoader(
        TorchCLDataset(dataset.x_test, dataset.labels_test, device),
        batch_size=args.batch_size,
        shuffle=False)

    val_data_loader = DataLoader(
        TorchCLDataset(dataset.x_val, dataset.labels_val, device),
        batch_size=args.batch_size,
        shuffle=False)

    model = SimpleDense().to(device)
    summary(model, input_size=(57,))

    # criterion = losses.SimCLRLoss()
    #criterion = losses.VICRegLoss()
    criterion = losses.SimCLRloss_nolabels_fast()
    #Standard schedule
    """ optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3) #Adams pytorch impl. of weight decay is equiv. to the L2 penalty.
    scheduler_1 = torch.optim.lr_scheduler.ConstantLR(optimizer, total_iters=5)
    scheduler_2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler_3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler_1, scheduler_2, scheduler_3], milestones=[5, 20]) """
    #Version 2 schedule
    #optimizer = torch.optim.SGD(model.parameters(), lr=0, weight_decay=1e-3)
    optimizer = LARS(model.parameters(), lr=0, weight_decay=1e-6, weight_decay_filter=exclude_bias_and_norm, lars_adaptation_filter=exclude_bias_and_norm)

    def train_one_epoch(epoch_index, tb_writer):
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
            #embedded_values_orig = model(augmentations.naive_masking(val,device=device, rand_number=0))
            embedded_values_orig = model(augmentations.gaussian_resampling_pT(augmentations.naive_masking(val, device=device, rand_number=0), device=device, rand_number=0))
            #embedded_values_aug = model(first_val_repeated)
            #embedded_values_aug = model((augmentations.permutation(augmentations.rot_around_beamline(val, device=device), device=device)).reshape(-1,19,3))
            #embedded_values_aug = model(augmentations.naive_masking(val,device=device, rand_number=42))
            embedded_values_aug = model(augmentations.gaussian_resampling_pT(augmentations.naive_masking(val, device=device, rand_number=42), device=device, rand_number=42))
            feature = torch.cat([embedded_values_orig.unsqueeze(dim=1),embedded_values_aug.unsqueeze(dim=1)],dim=1)
            #similar_embedding_loss = criterion(embedded_values_orig.reshape((-1,96)), embedded_values_aug.reshape((-1,96)))
            similar_embedding_loss = criterion(feature)

            optimizer.zero_grad()
            similar_embedding_loss.backward()
            optimizer.step()
            # Gather data and report
            running_sim_loss += similar_embedding_loss.item()
            if idx % 500 == 0:
                last_sim_loss = running_sim_loss / 500
                tb_x = epoch_index * len(train_data_loader) + idx
                tb_writer.add_scalar('SimLoss/train', last_sim_loss, tb_x)
                running_sim_loss = 0.
        return last_sim_loss


    def val_one_epoch(epoch_index, tb_writer):
        running_sim_loss = 0.
        last_sim_loss = 0.

        for idx, val in enumerate(val_data_loader, 1):
            val = val[0]
            if val.shape[0] != args.batch_size:
                continue

            #first_val_repeated = val[0].repeat(args.batch_size, 1)

            #embedded_values_orig = model(augmentations.naive_masking(val,device=device, rand_number=0))
	        #embedded_values_aug = model(first_val_repeated)
            embedded_values_orig = model(augmentations.gaussian_resampling_pT(augmentations.naive_masking(val, device=device, rand_number=0), device=device, rand_number=0))
            #embedded_values_aug = model((augmentations.permutation(augmentations.rot_around_beamline(val, device=device), device=device)).reshape(-1,19,3))
            #embedded_values_aug = model(augmentations.naive_masking(val,device=device, rand_number=42))
            embedded_values_aug = model(augmentations.gaussian_resampling_pT(augmentations.naive_masking(val, device=device, rand_number=42), device=device, rand_number=42))
            feature = torch.cat([embedded_values_orig.unsqueeze(dim=1),embedded_values_aug.unsqueeze(dim=1)],dim=1)
            #similar_embedding_loss = criterion(embedded_values_orig.reshape((-1,96)), embedded_values_aug.reshape((-1,96)))
            similar_embedding_loss = criterion(feature)

            running_sim_loss += similar_embedding_loss.item()
            if idx % 50 == 0:
                last_sim_loss = running_sim_loss / 50
                tb_x = epoch_index * len(val_data_loader) + idx + 1
                tb_writer.add_scalar('SimLoss/val', last_sim_loss, tb_x)
                tb_writer.flush()
                running_sim_loss = 0.
        tb_writer.flush()
        return last_sim_loss

    def adjust_learning_rate(args, warmup_epochs ,epoch, optimizer, base_lr):
        max_epochs = args.epochs
        base_lr = base_lr * args.batch_size / 256 #Scale like suggested by VICReg for base_lr = 0.2 (SimCLR has base_lr = 0.3)
        if epoch <= warmup_epochs:
            lr = base_lr * epoch / warmup_epochs #Linear warmup
        else:
            epoch -= warmup_epochs
            max_epochs -= warmup_epochs
            q = 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))
            end_lr = base_lr * 0.001
            lr = base_lr * q + end_lr * (1-q)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr


    writer = SummaryWriter("output/results", comment="Similarity with LR=1e-3", flush_secs=5)

    if args.train:
        train_losses = []
        val_losses = []
        for epoch in range(1, args.epochs+1):
            print(f'EPOCH {epoch}')
            #Adjust the learning rate with Version 2 schedule (see OneNote)
            lr = adjust_learning_rate(args, 10, epoch, optimizer, base_lr=0.2)
            print("current Learning rate: ", lr)
            writer.add_scalar('Learning_rate', lr, epoch)
            # Gradient tracking
            model.train(True)
            avg_train_loss = train_one_epoch(epoch, writer)
            train_losses.append(avg_train_loss)

            # no gradient tracking, for validation
            model.train(False)
            avg_val_loss = val_one_epoch(epoch, writer)
            val_losses.append(avg_val_loss)

            print(f"Train/Val Sim Loss after epoch: {avg_train_loss:.4f}/{avg_val_loss:.4f}")

            #scheduler.step()
        writer.flush()
        torch.save(model.state_dict(), args.model_name)

        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.xlabel('iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('output/loss.pdf')
        
    else:
        model.load_state_dict(torch.load(args.model_name, map_location=torch.device(device)))
        model.eval()

        #Save the embedding output seperately for the background and signal part
        #dataset.save(args.output_filename, model)
        CLBackgroundDataset(args.background_dataset, args.background_ids, n_events=args.sample_size,
            preprocess=args.scaling_filename,
            divisions=[0.592,0.338,0.067,0.003],
            device=device).save_embedding(args.output_filename, model)
        CLSignalDataset(args.anomaly_dataset,n_events=args.sample_size, preprocess=args.scaling_filename, device=device).save_embedding(f'output/anomalies_embedding.npz', model)

class LARS(torch.optim.Optimizer): #Implementation from https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py.
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])

def exclude_bias_and_norm(p):
    return p.ndim == 1

class TorchCLDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, features, labels, device):
        'Initialization'
        self.device = device
        self.features = torch.from_numpy(features).to(dtype=torch.float32, device=self.device)
        self.labels = torch.from_numpy(labels).to(dtype=torch.float32, device=self.device)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.features)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.features[index]
        y = self.labels[index]

        return X, y
  
if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()

    # If not using full data, name of smaller dataset to pull from
    parser.add_argument('background_dataset', type=str)
    parser.add_argument('background_ids', type=str)
    parser.add_argument('anomaly_dataset', type=str)

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--loss-temp', type=float, default=0.07)
    parser.add_argument('--model-name', type=str, default='output/vae.pth')
    parser.add_argument('--scaling-filename', type=str)
    parser.add_argument('--output-filename', type=str, default='output/embedding.npz')
    parser.add_argument('--sample-size', type=int, default=-1)
    parser.add_argument('--mix-in-anomalies', action='store_true')
    parser.add_argument('--train', action='store_true')

    args = parser.parse_args()
    main(args)
