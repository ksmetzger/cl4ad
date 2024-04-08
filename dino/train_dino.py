import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import time
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import dino_loss
from dataset import TorchCLDataset, CLBackgroundDataset, CLSignalDataset, CLBackgroundSignalDataset
from transformer import TransformerEncoder
import augmentations
import utils

def main(args):
    '''
    Infastructure for training transformer with DINO (background specific and with anomalies)
    '''
    #For now only train on one gpu (look into distributed afterwards)
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
        TorchCLDataset(dataset.x_train[0:5000], dataset.labels_train[0:5000], device),
        batch_size=args.batch_size,
        shuffle=True, drop_last=True)

    test_data_loader = DataLoader(
        TorchCLDataset(dataset.x_test, dataset.labels_test, device),
        batch_size=args.batch_size,
        shuffle=False)

    val_data_loader = DataLoader(
        TorchCLDataset(dataset.x_val[0:5000], dataset.labels_val[0:5000], device),
        batch_size=args.batch_size,
        shuffle=False, drop_last=True)

    #Build student and teacher models and move them to device
    student = TransformerEncoder().to(device)
    teacher = TransformerEncoder().to(device)
    summary(student, input_size=(19,3))
    # teacher and student start with the same weights
    teacher.load_state_dict(student.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    #Prepare the DINO loss
    criterion = dino_loss.DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).to(device)

    #Prepare the optimizer
    params_groups = utils.get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)
    #Schedulers
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(train_data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(train_data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule (For EMA updates)
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(train_data_loader))



    def train_one_epoch(epoch_index, tb_writer):
        running_sim_loss = 0.
        last_sim_loss = 0.

        for it, val in enumerate(train_data_loader, 1):
            val = val[0]
            # only applicable to the final batch
            if val.shape[0] != args.batch_size:
                continue

            it = len(train_data_loader) * (epoch_index-1) + it  # global training iteration
            #Update the schedulers
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]

            # teacher and student forward passes + compute dino loss
            embedded_values_orig_student = student(val.reshape(-1,19,3))
            embedded_values_aug_student = student((augmentations.permutation(augmentations.rot_around_beamline(val, device=device), device=device)).reshape(-1,19,3))
            embedded_values_orig_teacher = teacher(val.reshape(-1,19,3))
            embedded_values_aug_teacher = teacher((augmentations.permutation(augmentations.rot_around_beamline(val, device=device), device=device)).reshape(-1,19,3))
            teacher_output = torch.cat([embedded_values_orig_teacher,embedded_values_aug_teacher],dim=0)
            student_output = torch.cat([embedded_values_orig_student,embedded_values_aug_student],dim=0)
            loss = criterion(student_output, teacher_output, epoch_index)

            #Update the student network
            optimizer.zero_grad()
            param_norms = None
            loss.backward()
            optimizer.step()

            #Update the teacher network via EMA
            with torch.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # Gather data and report
            running_sim_loss += loss.item()
            if it % 500 == 0:
                last_sim_loss = running_sim_loss / 500
                tb_x = epoch_index * len(train_data_loader) + it
                tb_writer.add_scalar('DINOLoss/train', last_sim_loss, tb_x)
                running_sim_loss = 0.
        return last_sim_loss

    @torch.no_grad()
    def val_one_epoch(epoch_index, tb_writer):
        running_sim_loss = 0.
        last_sim_loss = 0.

        for it, val in enumerate(val_data_loader, 1):
            val = val[0]
            if val.shape[0] != args.batch_size:
                continue

            # teacher and student forward passes + compute dino loss
            embedded_values_orig_student = student(val.reshape(-1,19,3))
            embedded_values_aug_student = student((augmentations.permutation(augmentations.rot_around_beamline(val, device=device), device=device)).reshape(-1,19,3))
            embedded_values_orig_teacher = teacher(val.reshape(-1,19,3))
            embedded_values_aug_teacher = teacher((augmentations.permutation(augmentations.rot_around_beamline(val, device=device), device=device)).reshape(-1,19,3))
            teacher_output = torch.cat([embedded_values_orig_teacher,embedded_values_aug_teacher],dim=0)
            student_output = torch.cat([embedded_values_orig_student,embedded_values_aug_student],dim=0)
            loss = criterion(student_output, teacher_output, epoch_index)

            running_sim_loss += loss.item()
            if it % 50 == 0:
                last_sim_loss = running_sim_loss / 50
                tb_x = epoch_index * len(val_data_loader) + it + 1
                tb_writer.add_scalar('DINOLoss/val', last_sim_loss, tb_x)
                tb_writer.flush()
                running_sim_loss = 0.
        tb_writer.flush()
        return last_sim_loss

    writer = SummaryWriter("output/results", comment="Similarity with standard DINOv1 settings", flush_secs=5)

    if args.train:
        train_losses = []
        val_losses = []
        start_time = time.time()
        print("Starting DINO training !")
        for epoch in range(1, args.epochs+1):
            print(f'EPOCH {epoch}')
            # Gradient tracking
            student.train(True), teacher.train(True)
            avg_train_loss = train_one_epoch(epoch, writer)
            train_losses.append(avg_train_loss)

            # no gradient tracking, for validation
            student.train(False), teacher.train(False)
            avg_val_loss = val_one_epoch(epoch, writer)
            val_losses.append(avg_val_loss)

            print(f"Train/Val DINO Loss after epoch: {avg_train_loss:.4f}/{avg_val_loss:.4f}")

        #Save both networks for now
        torch.save(student.state_dict(), str(args.model_name + "_student"))
        torch.save(teacher.state_dict(), str(args.model_name + "_teacher"))
        #Add timing and print it
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.xlabel('iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('output/loss.pdf')
        
    else:
        student.load_state_dict(torch.load(args.model_name, map_location=torch.device(device)))
        student.eval()

    #Save the embedding output seperately for the background and signal part
    #dataset.save(args.output_filename, model)
    with torch.no_grad():
        CLBackgroundDataset(args.background_dataset, args.background_ids, n_events=args.sample_size,
            preprocess=args.scaling_filename,
            divisions=[0.592,0.338,0.067,0.003],
            device=device).save(args.output_filename, student)
        CLSignalDataset(args.anomaly_dataset,n_events=args.sample_size, preprocess=args.scaling_filename, device=device).save(f'output/anomalies_embedding.npz', student)


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
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')
    parser.add_argument('--out_dim', default=512, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    
    args = parser.parse_args()
    main(args)
