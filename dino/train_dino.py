import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import time
import datetime
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import dino_loss
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
    dataset = np.load(os.path.join(os.getcwd(), args.dataset))

    train_data_loader = DataLoader(
        TorchCLDataset(dataset['x_train'], dataset['labels_train'], device),
        batch_size=args.batch_size,
        shuffle=True, drop_last=True)

    test_data_loader = DataLoader(
        TorchCLDataset(dataset['x_test'], dataset['labels_test'], device),
        batch_size=args.batch_size,
        shuffle=False)

    val_data_loader = DataLoader(
        TorchCLDataset(dataset['x_val'], dataset['labels_val'], device),
        batch_size=args.batch_size,
        shuffle=False, drop_last=True)
    
    #Transformer + DINO head architecture args
    transformer_args = dict(
        input_dim=3, 
        model_dim=64, 
        output_dim=args.out_dim, 
        n_heads=8, 
        dim_feedforward=256, 
        n_layers=4,
        hidden_dim_dino_head=256,
        bottleneck_dim_dino_head=64,
    )

    #Build student and teacher models and move them to device
    student = TransformerEncoder(**transformer_args, norm_last_layer=args.norm_last_layer).to(device)
    teacher = TransformerEncoder(**transformer_args).to(device)
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
            embedded_values_orig_student = student(augmentations.naive_masking(val, device=device, rand_number=0, p=0.5, mask_full_particle=False).reshape(-1,19,3))
            embedded_values_aug_student = student(augmentations.naive_masking(val, device=device, rand_number=42, p=0.5, mask_full_particle=False).reshape(-1,19,3))
            embedded_values_orig_teacher = teacher(augmentations.naive_masking(val, device=device, rand_number=0, p=0.5, mask_full_particle=False).reshape(-1,19,3))
            embedded_values_aug_teacher = teacher(augmentations.naive_masking(val, device=device, rand_number=42, p=0.5, mask_full_particle=False).reshape(-1,19,3))
            teacher_output = torch.cat([embedded_values_orig_teacher,embedded_values_aug_teacher],dim=0)
            student_output = torch.cat([embedded_values_orig_student,embedded_values_aug_student],dim=0)
            loss = criterion(student_output, teacher_output, epoch_index)

            #Update the student network
            optimizer.zero_grad()
            param_norms = None
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
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
            embedded_values_orig_student = student(augmentations.naive_masking(val, device=device, rand_number=0, p=0.5, mask_full_particle=False).reshape(-1,19,3))
            embedded_values_aug_student = student(augmentations.naive_masking(val, device=device, rand_number=42, p=0.5, mask_full_particle=False).reshape(-1,19,3))
            embedded_values_orig_teacher = teacher(augmentations.naive_masking(val, device=device, rand_number=0, p=0.5, mask_full_particle=False).reshape(-1,19,3))
            embedded_values_aug_teacher = teacher(augmentations.naive_masking(val, device=device, rand_number=42, p=0.5, mask_full_particle=False).reshape(-1,19,3))
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

    writer = SummaryWriter(os.path.join(os.getcwd(), "output/results"), comment="Similarity with standard DINOv1 settings", flush_secs=5)

    if args.train:
        train_losses = []
        val_losses = []
        start_time = time.time()
        #Initialize the Early Stopper
        EarlyStopper = EarlyStopping(patience=5, delta=0, path=args.model_name, verbose=True)
        print("Starting DINO training !")
        for epoch in range(1, args.epochs+1):
            print(f'EPOCH {epoch}')
            temp_time = time.time()
            # Gradient tracking
            student.train(True), teacher.train(True)
            avg_train_loss = train_one_epoch(epoch, writer)
            train_losses.append(avg_train_loss)

            # no gradient tracking, for validation
            student.train(False), teacher.train(False)
            avg_val_loss = val_one_epoch(epoch, writer)
            val_losses.append(avg_val_loss)

            temp_time = time.time()-temp_time
            print(f"Train/Val DINO Loss after epoch: {avg_train_loss:.4f}/{avg_val_loss:.4f}")
            print(f"taking {temp_time:.1f}s to complete")

            #Check whether to EarlyStop
            EarlyStopper(avg_val_loss, [teacher, student], epoch)
            if EarlyStopper.early_stop:
                break

        #Save both networks for now
        #torch.save(student.state_dict(), os.path.join(os.getcwd(), "_student_" + args.model_name))
        #torch.save(teacher.state_dict(), os.path.join(os.getcwd(), "_teacher_" + args.model_name))
        #Add timing and print it
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.xlabel('iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), 'output/loss.pdf'))
        
    else:
        #Evaluate using the teacher weights (backbone) as suggested by first author
        teacher.load_state_dict(torch.load(os.path.join(os.getcwd(), "_teacher_" + args.model_name), map_location=torch.device(device)))
        teacher.eval()

        #Save the embedding output for the background and signal part
        embedding_dict = dict()
        train_data_loader = DataLoader(
        TorchCLDataset(dataset['x_train'], dataset['labels_train'], device),
        batch_size=args.batch_size,
        shuffle=False)
        val_data_loader = DataLoader(
        TorchCLDataset(dataset['x_val'], dataset['labels_val'], device),
        batch_size=args.batch_size,
        shuffle=False)
        for loader, name in zip([train_data_loader, test_data_loader, val_data_loader],['train','test','val']):
            with torch.no_grad():
                embedding = np.concatenate([teacher.representation(data).cpu().detach().numpy() for (data, label) in loader], axis=0)
                embedding_dict[f"embedding_{name}"] = embedding

        np.savez(args.output_filename, **embedding_dict)
        print(f"Successfully saved embedding under {args.output_filename}")

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

class EarlyStopping:
    '''Defines an EarlyStopper which stops training when the validation loss does not decrease with certain patience'''
    def __init__(self, patience=5, delta=0, path='default.pth', verbose=False):
        '''
        Args:
            patience: How many epochs to wait for val_loss to improve again (default: 5)
            delta: minimum change in the val_loss metric to qualify as an improvement (default: 0)
            path: output path of the checkpointed model (default: 'default.pth')
            verbose: print everytime the val_loss significantly lowers and the model is saved (default: False)
        '''
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        
        self.counter = 0
        self.min_val_loss = float(np.Inf)
        self.early_stop = False

    def __call__(self, val_loss, model, epoch):
        
        assert(val_loss != np.nan)

        if val_loss < self.min_val_loss - self.delta:
            if self.verbose:
                print(f"Validation loss lowered from {self.min_val_loss:.4f} ---> to {val_loss:.4f} and the model was saved!")
            self.min_val_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        elif val_loss > self.min_val_loss - self.delta:
            self.counter += 1
            print(f"Early Stopper count at {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early Stopped at epoch {epoch}! And saved model from epoch {epoch-self.counter} with validation loss {self.min_val_loss}")
        else:
            assert(False)
                
    def save_checkpoint(self, model):
        '''Saves the model if the val_loss is decreasing'''
        names = ["teacher", "student"]
        for name, mod in zip(names, model):
            torch.save(mod.state_dict(), os.path.join(os.getcwd(), f"_{name}_{self.path}"))

if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()

    # If not using full data, name of smaller dataset to pull from
    parser.add_argument('dataset', type=str)

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--loss-temp', type=float, default=0.07)
    parser.add_argument('--model-name', type=str, default='dino_transformer.pth')
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
    parser.add_argument('--out_dim', default=64, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--weight_decay', type=float, default=0.004, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.04, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument("--lr", default=0.00005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=5, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-7, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    
    args = parser.parse_args()
    main(args)
