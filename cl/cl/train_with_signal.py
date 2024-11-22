import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os
import time, datetime
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import losses
from models import CVAE, SimpleDense, DeepSets, SimpleDense_small, SimpleDense_JetClass, CVAE_JetClass, SimpleDense_ADC
from transformer import TransformerEncoder
import augmentations
import math
import h5py
from tqdm import tqdm
from sklearn.utils import shuffle

def main(args, train_idx, val_idx):
    '''
    Infastructure for training CVAE (background specific and with anomalies)
    '''

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    #Seed
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    #Dataset with signals and original divisions=[0.592,0.338,0.067,0.003]
    #dataset = np.load(args.dataset)
    print(f"Using train folds: {train_idx}")
    print(f"and using val fold: {val_idx}")
    with h5py.File(f'{args.dataset}', 'r') as f:
        train_fold = np.concatenate([np.array(f[f'x_train_fold_{idx}'][...]) for idx in train_idx], axis=0)
        labels_train_fold = np.concatenate([np.array(f[f'labels_train_fold_{idx}'][...]) for idx in train_idx], axis=0)
        val_fold = np.array(f[f'x_train_fold_{val_idx}'][...])
        labels_val_fold = np.array(f[f'labels_train_fold_{val_idx}'][...])
        x_test = np.array(f['x_test'][...])
        labels_test = np.array(f['labels_test'][...])
    #Shuffle the train dataset
    train_fold, labels_train_fold = shuffle(train_fold, labels_train_fold, random_state=0)

    feat_dim = 57
    if args.type == 'JetClass' or args.type == 'JetClass_Transformer':
        feat_dim = 512

    #For corruption augm. get min, max, mean, std values of all the features in the training dataset
    charac_trainset = dict(
        feat_low = np.min(train_fold.reshape(-1,feat_dim), axis=0), 
        feat_high = np.max(train_fold.reshape(-1,feat_dim), axis=0),
        feat_mean = np.mean(train_fold.reshape(-1,feat_dim), axis=0), 
        feat_std = np.std(train_fold.reshape(-1, feat_dim), axis=0),
    )
    #Initialize transform (empty list: None)
    transform = augmentations.Transform(["naive_masking"], feat_dim)
    
    train_data_loader = DataLoader(
        TorchCLDataset(train_fold.reshape(-1,feat_dim), labels_train_fold.reshape(-1), device),
        batch_size=args.batch_size,
        shuffle=True)

    test_data_loader = DataLoader(
        TorchCLDataset(x_test.reshape(-1,feat_dim), labels_test.reshape(-1), device),
        batch_size=args.batch_size,
        shuffle=False)

    val_data_loader = DataLoader(
        TorchCLDataset(val_fold.reshape(-1,feat_dim), labels_val_fold.reshape(-1), device),
        batch_size=args.batch_size,
        shuffle=False)

    if args.type == 'JetClass':
        model = SimpleDense_JetClass(args.latent_dim).to(device)
        #model = CVAE_JetClass(args.latent_dim).to(device)
        summary(model, input_size=(512,))
    elif args.type == 'JetClass_Transformer':
        transformer_args_jetclass = dict(
        input_dim=4, 
        model_dim=128, 
        output_dim=64,
        embed_dim=6,   #Only change embed_dim without describing new transformer architecture
        n_heads=8, 
        dim_feedforward=256, 
        n_layers=4,
        hidden_dim_dino_head=256,
        bottleneck_dim_dino_head=64,
        pos_encoding = True,
        use_mask = True,
        mode='cls',
        )
        model = TransformerEncoder(**transformer_args_jetclass)
        summary(model, input_size=(128,4))
    else:
        #model = SimpleDense(args.latent_dim).to(device)
        #model = SimpleDense_small().to(device)
        model = SimpleDense_ADC(args.latent_dim).to(device)
        summary(model, input_size=(57,))

    # criterion = losses.SimCLRLoss()
    #criterion = losses.VICRegLoss(inv_weight=10, var_weight=25, cov_weight=10)
    criterion = losses.SimCLRloss_nolabels_fast(temperature=args.loss_temp, base_temperature=args.loss_temp, contrast_mode='one')
    #Standard schedule
    """ optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3) #Adams pytorch impl. of weight decay is equiv. to the L2 penalty.
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
        loop = tqdm(train_data_loader)
        for idx, (val, labels) in enumerate(loop, 1):
            loop.set_description(f'Epoch {epoch_index}')
            # only applicable to the final batch
            if val.shape[0] != args.batch_size:
                continue

            # embed entire batch with first value of the batch repeated
            #first_val_repeated = val[0].repeat(args.batch_size, 1)
            #For DeepSets needs input shape (bsz, 19 , 3)
            #embedded_values_orig = model(val)
            if 'Transformer' in args.type:
                embedded_values_orig = model(transform(val).reshape(-1,128,4).to(device=device))
                embedded_values_aug = model(transform(val).reshape(-1,128,4).to(device=device))
            else:
                #embedded_values_orig = model(transform(val).to(device=device))
                #embedded_values_aug = model(transform(val).to(device=device))
                pass
            #embedded_values_orig = model(augmentations.naive_masking(val,device=device, rand_number=0))
            #embedded_values_orig = model(augmentations.permutation(augmentations.rot_around_beamline(augmentations.gaussian_resampling_pT(augmentations.naive_masking(val, device=device, rand_number=0), device=device, rand_number=0), device=device, rand_number=0), device=device, rand_number=0))
            #embedded_values_aug = model(first_val_repeated)
            #embedded_values_aug = model(augmentations.corruption(val, **charac_trainset, rand_number=0, device=device, mode='gaussian'))
            #embedded_values_aug = model((augmentations.permutation(augmentations.rot_around_beamline(val, device=device), device=device)).reshape(-1,19,3))
            #embedded_values_aug = model(augmentations.naive_masking(val,device=device, rand_number=42))
            #embedded_values_aug = model(augmentations.permutation(augmentations.rot_around_beamline(augmentations.gaussian_resampling_pT(augmentations.naive_masking(val, device=device, rand_number=42), device=device, rand_number=42), device=device, rand_number=42), device=device, rand_number=42))
            #feature = torch.cat([embedded_values_orig.unsqueeze(dim=1),embedded_values_aug.unsqueeze(dim=1)],dim=1)
            #similar_embedding_loss = criterion(embedded_values_orig, embedded_values_aug)

            #similar_embedding_loss = criterion(feature)
            
            #For supervised input, only give one view
            embedded_values_orig = model(val.to(device))
            feature = embedded_values_orig.unsqueeze(dim=1)

            similar_embedding_loss = criterion(feature, labels.reshape(-1))

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
            loop.set_postfix(train_loss=last_sim_loss)
        return last_sim_loss


    def val_one_epoch(epoch_index, tb_writer):
        running_sim_loss = 0.
        last_sim_loss = 0.

        with torch.no_grad():
            for idx, (val, labels) in enumerate(val_data_loader, 1):
                #val = val[0]
                #labels = val[1]
                if val.shape[0] != args.batch_size:
                    continue

                #first_val_repeated = val[0].repeat(args.batch_size, 1)
                if 'Transformer' in args.type:
                    embedded_values_orig = model(transform(val).reshape(-1,128,4).to(device=device))
                    embedded_values_aug = model(transform(val).reshape(-1,128,4).to(device=device))
                else:
                    #embedded_values_orig = model(transform(val).to(device=device))
                    #embedded_values_aug = model(transform(val).to(device=device))
                    pass
                #embedded_values_orig = model(val)
                #embedded_values_orig = model(augmentations.naive_masking(val,device=device, rand_number=0))
                #embedded_values_aug = model(first_val_repeated)
                #embedded_values_orig = model(augmentations.permutation(augmentations.rot_around_beamline(augmentations.gaussian_resampling_pT(augmentations.naive_masking(val, device=device, rand_number=0), device=device, rand_number=0), device=device, rand_number=0), device=device, rand_number=0))
                #embedded_values_aug = model(augmentations.corruption(val, **charac_trainset, rand_number=0, device=device, mode='gaussian'))
                #embedded_values_aug = model((augmentations.permutation(augmentations.rot_around_beamline(val, device=device), device=device)).reshape(-1,19,3))
                #embedded_values_aug = model(augmentations.naive_masking(val,device=device, rand_number=42))
                #embedded_values_aug = model(augmentations.permutation(augmentations.rot_around_beamline(augmentations.gaussian_resampling_pT(augmentations.naive_masking(val, device=device, rand_number=42), device=device, rand_number=42), device=device, rand_number=42), device=device, rand_number=42))
                #feature = torch.cat([embedded_values_orig.unsqueeze(dim=1),embedded_values_aug.unsqueeze(dim=1)],dim=1)
                #similar_embedding_loss = criterion(embedded_values_orig, embedded_values_aug)
                
                #similar_embedding_loss = criterion(feature)

                #For supervised input, only give one view
                embedded_values_orig = model(val.to(device))
                feature = embedded_values_orig.unsqueeze(dim=1)
                
                similar_embedding_loss = criterion(feature, labels.reshape(-1))

                running_sim_loss += similar_embedding_loss.item()
                if idx % 50 == 0:
                    last_sim_loss = running_sim_loss / 50
                    tb_x = epoch_index * len(val_data_loader) + idx + 1
                    tb_writer.add_scalar('SimLoss/val', last_sim_loss, tb_x)
                    tb_writer.flush()
                    running_sim_loss = 0.
        tb_writer.flush()
        return last_sim_loss

    writer = SummaryWriter("output/results", comment="Similarity with LR=1e-3", flush_secs=5)

    if args.train:
        train_losses = []
        val_losses = []
        #Initialize the Early Stopper
        folder = "output\checkpoints"
        os.makedirs(folder, exist_ok=True)
        #EarlyStopper = EarlyStopping(patience=8, delta=0, path=os.path.join(folder,args.model_name + f"_valfold_{val_idx}"), verbose=True)
        start_time = time.time()

        for epoch in range(1, args.epochs+1):
            print(f'EPOCH {epoch}')
            temp_time= time.time()
            #Adjust the learning rate with Version 2 schedule (see OneNote)
            lr = adjust_learning_rate(args, 10, epoch, optimizer, base_lr=0.3)
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

            temp_time = time.time()-temp_time
            print(f"Train/Val Sim Loss after epoch: {avg_train_loss:.4f}/{avg_val_loss:.4f}")
            print(f"taking {temp_time:.1f}s to complete")

            #Save checkpoint
            path = os.path.join(folder, args.model_name)
            torch.save(model.state_dict(), f"{path}_valfold_{val_idx}_vae_ep_{epoch}.pth")

            """ #Check whether to EarlyStop
            EarlyStopper(avg_val_loss, model, epoch)
            if EarlyStopper.early_stop:
                break """

            #scheduler.step()
        writer.flush()
        #torch.save(model.state_dict(), args.model_name)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.xlabel('iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'output/loss_valfold_{val_idx}.pdf')
        
    else:
        model.load_state_dict(torch.load(args.model_name, map_location=torch.device(device)))
        model.eval()

        """ #Save the embedding output for the background and signal part
        embedding_dict = dict()
        train_data_loader = DataLoader(
        TorchCLDataset(dataset['x_train'], dataset['labels_train'], device),
        batch_size=args.batch_size,
        shuffle=False)
        for loader, name in zip([train_data_loader, test_data_loader, val_data_loader],['train','test','val']):
            with torch.no_grad():
                embedding = np.concatenate([model.representation(data.to(device)).cpu().detach().numpy() for (data, label) in loader], axis=0)
                embedding_dict[f"embedding_{name}"] = embedding
                embedding_dict[f"labels_{name}"] = dataset[f'labels_{name}']

        np.savez(args.output_filename, **embedding_dict)
        print(f"Successfully saved embedding under {args.output_filename}") """

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

def adjust_learning_rate(args, warmup_epochs ,epoch, optimizer, base_lr):
        max_epochs = args.epochs
        base_lr = base_lr * args.batch_size / 256 #Scale like suggested by VICReg for base_lr = 0.4 (SimCLR has base_lr = 0.3)
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

class TorchCLDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, features, labels, device):
        'Initialization'
        self.device = device
        self.mean = np.mean(features)
        self.std = np.std(features)
        #print(f"Mean: {self.mean} and std: {self.std}")
        self.features = torch.from_numpy(features).to(dtype=torch.float32)
        self.labels = torch.from_numpy(labels).to(dtype=torch.float32)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.features)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.features[index]
        y = self.labels[index]
        #Normalize again (!!! has already been pre-normalized with z_score pT norm.)
        #X = (X-self.mean)/self.std

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
            self.save_checkpoint(model, epoch)
        elif val_loss >= self.min_val_loss - self.delta:
            self.counter += 1
            print(f"Early Stopper count at {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early Stopped at epoch {epoch}! And saved model from epoch {epoch-self.counter} with validation loss {self.min_val_loss}")
        else:
            assert(False)
                
    def save_checkpoint(self, model, epoch):
        '''Saves the model if the val_loss is decreasing'''
        file_path = f"{self.path}_ep_{epoch}.pth"
        torch.save(model.state_dict(), file_path)

if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()

    # If not using full data, name of smaller dataset to pull from
    parser.add_argument('dataset', type=str)

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--loss-temp', type=float, default=0.07)
    parser.add_argument('--model-name', type=str, default='SimpleDense_Small')
    parser.add_argument('--scaling-filename', type=str)
    parser.add_argument('--output-filename', type=str, default='output/embedding.npz')
    parser.add_argument('--sample-size', type=int, default=-1)
    parser.add_argument('--mix-in-anomalies', action='store_true')
    parser.add_argument('--latent-dim', type=int, default=48)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--type', choices=('Delphes', 'JetClass', 'JetClass_Transformer'))
    parser.add_argument('--k-fold', action='store_true')


    args = parser.parse_args()
    #Do the k-folding (runs the main loop five times)
    if args.k_fold:
        for i in range(5):
            train_idx = [0,1,2,3,4]
            train_idx.remove(i)
            val_idx = i
            main(args, train_idx=train_idx, val_idx=val_idx)
    else:
        #To test just set the trainset to fold0 and valset to fold1
        main(args, train_idx=[0], val_idx=1)
