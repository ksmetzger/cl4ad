import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from transformer import TransformerEncoder, Identity
import augmentations
import argparse
from pathlib import Path
import warnings
import torch.nn as nn
from torchsummary import summary
import time
import json
import sys
import random
import h5py

'''Linear evaluation of self-supervised embedding in order to calculate top1/top5 accuracies as
   standard in literature and leaderboards. Inspired by https://github.com/facebookresearch/vicreg/blob/main/evaluate.py.'''

def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate the linear top1/top5 accuracy given the pretrained embedding")
    
    parser.add_argument('dataset', type=str)

    parser.add_argument('--scaling-filename', type=str)
    parser.add_argument('--sample-size', type=int, default=-1)

    parser.add_argument("--pretrained",default="output/runs1/_teacher_dino_transformer.pth", type=Path, help="path to pretrained model")
    parser.add_argument("--epochs", default=10, type=int, metavar="N", help="number of epochs to train linear layer")
    parser.add_argument("--batch-size", default=1024, type=int, metavar="N")
    parser.add_argument("--weight-decay", default=1e-6, type=float, metavar="W", help="weight decay (for SGD and ADAM this is equivilant to the L2 penalty)")
    parser.add_argument("--arch", type=str, choices=("TransformerEncoder", "TransformerEncoder_JetClass", "NoEmbedding"), help="Type of model used to train embedding (backbone)")
    parser.add_argument("--num-classes", default=8, type=int, help="number of classes (background + signals)")
    parser.add_argument("--type", default="freeze", type=str, choices=("freeze", "finetune"), help="Whether to freeze weights and train full data on linear layer or finetune the model with a semi-supervised approach.")
    parser.add_argument("--percent", default=1, type=int, choices=(1,10), help="If finetune, choose percentage of labeled train data the model is finetuned on.")
    parser.add_argument('--head-name', type=str, default='output/head.pth')
    parser.add_argument('--backbone-name', type=str, default='output/backbone_finetuned.pth')
    parser.add_argument('--k-fold', type=int, default=-1)
    return parser


def main():
    #Set Seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    #Get parser and device
    parser = get_arguments()
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    #Stats file
    if args.type == "freeze":
        stats_file = open("output/stats.txt", "a", buffering=1)
    elif args.type == "finetune":
        stats_file = open("output/stats_finetuned.txt", "a", buffering=1)
    else:
        assert False
    print(" ".join(sys.argv))
    print(" ".join(sys.argv), file=stats_file)
    best_acc = argparse.Namespace(top1=0, top5=0)

    if args.arch == "TransformerEncoder_JetClass":
        feat_dim = 4
        num_const = 128
    else:
        num_const = 19
        feat_dim = 3

    #Dataset with signals and original divisions=[0.592,0.338,0.067,0.003]
    #x_train, x_test, x_val, labels_train, labels_test, labels_val = load_data("C:\\Users\\Kyle\\OneDrive\\Transfer Master project\\orca_fork\\cl4ad\\cl\\cl\\"+args.dataset, feat_dim, num_const)
    #Load k-folding if k-fold != -1
    if args.k_fold != -1:
        train_idx = [0,1,2,3,4]
        train_idx.remove(args.k_fold)
        val_idx = args.k_fold
    else:
        train_idx = [0]
        val_idx = 1
    x_train, x_test, x_val, labels_train, labels_test, labels_val = load_data_nfolds(args.dataset, feat_dim, num_const, train_idx, val_idx)

    #Get pretrained model
    if args.arch == "TransformerEncoder":
        #Transformer + DINO head architecture args
        transformer_args_standard = dict(
        input_dim=3, 
        model_dim=64, 
        output_dim=64, 
        embed_dim=32,
        n_heads=8, 
        dim_feedforward=256, 
        n_layers=4,
        hidden_dim_dino_head=256,
        bottleneck_dim_dino_head=64,
        pos_encoding = False,
        use_mask = False,
        mode='flatten',
        dropout=0,
        )
        transformer_args_small = dict(
        input_dim=3, 
        model_dim=6, 
        output_dim=64, 
        n_heads=3, 
        dim_feedforward=64, 
        n_layers=4,
        hidden_dim_dino_head=64,
        bottleneck_dim_dino_head=32,
        pos_encoding=True,
        use_mask=False,
    )
        #embed_dim = 64
        #embed_dim=6
        embed_dim = transformer_args_standard["embed_dim"]
        backbone = TransformerEncoder(**transformer_args_standard)
    elif args.arch == "TransformerEncoder_JetClass":
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
        use_mask = False,
    )
        embed_dim = transformer_args_jetclass["embed_dim"]
        backbone = TransformerEncoder(**transformer_args_jetclass)

    elif args.arch == "NoEmbedding":
        embed_dim = 57
        backbone = Identity()
    else: warnings.warn("Model architecture is not listed")

    #Load state_dict of embedding and freeze the layers
    state_dict = torch.load(args.pretrained, map_location='cpu')
    backbone.load_state_dict(state_dict=state_dict)
    
    head = nn.Linear(embed_dim, args.num_classes)
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()

    if args.type == "freeze":
        backbone.requires_grad_(False)
        head.requires_grad_(True)
    
    backbone.to(device=device)
    head.to(device=device)
    summary(backbone, input_size=(19,3))
    summary(head, input_size=(embed_dim,))

    #For semi-supervised fine-tuning create {percent} of the original x_train
    if args.type == "finetune":
        idx_percent = np.random.choice(labels_train.shape[0], size = int(args.percent/100*labels_train.shape[0]) ,replace=False)
        x_train = x_train[idx_percent]
        labels_train = labels_train[idx_percent]


    #Dataloaders
    train_data_loader = DataLoader(
        TorchCLDataset(x_train, labels_train, device),
        batch_size=args.batch_size,
        shuffle=True)
    print("Length of the train Dataloader: ",len(train_data_loader))
    val_data_loader = DataLoader(
        TorchCLDataset(x_val, labels_val, device),
        batch_size=args.batch_size,
        shuffle=False)
    print("Length of the val Dataloader: ",len(val_data_loader))
    test_data_loader = DataLoader(
        TorchCLDataset(x_test, labels_test, device),
        batch_size=args.batch_size,
        shuffle=False)
    print("Length of the test Dataloader: ",len(test_data_loader))
    #Loss function set to the standard CrossEntropyLoss
    criterion = nn.CrossEntropyLoss().to(device=device)
    
    param_groups = [dict(params=head.parameters())]
    if args.type == "finetune":
        param_groups.append(dict(params=backbone.parameters()))
    optimizer = torch.optim.SGD(param_groups, lr=1e-3, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    start_time = time.time()
    for epoch in range(args.epochs):
        if args.type == "freeze":
            backbone.eval()
            head.eval()
            freq = 500
        elif args.type == "finetune":
            backbone.train()
            head.train()
            freq = int(500/100*args.percent)
        else:
            assert False

        #Train
        for step, (data, target) in enumerate(train_data_loader, start = epoch*len(train_data_loader)):
            target = target.long().to(device)
            #output = head(backbone.representation(augmentations.naive_masking(data, device=device, rand_number=0))) #Train with the same augmentations as the contrastive objective
            output = head(backbone.representation(data.to(device), data.to(device)))
            loss = criterion(output, target.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % freq == 0:
                lr = scheduler.get_last_lr()[0]
                stats = dict(
                    epoch = epoch,
                    step = step,
                    learning_rate = lr,
                    loss = loss.item(),
                    time = int(time.time()-start_time),
                    )
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
        #Evaluate
        backbone.eval()
        head.eval()
        top1 = AverageMeter("Acc@1")
        top5 = AverageMeter("Acc@5")
        with torch.no_grad():
            for data, target in val_data_loader:
                target = target.to(device)
                output = head(backbone.representation(data.to(device), data.to(device)))
                acc1, acc5, = accuracy(output, target, topk=(1,2))
                top1.update(acc1[0].item(), data.size(0))
                top5.update(acc5[0].item(), data.size(0))

        best_acc.top1 = max(best_acc.top1, top1.avg)
        best_acc.top5 = max(best_acc.top5, top5.avg)
        stats = dict(
                epoch=epoch,
                acc1=top1.avg,
                acc5=top5.avg,
                best_acc1=best_acc.top1,
                best_acc5=best_acc.top5,
            )
        print(json.dumps(stats))
        print(json.dumps(stats), file=stats_file)

        scheduler.step()

    #Save the model for inference
    if args.type == 'freeze':
        #Just save the head
        torch.save(head.state_dict(), args.head_name)
        print(f"Classification head successfully saved!")
    elif args.type == 'finetune':
        #Save the finetuned backbone + the classification head
        torch.save(head.state_dict(), args.head_name)
        torch.save(backbone.state_dict(), str(args.percent) + '_' + args.backbone_name)
        print(f"Classification head + finetuned backbone (with {args.percent}% labeled data) successfully saved!")
    
    #Print the penultimate performance also on the test dataset to confirm the changes made based on the accuracies of the validation dataset
    top1, top5 = test_accuracy(test_data_loader, backbone, head, device)
    stats_test = dict(
        acc1_on_testset = top1,
        acc5_on_testset = top5,
    )
    print(json.dumps(stats_test), file=stats_file)


#Copied straight from: https://github.com/facebookresearch/vicreg/blob/main/evaluate.py
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
#Confirm the accuracy on the test set
def test_accuracy(dataloader, backbone, head, device):
    #Evaluate
    backbone.eval()
    head.eval()
    best_acc = argparse.Namespace(top1=0, top5=0)
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")
    with torch.no_grad():
        for data, target in dataloader:
            output = head(backbone.representation(data.to(device), data.to(device)))
            target = target.to(device)
            acc1, acc5, = accuracy(output, target, topk=(1,2))
            top1.update(acc1[0].item(), data.size(0))
            top5.update(acc5[0].item(), data.size(0))
    best_acc.top1 = max(best_acc.top1, top1.avg)
    best_acc.top5 = max(best_acc.top5, top5.avg)
    print(f"On the test dataset we get the best accuracies for top-1 {best_acc.top1} and top-5 {best_acc.top5}")
    return best_acc.top1, best_acc.top5

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
    
def load_data(data_dir='', feat_dim=3, num_const=19):
    dataset = np.load(data_dir)
    x_train, labels_train = dataset['x_train'].reshape(-1,num_const,feat_dim), dataset['labels_train'].reshape(-1)
    x_test, labels_test = dataset['x_test'].reshape(-1,num_const,feat_dim), dataset['labels_test'].reshape(-1)
    x_val, labels_val = dataset['x_val'].reshape(-1,num_const,feat_dim), dataset['labels_val'].reshape(-1)
    return x_train, x_test, x_val, labels_train, labels_test, labels_val

def load_data_nfolds(data_dir='', feat_dim=3, num_const=19, train_idx=[0], val_idx=1):
    with h5py.File("/eos/user/k/kmetzger/test/dino/"+f'{data_dir}', 'r') as f:
        train_fold = np.concatenate([np.array(f[f'x_train_fold_{idx}'][...]) for idx in train_idx], axis=0)
        labels_train_fold = np.concatenate([np.array(f[f'labels_train_fold_{idx}'][...]) for idx in train_idx], axis=0)
        val_fold = np.array(f[f'x_train_fold_{val_idx}'][...])
        labels_val_fold = np.array(f[f'labels_train_fold_{val_idx}'][...])
        x_test = np.array(f['x_test'][...])
        labels_test = np.array(f['labels_test'][...])
    x_train, labels_train = train_fold.reshape(-1,num_const,feat_dim), labels_train_fold.reshape(-1)
    x_test, labels_test = x_test.reshape(-1,num_const,feat_dim), labels_test.reshape(-1)
    x_val, labels_val = val_fold.reshape(-1,num_const,feat_dim), labels_val_fold.reshape(-1)
    return x_train, x_test, x_val, labels_train, labels_test, labels_val

class TorchCLDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, features, labels, device):
        'Initialization'
        self.device = device
        #self.mean = np.mean(features)
        #self.std = np.std(features)
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

if __name__ == "__main__":
    main()