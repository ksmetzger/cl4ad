import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
import open_world_delphes as datasets
from utils import cluster_acc, AverageMeter, entropy, MarginLoss, accuracy, TransformTwice
from sklearn import metrics
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle
import time
from models.models_cl import SimpleDense, SimpleDense_small, Identity, TransformerEncoder, transformer_args_standard
from torchsummary import summary


def train(args, model, device, train_label_loader, train_unlabel_loader, optimizer, m, epoch, tf_writer):
    model.train()
    #Set frozen parts of the model to eval() (for correct handling of dropout, batchnorm, ...)
    print(model[0])
    model[0].eval()

    bce = nn.BCELoss()
    #Change the strength of the adaptive uncertainty with a factor lambda_uncert
    lambda_uncert = 1.
    m = min(m, 0.5) * lambda_uncert
    ce = MarginLoss(m=-1*m)
    unlabel_loader_iter = cycle(train_unlabel_loader)
    bce_losses = AverageMeter('bce_loss', ':.4e')
    ce_losses = AverageMeter('ce_loss', ':.4e')
    entropy_losses = AverageMeter('entropy_loss', ':.4e')
    
    for batch_idx, ((x, x2), target) in enumerate(train_label_loader):
        
        ((ux, ux2), _) = next(unlabel_loader_iter)

        x = torch.cat([x, ux], 0)
        x2 = torch.cat([x2, ux2], 0)
        labeled_len = len(target)

        x, x2, target = x.to(device), x2.to(device), target.to(device)
        optimizer.zero_grad()
        output, feat = model(x.float())
        output2, feat2 = model(x2.float())
        prob = F.softmax(output, dim=1)
        prob2 = F.softmax(output2, dim=1)
        
        feat_detach = feat.detach()
        feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
        cosine_dist = torch.mm(feat_norm, feat_norm.t())
        labeled_len = len(target)

        pos_pairs = []
        target_np = target.cpu().numpy()
        
        # label part
        for i in range(labeled_len):
            target_i = target_np[i]
            idxs = np.where(target_np == target_i)[0]
            if len(idxs) == 1:
                pos_pairs.append(idxs[0])
            else:
                selec_idx = np.random.choice(idxs, 1)
                while selec_idx == i:
                    selec_idx = np.random.choice(idxs, 1)
                pos_pairs.append(int(selec_idx))

        # unlabel part
        unlabel_cosine_dist = cosine_dist[labeled_len:, :]
        vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
        pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
        pos_pairs.extend(pos_idx)
        
        pos_prob = prob2[pos_pairs, :]
        pos_sim = torch.bmm(prob.view(args.batch_size, 1, -1), pos_prob.view(args.batch_size, -1, 1)).squeeze()
        ones = torch.ones_like(pos_sim)
        bce_loss = bce(pos_sim, ones)
        ce_loss = ce(output[:labeled_len], target)
        entropy_loss = entropy(torch.mean(prob, 0))
        
        #Try weighing the pairwise objective (bce_loss) more
        pairwise_weight = 1.
        loss = - entropy_loss + ce_loss + pairwise_weight* bce_loss

        bce_losses.update(bce_loss.item(), args.batch_size)
        ce_losses.update(ce_loss.item(), args.batch_size)
        entropy_losses.update(entropy_loss.item(), args.batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(batch_idx,"train")

    tf_writer.add_scalar('loss/bce', bce_losses.avg, epoch)
    tf_writer.add_scalar('loss/ce', ce_losses.avg, epoch)
    tf_writer.add_scalar('loss/entropy', entropy_losses.avg, epoch)
    #Print the total loss after each epoch
    print("Loss: ", -entropy_losses.avg + ce_losses.avg + bce_losses.avg)

def test(args, model, labeled_num, device, test_loader, epoch, tf_writer):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    
    
    #Initialize array to store the softmax prob output (8,)
    prob_softmax = np.empty((0,8))
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            output, _ = model(x.float())
            prob = F.softmax(output, dim=1)
            prob_softmax = np.append(prob_softmax,prob.cpu().numpy(),axis=0) #Append the softmax prob output
            conf, pred = prob.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
    targets = targets.astype(int)
    preds = preds.astype(int)

    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    print(np.shape(preds),"fff",np.shape(targets))
    overall_acc = cluster_acc(preds, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
    #Add adjusted mutual info score
    unseen_nmi_adjusted = metrics.adjusted_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
    mean_uncert = 1 - np.mean(confs)
    print('Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(overall_acc, seen_acc, unseen_acc))
    tf_writer.add_scalar('acc/overall', overall_acc, epoch)
    tf_writer.add_scalar('acc/seen', seen_acc, epoch)
    tf_writer.add_scalar('acc/unseen', unseen_acc, epoch)
    tf_writer.add_scalar('nmi/unseen', unseen_nmi, epoch)
    tf_writer.add_scalar('nmi-adjusted/unseen', unseen_nmi_adjusted, epoch)
    tf_writer.add_scalar('uncert/test', mean_uncert, epoch)
    
    
    
    return mean_uncert , [targets, preds, confs] , prob_softmax#, latent


def main():
    parser = argparse.ArgumentParser(description='orca')
    parser.add_argument('--milestones', nargs='+', type=int, default=[10, 20]) #Changed from [140,180] to [50,90]
    parser.add_argument('--dataset', default='cifar100', help='dataset setting')
    parser.add_argument('--labeled-num', default=4, type=int)
    parser.add_argument('--labeled-ratio', default=0.5, type=float)
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--exp_root', type=str, default='./results/')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size')
    parser.add_argument('--size', type=str, default='large')
    parser.add_argument('--runs', type=str, default='runs59')
    parser.add_argument('--latent-dim', type=int, default=48)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--embedding-type', type=str, choices=('mlp', 'dino_transformer'), default='mlp')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("Is cuda available: ", args.cuda)
    device = torch.device("cuda" if args.cuda else "cpu")

    args.savedir = os.path.join(args.exp_root, args.name)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)


    if args.dataset == 'background_like_cifar':
        train_label_set = datasets.BACKGROUND(root='./datasets',labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download = False, transform=TransformTwice(datasets.dict_transform['cifar_train_kyle']))
        train_unlabel_set = datasets.BACKGROUND(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=False, transform=TransformTwice(datasets.dict_transform['cifar_train_kyle']), unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.BACKGROUND(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=False, transform=datasets.dict_transform['cifar_test_kyle'], unlabeled_idxs=train_label_set.unlabeled_idxs)
        num_classes = 4
    elif args.dataset == 'background_with_signal':
        train_label_set = datasets.BACKGROUND_SIGNAL(root='./datasets',datatype='train_labeled', transform=TransformTwice(datasets.dict_transform['cifar_train_kyle']))
        train_unlabel_set = datasets.BACKGROUND_SIGNAL(root='./datasets',datatype='train_unlabeled', transform=TransformTwice(datasets.dict_transform['cifar_train_kyle']))
        test_set = datasets.BACKGROUND_SIGNAL(root='./datasets',datatype='test', transform=datasets.dict_transform['cifar_test_kyle'])
        num_classes = 8
        args.labeled_num = 4
    elif args.dataset == 'background_with_signal_cvae':
        train_label_set = datasets.BACKGROUND_SIGNAL_CVAE(root='./datasets',datatype='train_labeled', transform=TransformTwice(datasets.dict_transform['cifar_train_kyle_cvae']))
        train_unlabel_set = datasets.BACKGROUND_SIGNAL_CVAE(root='./datasets',datatype='train_unlabeled', transform=TransformTwice(datasets.dict_transform['cifar_train_kyle_cvae']))
        test_set = datasets.BACKGROUND_SIGNAL_CVAE(root='./datasets',datatype='test', transform=datasets.dict_transform['cifar_test_kyle_cvae'])
        num_classes = 8
        args.labeled_num = 4
    elif args.dataset == 'background_with_signal_cvae_latent':
        train_label_set = datasets.BACKGROUND_SIGNAL_CVAE_LATENT(root='./datasets',datatype='train_labeled', transform=TransformTwice(datasets.dict_transform['cifar_train_kyle_cvae']))
        train_unlabel_set = datasets.BACKGROUND_SIGNAL_CVAE_LATENT(root='./datasets',datatype='train_unlabeled', transform=TransformTwice(datasets.dict_transform['cifar_train_kyle_cvae']))
        test_set = datasets.BACKGROUND_SIGNAL_CVAE_LATENT(root='./datasets',datatype='test', transform=datasets.dict_transform['cifar_test_kyle_cvae'])
        num_classes = 8
        args.labeled_num = 4
    elif args.dataset == 'background_with_signal_dense_latent':
        train_label_set = datasets.BACKGROUND_SIGNAL_DENSE_LATENT(root='./datasets',datatype='train_labeled', transform=TransformTwice(datasets.dict_transform['cifar_train_kyle_cvae']))
        train_unlabel_set = datasets.BACKGROUND_SIGNAL_DENSE_LATENT(root='./datasets',datatype='train_unlabeled', transform=TransformTwice(datasets.dict_transform['cifar_train_kyle_cvae']))
        test_set = datasets.BACKGROUND_SIGNAL_DENSE_LATENT(root='./datasets',datatype='test', transform=datasets.dict_transform['cifar_test_kyle_cvae'])
        num_classes = 8
        args.labeled_num = 4
    elif args.dataset == 'background_with_signal_cvae_latent_pytorch':
        train_label_set = datasets.BACKGROUND_SIGNAL_CVAE_LATENT_PYTORCH(root='./datasets',datatype='train_labeled', transform=TransformTwice(datasets.dict_transform['cifar_train_kyle_cvae']))
        train_unlabel_set = datasets.BACKGROUND_SIGNAL_CVAE_LATENT_PYTORCH(root='./datasets',datatype='train_unlabeled', transform=TransformTwice(datasets.dict_transform['cifar_train_kyle_cvae']))
        test_set = datasets.BACKGROUND_SIGNAL_CVAE_LATENT_PYTORCH(root='./datasets',datatype='test', transform=datasets.dict_transform['cifar_test_kyle_cvae'])
        num_classes = 8
        args.labeled_num = 4
    elif args.dataset == 'background_with_signal_inference_latent':
        num_classes = 8
        args.labeled_num = 4
        train_label_set = datasets.BACKGROUND_SIGNAL_INFERENCE_LATENT(root='./datasets',datatype='train_labeled', runs=args.runs, latent_dim=args.latent_dim, labeled_num=args.labeled_num, finetune=args.finetune, embedding_type=args.embedding_type, transform=TransformTwice(datasets.dict_transform['cifar_train_kyle_cvae']))
        train_unlabel_set = datasets.BACKGROUND_SIGNAL_INFERENCE_LATENT(root='./datasets',datatype='train_unlabeled', runs=args.runs, latent_dim=args.latent_dim, labeled_num=args.labeled_num, finetune=args.finetune, embedding_type=args.embedding_type, transform=TransformTwice(datasets.dict_transform['cifar_train_kyle_cvae']))
        test_set = datasets.BACKGROUND_SIGNAL_INFERENCE_LATENT(root='./datasets',datatype='test', runs=args.runs, latent_dim=args.latent_dim, labeled_num=args.labeled_num, finetune=args.finetune, embedding_type=args.embedding_type, transform=datasets.dict_transform['cifar_test_kyle_cvae'])
        
    else:
        warnings.warn('Dataset is not listed')
        return

    labeled_len = len(train_label_set)
    unlabeled_len = len(train_unlabel_set)
    labeled_batch_size = int(args.batch_size * labeled_len / (labeled_len + unlabeled_len))

    # Initialize the splits
    train_label_loader = torch.utils.data.DataLoader(train_label_set, batch_size=labeled_batch_size, shuffle=True, num_workers=1, drop_last=True) #Changed num_workers=1, as it sometimes gave an error.
    train_unlabel_loader = torch.utils.data.DataLoader(train_unlabel_set, batch_size=args.batch_size - labeled_batch_size, shuffle=True, num_workers=1, drop_last=True) #Changed num_workers=1, as it sometimes gave an error.
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=1) #Batch size changed from 100 to 512 for more data points (DELPHES), back to 100 as a test
    #Print the lengths of the dataloaders
    print('Len of train_label_loader: ',len(train_label_loader))
    print('Len of train_unlabel_loader: ',len(train_unlabel_loader))
    print('Len of test_label_loader: ',len(test_loader))

 
    #Choose appropriate model
    if args.dataset == 'background_with_signal_cvae':
        if args.size == 'large':
            model = models.CVAE_direct(num_classes=num_classes)
        elif args.size == 'simple':
            model = models.CVAE_direct_simple(num_classes=num_classes)
            #model = models.Dense_latent_simple(num_classes=num_classes, latent_dim=57)
    elif args.dataset == 'background_with_signal_cvae_latent':
        if args.size == 'large':
            model = models.CVAE_latent(num_classes=num_classes)
        elif args.size == 'simple':
            model = models.CVAE_latent_simple(num_classes=num_classes)
    elif args.dataset == 'background_with_signal_dense_latent' or args.dataset=='background_with_signal_inference_latent':
        if args.size == 'simple':
            model = models.Dense_latent_simple(num_classes=num_classes, latent_dim=args.latent_dim)
            #model = models.CVAE_latent(num_classes=num_classes)
        elif args.size == 'large':
            model = models.Dense_latent_large(num_classes=num_classes, latent_dim=args.latent_dim)
    else:
        warnings.warn('Model is not listed')
        return
    #If finetune, add embedding model to finetune layers with ORCA objective
    if args.finetune:
        #Load pretrained weights
        if args.embedding_type == 'mlp':
            drive_path = f'C:\\Users\\Kyle\\OneDrive\\Transfer Master project\\orca_fork\\cl4ad\\cl\\cl\\'
            if args.latent_dim == 48:
                finetune = SimpleDense(args.latent_dim)
            elif args.latent_dim == 6:
                finetune = SimpleDense_small(args.latent_dim)
            finetune.load_state_dict(torch.load(drive_path+f'output/{args.runs}/vae.pth', map_location=torch.device(device)))
            finetune.expander = nn.Identity()
        elif args.embedding_type == 'dino_transformer':
            drive_path = f'C:\\Users\\Kyle\\OneDrive\\Transfer Master project\\orca_fork\\cl4ad\\dino\\'
            finetune = TransformerEncoder(**transformer_args_standard)
            finetune.load_state_dict(torch.load(drive_path+f'output/{args.runs}/_teacher_dino_transformer.pth', map_location=torch.device(device)))
            finetune.dino_head = nn.Identity()
            #Freeze the model except for the MLP
            for param in finetune.parameters(): #First freeze everything
                param.requires_grad = False
            finetune.downsize.requires_grad_(True) #Unfreeze MLP part
            finetune.eval() #Set the model to eval
        else:
            assert(False)

        model = nn.Sequential(finetune, model)
        
        #Freeze every layer except for the last one of the embedding backbone (finetune)
        # for param in finetune.parameters():
        #     param.requires_grad = False
        # for param in finetune[-1].parameters():
        #     param.requires_grad = True
        #Watch out for batchnorm in eval vs train mode if freezing certain layers?

    model = model.to(device)
    model = model.float()


    #Print the name and parameters of the model
    for name, param in model.named_parameters():
        print(name)
        print(param)
    summary(model)
 
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Set the optimizer
    #Try out the Adam optimizer for the delphes data (no weight decay and momentum first)
    optimizer = optim.Adam(model.parameters() , lr=1e-3) 
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    tf_writer = SummaryWriter(log_dir=args.savedir)
    
    #Initialize arrays for saving output
    pred_conf_output = np.empty((int((args.epochs/5)+1),3, unlabeled_len)) #Save the class predictions and confidence (every 5th epoch)
    prob_softmax_output = np.empty((int((args.epochs/5)+1),unlabeled_len,8)) #Save the softmax prob (every 5th epoch)
    count = 0
    for epoch in range(args.epochs):
        mean_uncert, targets_preds_conf, prob_softmax = test(args, model, args.labeled_num, device, test_loader, epoch, tf_writer) #removed , latent for memory reasons 
        print("Mean uncertainty m: ", mean_uncert)
        train(args, model, device, train_label_loader, train_unlabel_loader, optimizer, mean_uncert, epoch, tf_writer)
        print('Epoch: ',epoch)

        if (epoch+1) % 5 == 0 or epoch == 0:
            pred_conf_output[count,...] = targets_preds_conf
            prob_softmax_output[count,...] = prob_softmax
            count += 1
        scheduler.step()
        
    #Get the date/time and save the latent/conf_pred output in a file
    timestr = time.strftime("%Y%m%d-%H%M%S")
    np.savez('latent/target_pred_conf_kyle'+timestr, target_pred_conf = pred_conf_output, prob_softmax = prob_softmax_output)
    
if __name__ == '__main__':
    main()
