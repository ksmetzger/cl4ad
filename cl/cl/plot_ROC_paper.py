""" Compute and plot ROC curves for the linear classifier trained ontop of the contrastive embeddings 
    developed for the ADC_Delphes dataset, supporting the paper: https://arxiv.org/abs/2502.15926.
    
    The multiclass classification problem distinguishes four background collision event classes,
    as listed in the color dictionary.
"""
#Define color and name dicts
#Linear layer is only trained on the following labeled background classes
dict_labels_color = {0: 'teal', 
                     1: 'lightseagreen', 
                     2: 'springgreen', 
                     3: 'darkgreen', 

                             }
dict_labels_names = {0: 'W-boson', 
                     1: 'QCD Multijet', 
                     2: 'Z-boson', 
                     3: 'ttbar', 
 
                             }

import os 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as mlines
from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA 
from sklearn.metrics import roc_curve, auc
import argparse
import torch
from pathlib import Path
from models import SimpleDense, Identity, SimpleDense_small, SimpleDense_JetClass, DeepSets, CVAE, CVAE_JetClass, SimpleDense_ADC
from transformer import TransformerEncoder
from torch.utils.data import DataLoader, Dataset
from train_with_signal import TorchCLDataset
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
from PIL import Image
import re
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


def classification_score(backbone_name, backbone_type, head_name, input_data, input_labels, device=None, embed_dim=4, num_classes=4, mode='all'):
    '''
    Get the classification scores (softmax class probabilities) needed for computing the ROC curves
    If  mode = 'roc' output only (softmax) probabilities for  background (class 0) vs anomaly (class 1)
        mode = 'all' output (softmax) probabilities for all the classes (4 backgrounds)
    '''
    if device == None:
        device = torch.device('cpu')
    else: 
        device = device

    #Import model for embedding
    if backbone_type == 'NoEmbedding':
        backbone = Identity()
        embed_dim = 57
    elif backbone_type == "MLP":
        backbone = SimpleDense_ADC(latent_dim=embed_dim).to(device)
    elif backbone_type == "Transformer":
        transformer_args_standard = dict(
        input_dim=3, 
        model_dim=64, 
        output_dim=64, 
        embed_dim=embed_dim,
        n_heads=8, 
        dim_feedforward=256, 
        n_layers=4,
        hidden_dim_dino_head=256,
        bottleneck_dim_dino_head=64,
        pos_encoding = True,
        use_mask = False,
        mode='cls',
        dropout=0.025,
        )
        backbone = TransformerEncoder(**transformer_args_standard).to(device)
    else:
        raise ValueError("Non-listed embedding architecture")

    backbone.load_state_dict(torch.load(backbone_name, map_location=torch.device(device)))

    head = nn.Linear(embed_dim, num_classes).to(device)
    head.load_state_dict(torch.load(head_name, map_location=torch.device(device)))
    backbone.eval()
    head.eval()

    #Get logits trained from linear_evaluation.py
    data_loader = DataLoader(
        TorchCLDataset(input_data, input_labels, device),
        batch_size=1024,
        shuffle=False)
    with torch.no_grad():
        if backbone_type == "Transformer":
            softmax = np.concatenate([F.softmax(head(backbone.representation(data.reshape(-1,19,3), data.reshape(-1,19,3))), dim=1).cpu().detach().numpy() for (data, label) in data_loader], axis=0)
        elif backbone_type == "MLP":
            softmax = np.concatenate([F.softmax(head(backbone.representation(data)), dim=1).cpu().detach().numpy() for (data, label) in data_loader], axis=0)
        else:
            assert(False)
    #Get the softmax predictions
    if mode == 'roc':
        predictions = np.array([np.sum(softmax[:,:4], axis=1), np.sum(softmax[:,4:], axis=1)]).T
    elif mode == 'all':
        predictions = softmax
    return predictions

def plot_ROC_multiclass(predictions, labels, filename, title, folder='ROC_curves_paper', num_classes = 4):
    '''
    Plots ROC Curves for the linear evaluation on-top of the contrastive embedding in one-vs-rest (OvR) fashion
    predictions: array with target scores for each class
    '''
    print("Plotting ROC Plots!")   
    plt.figure()

    fpr, tpr, roc_auc = {}, {}, {}
    # Defines true class. Looping over all the classes, the target class has label 1, the rest label 0.
    for idx in range(num_classes):
        true = np.empty_like(labels)
        true[(labels == idx)] = 1
        true[~(labels == idx)] = 0
        prediction = predictions[:,idx]
        # Calculates ROC properties and plots on same plt
        false_pos_rate, true_pos_rate, threshold = roc_curve(true, prediction)
        area_under_curve = auc(false_pos_rate, true_pos_rate)

        fpr[idx], tpr[idx], roc_auc[idx] = false_pos_rate, true_pos_rate, area_under_curve

        plt.plot(false_pos_rate, true_pos_rate, label=f'{dict_labels_names[idx]} AUC: {area_under_curve*100:.2f}%', 
                linewidth=2, color=dict_labels_color[idx], ds='steps')
        
    
    #Add micro and macro average ROC curves
    #Binarize labels for all classes
    #Micro: Sample weighted
    Y_true = label_binarize(labels, classes=list(range(num_classes)))
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_true.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average AUC: {roc_auc["micro"]*100:.2f}%',
             linestyle='--', linewidth=2, color='black')

    #Macro: Class weighted
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    roc_auc["macro"] = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr,
             label=f'Macro-average AUC: {roc_auc["macro"]*100:.2f}%',
             linestyle='--', linewidth=2, color='navy')
    
    plt.title(title, size=13)
    plt.xlabel("FPR", size=12)
    plt.ylabel("TPR", size=12)
    plt.grid()
    # Creates x=y line to compare model against random classification performance
    plt.plot(np.linspace(0, 1),np.linspace(0, 1), ':', color='0.75', linewidth=2)
    plt.legend(loc='lower right', frameon=False,prop={'size': 10})
    
    # Saves plot and reports success 
    subfolder = os.path.dirname(__file__)
    subfolder = os.path.join(subfolder, folder)
    os.makedirs(subfolder, exist_ok=True)
    file_path = os.path.join(subfolder, filename)
    plt.savefig(file_path) 
    plt.close('all')
    print(f"ROC Plot multiclass saved at '{filename}'")


def main(args):
    #Load the data
    with h5py.File(f'ADC_Delphes_original_divisions.hdf5', 'r') as f:
        x_test = np.array(f['x_test'][...])
        labels_test = np.array(f['labels_test'][...])
        val_fold = np.array(f[f'x_train_fold_{args.val_fold}'][...])
        labels_val_fold = np.array(f[f'labels_train_fold_{args.val_fold}'][...])

    #Get the predictions (classification scores)
    predictions = classification_score(f'output/{args.backbone_name}/vae{args.val_fold}.pth', args.backbone_type, f'output/{args.backbone_name}/head{args.val_fold}.pth', val_fold, labels_val_fold, embed_dim=args.embed_dim, mode='all')
    #Plot the ROC curves
    plot_ROC_multiclass(predictions, labels_val_fold, title=f'ROC curves for {args.supervision} {args.embed_dim}D SimCLR {args.backbone_type} embedding', filename=f'OvR_ROCcurve_{args.backbone_type}_{args.supervision}_dim{args.embed_dim}.pdf', num_classes=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for computing ROC curves for the contrastive embedding linear evaluation")
    parser.add_argument("--backbone_name",default="runs1", type=Path, help="path to pretrained backbone weights")
    parser.add_argument("--backbone_type", default="MLP", type=str, help="pretrained backbone architecture used")
    parser.add_argument("--embed_dim", default=4, type=int, help="dimensionality of the contrastive embedding")
    parser.add_argument("--supervision", default="supervised", type=str, choices=("supervised", "self_supervised"))
    parser.add_argument("--val_fold", default=0, type=int, help="Which fold was withheld during training and should now be used for plotting")
    args = parser.parse_args()
    main(args)