import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import losses
from models import CVAE, SimpleDense, DeepSets, Identity, SimpleDense_small, SimpleDense_JetClass
from train_with_signal import TorchCLDataset
import augmentations
import argparse
from pathlib import Path
import warnings
import torch.nn as nn
from torchsummary import summary
import time
import json
import sys
from tqdm import tqdm
from sklearn.decomposition import PCA
from plotter import corner_plot
import itertools

dict_labels_names = {'W-boson': 0, 
                     'QCD Multijet': 1, 
                     'Z-boson': 2, 
                     'ttbar': 3, 
                     'leptoquark': 4, 
                     'ato4l': 5, 
                     'hChToTauNu': 6, 
                     'hToTauTau': 7
                             }
dict_labels_names_JetClass = {
                    'QCD-background': 0, 
                    'hToBB': 1, 
                    'hToCC': 2, 
                    'hToGG': 3, 
                    'hTo4q': 4, 
                    'hTolvqq': 5, 
                    'tTobqq': 6, 
                    'tToblv': 7,
                    'WToqq': 8,
                    'ZToqq': 9,
}
dict_labels_color = {0: 'teal', 
                     1: 'lightseagreen', 
                     2: 'springgreen', 
                     3: 'darkgreen', 
                     4: 'lightcoral', 
                     5: 'maroon', 
                     6: 'fuchsia', 
                     7: 'indigo'
                             }
dict_labels_color_JetClass = {
                    0: 'teal', 
                    1: 'lightseagreen', 
                    2: 'springgreen', 
                    3: 'darkgreen', 
                    4: 'lightcoral', 
                    5: 'maroon', 
                    6: 'fuchsia', 
                    7: 'indigo',
                    8: 'orange',
                    9: 'brown'
                             }

def get_gradient(signal, model, classification_head, drive_path, device):
    #Check signal
    num = dict_labels_names[signal]
    #Data
    data = np.load(drive_path+'dataset_background_signal.npz')
    input_data = data['x_test']
    input_labels = data['labels_test']
    #Model
    model.eval()
    classification_head.eval()
    for param in classification_head.parameters():
        print(param)
    #Turn input into dataloader (which requires grad)
    data_loader = DataLoader(
        TorchCLDataset(input_data, input_labels, device),
        batch_size=1024,
        shuffle=False)
    gradients = []
    #Embed the points from the dataloader
    with torch.no_grad():
        output = np.concatenate([model.representation(data.to(device)).cpu().detach().numpy() for (data, label) in tqdm(data_loader)], axis=0)
    #Loop over embedded points and save the gradients (from the classifier_head) for the corresponding signal
    input_data = output[:50000]
    input_labels = input_labels[:50000]
    for point in tqdm(input_data):
        input = torch.from_numpy(point).to(dtype=torch.float32, device=device).requires_grad_() #Record gradients of the input
        logits = classification_head(input)[num]
        logits.backward()
        gradients.append(input.grad)

    print(np.shape(np.array(gradients)))
    return input_data, input_labels, gradients

def get_gradient_JetClass(signal, model, classification_head, drive_path, device):
    #Check signal
    num = dict_labels_names_JetClass[signal]
    #Data
    data = np.load(drive_path+'jetclass_dataset/JetClass_background_signal_reshaped.npz')
    input_data = data['x_test'].reshape(-1,512)
    input_labels = data['labels_test']
    #Model
    model.eval()
    classification_head.eval()
    for param in classification_head.parameters():
        print(param)
    #Turn input into dataloader (which requires grad)
    data_loader = DataLoader(
        TorchCLDataset(input_data, input_labels, device),
        batch_size=1024,
        shuffle=False)
    gradients = []
    #Embed the points from the dataloader
    with torch.no_grad():
        output = np.concatenate([model.representation(data.to(device)).cpu().detach().numpy() for (data, label) in tqdm(data_loader)], axis=0)
    #Loop over embedded points and save the gradients (from the classifier_head) for the corresponding signal
    input_data = output
    input_labels = input_labels
    for point in tqdm(input_data):
        input = torch.from_numpy(point).to(dtype=torch.float32, device=device).requires_grad_() #Record gradients of the input
        logits = classification_head(input)[num]
        logits.backward()
        gradients.append(input.grad)

    print(np.shape(np.array(gradients)))
    return input_data, input_labels, gradients

def standardize_gradients(gradients):
    std = np.std(gradients, axis=1)
    std_gradients = gradients/std[:, np.newaxis]
    return std_gradients

def get_subspace_eigenvectors(std_gradients):
    #Perform a PCA decomposition (components) in order to get the eigen decomposition of the covariance matrix
    pca = PCA()
    pca.fit(std_gradients)
    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_
    # print(np.shape(eigenvectors))
    # print(eigenvectors)
    # print(np.shape(eigenvalues))
    # print(eigenvalues)
    # print(pca.explained_variance_ratio_)
    return eigenvectors, eigenvalues
    
def visualize_subspace_first_eigenvector(input_data, input_labels, eigenvectors, signal):
    #Do a cornerplot of the input_data (which is embedded) and the corresponding eigenvectors
    dimensions = list(itertools.combinations_with_replacement(range(np.shape(input_data)[1]), 2))
    for dims in np.array(dimensions):
        dim_x = dims[0]
        dim_y = dims[1]
        if dim_x != dim_y:
            #First plot dimension 0, 1
            fig, (ax1, ax2) = plt.subplots(1,2)
            fig.supxlabel(f'Dimension {dim_x}')
            fig.supylabel(f'Dimension {dim_y}')
            #colors = np.array([dict_labels_color[int(x)] for x in input_labels])
            background_mask = (input_labels<4).reshape(-1)
            #colors[background_mask] = 'grey'
            signal_mask = (input_labels == dict_labels_names[signal]).reshape(-1)
            #mask = np.logical_or(background_mask, signal_mask)
            ax1.scatter(input_data[:,dim_x][background_mask], input_data[:,dim_y][background_mask], c='grey', s=.5)
            ax1.scatter(input_data[:,dim_x][signal_mask], input_data[:,dim_y][signal_mask], c=dict_labels_color[dict_labels_names[signal]], s=.5)
            first_eigen = eigenvectors[0]
            ax2.arrow(0, 0, first_eigen[dim_x], first_eigen[dim_y], head_width=.01, color=dict_labels_color[dict_labels_names[signal]])
            plt.show()

def visualize_std_gradients_marginally(input_data, input_labels, std_gradients):
    dimensions = list(itertools.combinations_with_replacement(range(np.shape(input_data)[1]), 2))
    for dims in np.array(dimensions):
        dim_x = dims[0]
        dim_y = dims[1]
        if dim_x != dim_y:
            plt.figure()
            plt.xlabel(f'Dimension {dim_x}')
            plt.ylabel(f'Dimension {dim_y}')
            print(np.shape(std_gradients[:,dim_x]))
            print(std_gradients[:,dim_x])
            plt.scatter(std_gradients[:,dim_x], std_gradients[:,dim_y],s=.5)
            plt.show()


def main(runs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    signal = 'hChToTauNu'
    latent_dim = 6
    num_classes = 8
    model = SimpleDense_small(latent_dim)
    head = nn.Linear(latent_dim, num_classes)
    model.load_state_dict(torch.load(f'output/{runs}/vae.pth', map_location=torch.device(device)))
    head.load_state_dict(torch.load(f'output/{runs}/head.pth', map_location=torch.device(device)))

    input_data, input_labels, gradients = get_gradient(signal=signal,model=model, classification_head=head, drive_path='C:\\Users\\Kyle\\OneDrive\\Transfer Master project\\orca_fork\\cl4ad\\cl\\cl\\', device=device)
    std_gradients = standardize_gradients(gradients)
    #print(gradients)
    eigenvectors, eigenvalues = get_subspace_eigenvectors(std_gradients)
    #print(std_gradients)
    visualize_subspace_first_eigenvector(input_data, input_labels, eigenvectors, signal)
    #visualize_std_gradients_marginally(input_data, input_labels, std_gradients)

    print(f"Norm of the first eigenvector: {np.linalg.norm(eigenvectors[0])}")
    print(f"First eigenvector in {latent_dim}-dimensional embedding space for the signal {signal}: {eigenvectors[0]} with eigenvalue: {eigenvalues[0]}")

if __name__ == '__main__':
    main('runs53')