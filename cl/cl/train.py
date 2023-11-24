import os
import sys
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    TensorDataset
    )
import models
import losses

class Dataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, features, labels):
        'Initialization'
        self.features = features
        self.labels = labels

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.features)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.features[index]
        y = self.labels[index]

        return X, y

def main(args):
    '''
    Infastructure for training CVAE (background specific and with anomalies)
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    features_dataset = np.load(args.dataset_filename)

    features_train = torch.from_numpy(features_dataset['x_train']).to(dtype=torch.float32, device=device)
    features_test =  torch.from_numpy(features_dataset['x_test']).to(dtype=torch.float32, device=device)
    features_val =  torch.from_numpy(features_dataset['x_val']).to(dtype=torch.float32, device=device)
    labels_train =  torch.from_numpy(features_dataset['labels_train']).to(dtype=torch.float32, device=device)
    labels_test = torch.from_numpy(features_dataset['labels_test']).to(dtype=torch.float32, device=device)
    labels_val = torch.from_numpy(features_dataset['labels_val']).to(dtype=torch.float32, device=device)

    train_set = Dataset(features_train, labels_train)

    dataloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False)

    model = models.CVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = losses.SimCLRLoss()

    for epoch in range(args.epochs):
        train_loss = 0
        for batch_idx, data in enumerate(dataloader):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 1000 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(dataloader.dataset)} ({100*batch_idx / len(dataloader):.0f}%)]\nLoss: {loss.item():.6f}')

        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(dataloader.dataset):.4f}')

    torch.save(model.state_dict(), args.encoder_name)


if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()

    # If not using full data, name of smaller dataset to pull from
    parser.add_argument('dataset_filename', type=str)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--loss-temp', type=float, default=0.07)
    parser.add_argument('--encoder-name', type=str, default='output/vae.pth')

    args = parser.parse_args()
    main(args)
