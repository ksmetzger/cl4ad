import os
import sys
import numpy as np
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from models import CVAE
from dataset import CLDataset

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(args):

    def create_embedding(features_name, labels_name=None):


        dataset = CLBackgroundDataset(args.background_dataset, args.background_ids,
            preprocess=args.scaling_filename,
            divisions=[0.30, 0.30, 0.20, 0.20]
        )
        dataset.report_specs()

        train_data_loader = DataLoader(
            TorchCLDataset(dataset.x_train, dataset.labels_train, device),
            batch_size=args.batch_size,
            shuffle=False)

        val_data_loader = DataLoader(
            TorchCLDataset(dataset.x_val, dataset.labels_val, device),
            batch_size=args.batch_size,
            shuffle=False)

        features_dataset = np.load(args.dataset_filename)

        features_train = torch.from_numpy(features_dataset[features_name]).to(dtype=torch.float32, device=DEVICE)
        labels_train = None if not labels_name else features_dataset[labels_name]
        train_set = CLBackgroundDataset(args.dataset_filename, labels_name, preprocess=True, n_events=-1, divisions=[0.30, 0.30, 0.20, 0.20])\
            if labels is not None else CLSignalDataset(data_filename, n_events=-1, preprocess=None)

        dataloader = DataLoader(
            train_set,
            batch_size=1024,
            shuffle=False)

        model = CVAE().to(DEVICE)
        model.load_state_dict(torch.load(args.pretrained_model))
        model.eval()

        for batch_idx, data in enumerate(dataloader):
            if batch_idx==0:
                embedding = model.representation(data[0]).cpu().detach().numpy()
            else:
                embedding = np.concatenate((embedding, model.representation(data[0]).cpu().detach().numpy() ))

        return features_train.cpu().detach().numpy(), embedding, labels_train

    datasets = dict()
    if anomalies:
        for features_name in ['leptoquark', 'ato4l', 'hChToTauNu', 'hToTauTau']:
            embedding_name = features_name.replace('x_', 'embedding_')
            datasets[features_name], datasets[embedding_name] = \
                create_embedding(features_name)
    else:
        for features_name in ['x_train','x_test','x_val']:
            labels_name = features_name.replace('x_', 'labels_')
            embedding_name = features_name.replace('x_', 'embedding_')
            datasets[features_name], datasets[embedding_name], datasets[labels_name] = \
                create_embedding(features_name, labels_name)

    np.savez(args.output_filename, **datasets)


    def save(self, filename, model):
        # Create and save new .npz with extracted features. Reports success
        self.scaled_dataset['embedding_train'] = model.representation(
                torch.from_numpy(self.scaled_dataset['x_train']
                    ).to(dtype=torch.float32, device=self.device)).cpu().detach().numpy()
        self.scaled_dataset['embedding_test'] = model.representation(
                torch.from_numpy(self.scaled_dataset['x_test']
                    ).to(dtype=torch.float32, device=self.device)).cpu().detach().numpy()
        self.scaled_dataset['embedding_val'] = model.representation(
                torch.from_numpy(self.scaled_dataset['x_val']
                    ).to(dtype=torch.float32, device=self.device)).cpu().detach().numpy()

        np.savez(filename, **self.scaled_dataset)
        print(f'{filename} successfully saved')


if __name__ == '__main__':
    #Parses terminal command
    parser = ArgumentParser()
    parser.add_argument('dataset_filename', type=str)

    parser.add_argument('--pretrained-model', type=str)
    parser.add_argument('--output-filename', type=str)
    parser.add_argument('--anomalies', action='store_true')

    args = parser.parse_args()
    main(args)

