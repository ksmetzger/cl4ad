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

    def create_embedding(features_name, labels_name):

        features_dataset = np.load(args.dataset_filename)

        features_train = torch.from_numpy(features_dataset[features_name]).to(dtype=torch.float32, device=DEVICE)
        labels_train = features_dataset[labels_name]
        train_set = CLDataset(features_train, labels_train)

        dataloader = DataLoader(
            train_set,
            batch_size=1024,
            shuffle=False)

        model = CVAE().to(DEVICE)
        model.load_state_dict(torch.load(args.pretrained_model))
        model.eval()

        for batch_idx, data in enumerate(dataloader):
            if batch_idx==0:
                embedding = model.representation(data[0])
            else:
                torch.cat((embedding, model.representation(data[0])))

        return features_train.cpu().detach().numpy(), embedding.cpu().detach().numpy(), labels_train

    datasets = dict()
    for features_name in ['x_train','x_test','x_val']:
        labels_name = features_name.replace('x_', 'labels_')
        embedding_name = features_name.replace('x_', 'embedding_')
        datasets[features_name], datasets[embedding_name], datasets[labels_name] = \
            create_embedding(features_name, labels_name)

    np.savez(args.output_filename, **datasets)


if __name__ == '__main__':
    #Parses terminal command
    parser = ArgumentParser()
    parser.add_argument('dataset_filename', type=str)

    parser.add_argument('--pretrained-model', type=str)
    parser.add_argument('--output-filename', type=str)

    args = parser.parse_args()
    main(args)

