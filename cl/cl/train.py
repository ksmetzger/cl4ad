import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
torch.autograd.set_detect_anomaly(True)


import losses
from models import CVAE


class TorchCLDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, features, ix, ixa, labels, criterion, device):
          'Initialization'
          self.device = device
          self.features = torch.from_numpy(features[ix]).to(dtype=torch.float32, device=self.device)
          self.augmentations = torch.from_numpy(features[ixa].copy()).to(dtype=torch.float32, device=self.device)
          self.labels = torch.from_numpy(labels[ix]).to(dtype=torch.float32, device=self.device)

    def __len__(self):
          'Denotes the total number of samples'
          return len(self.features)

    def __getitem__(self, index):
          'Generates one sample of data'
          # Load data and get label
          X = self.features[index]
          X_aug = self.augmentations[index]
          y = self.labels[index]

          return X, X_aug, y


def main(args):
    '''
    Infastructure for training CVAE (background specific and with anomalies)
    '''

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    dataset = np.load(args.background_dataset)

    # criterion = losses.SimCLRLoss()
    criterion = losses.VICRegLoss()

    train_data_loader = DataLoader(
        TorchCLDataset(
            dataset['x_train'],
            dataset['ix_train'],
            dataset['ixa_train'],
            dataset['labels_train'],
            criterion, device),
        batch_size=args.batch_size,
        shuffle=False)

    test_data_loader = DataLoader(
        TorchCLDataset(
            dataset['x_test'],
            dataset['ix_test'],
            dataset['ixa_test'],
            dataset['labels_test'],
            criterion, device),
        batch_size=args.batch_size,
        shuffle=False)

    val_data_loader = DataLoader(
        TorchCLDataset(
            dataset['x_val'],
            dataset['ix_val'],
            dataset['ixa_val'],
            dataset['labels_val'],
            criterion, device),
        batch_size=args.batch_size,
        shuffle=False)

    model = CVAE().to(device)
    summary(model, input_size=(57,))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler_1 = torch.optim.lr_scheduler.ConstantLR(optimizer, total_iters=5)
    scheduler_2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler_3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler_1, scheduler_2, scheduler_3], milestones=[5, 20])


    def train_one_epoch(epoch_index):
        running_sim_loss = 0.
        last_sim_loss = 0.

        for idx, (val, val_aug, _) in enumerate(train_data_loader, 1):
            # only applicable to the final batch
            if val.shape[0] != args.batch_size:
                continue

            embedded_values_orig = model(val)
            embedded_values_aug = model(val_aug)

            similar_embedding_loss = criterion(embedded_values_aug.reshape((-1,1,6)), \
                embedded_values_orig.reshape((-1,1,6)))

            optimizer.zero_grad()
            similar_embedding_loss.backward()
            optimizer.step()
            # Gather data and report
            running_sim_loss += similar_embedding_loss.item()
            if idx % 500 == 0:
                last_sim_loss = running_sim_loss / 500
                running_sim_loss = 0.

        return last_sim_loss


    def val_one_epoch(epoch_index):
        running_sim_loss = 0.
        last_sim_loss = 0.

        for idx,(val, val_aug, _) in enumerate(val_data_loader, 1):

            if val.shape[0] != args.batch_size:
                continue

            embedded_values_orig = model(val)
            embedded_values_aug = model(val_aug)

            similar_embedding_loss = criterion(embedded_values_aug.reshape((-1,1,6)), \
                embedded_values_orig.reshape((-1,1,6)))

            running_sim_loss += similar_embedding_loss.item()
            if idx % 500 == 0:
                last_sim_loss = running_sim_loss / 500
                running_sim_loss = 0.

        return last_sim_loss

    if args.train:
        train_losses = []
        val_losses = []
        for epoch in range(1, args.epochs+1):
            print(f'EPOCH {epoch}')
            # Gradient tracking
            model.train(True)
            avg_train_loss = train_one_epoch(epoch)
            train_losses.append(avg_train_loss)

            # no gradient tracking, for validation
            model.train(False)
            avg_val_loss = val_one_epoch(epoch)
            val_losses.append(avg_val_loss)

            print(f"Train/Val Sim Loss after epoch: {avg_train_loss:.4f}/{avg_val_loss:.4f}")

            scheduler.step()

        torch.save(model.state_dict(), args.model_name)
    else:
        model.load_state_dict(torch.load(args.model_name))
        model.eval()

    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('output/loss.pdf')


if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()

    # If not using full data, name of smaller dataset to pull from
    parser.add_argument('background_dataset', type=str)
    parser.add_argument('anomaly_dataset', type=str)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--loss-temp', type=float, default=0.07)
    parser.add_argument('--model-name', type=str, default='output/vae.pth')
    parser.add_argument('--scaling-filename', type=str)
    parser.add_argument('--output-filename', type=str, default='output/embedding.npz')
    parser.add_argument('--sample-size', type=int, default=-1)
    parser.add_argument('--mix-in-anomalies', action='store_true')
    parser.add_argument('--train', action='store_true')

    args = parser.parse_args()
    main(args)
