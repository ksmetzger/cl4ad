import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchsummary import summary

import losses
from models import CVAE
from dataset import CLDataset

def main(args):
    '''
    Infastructure for training CVAE (background specific and with anomalies)
    '''

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    features_dataset = np.load(args.dataset_filename)

    features_train = torch.from_numpy(features_dataset['x_train']).to(dtype=torch.float32, device=device)
    labels_train = torch.from_numpy(features_dataset['labels_train']).to(dtype=torch.float32, device=device)
    train_set = CLDataset(features_train, labels_train)

    features_val = torch.from_numpy(features_dataset['x_val']).to(dtype=torch.float32, device=device)
    labels_val = torch.from_numpy(features_dataset['labels_val']).to(dtype=torch.float32, device=device)
    val_set = CLDataset(features_val, labels_val)

    dataloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False)

    valloader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False)

    model = CVAE().to(device)
    summary(model, input_size=(57,))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = losses.SimCLRLoss()

    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        model.train(True)
        step_loss = []
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

            step_loss.append(loss.item())
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        step_vloss = []
        model.eval()
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(valloader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                step_vloss.append(vloss.item())

        train_losses.append(np.array(step_loss).mean())
        val_losses.append(np.array(step_vloss).mean())
        print(f'====> Epoch: {epoch} Average train loss: {np.array(step_loss).mean():.4f} val loss {np.array(step_vloss).mean():.4f}')

        # reduce LR on plateu
        scheduler.step(np.array(step_vloss).mean())

    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('output/loss.pdf')

    torch.save(model.state_dict(), args.model_name)


if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()

    # If not using full data, name of smaller dataset to pull from
    parser.add_argument('dataset_filename', type=str)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--loss-temp', type=float, default=0.07)
    parser.add_argument('--model-name', type=str, default='output/vae.pth')

    args = parser.parse_args()
    main(args)
