import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import losses
from dataset import TorchCLDataset, CLBackgroundDataset, CLSignalDataset, CLBackgroundSignalDataset
from models import CVAE, SimpleDense


def main(args):
    '''
    Infastructure for training CVAE (background specific and with anomalies)
    '''

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    #Dataset with signals and original divisions=[0.592,0.338,0.067,0.003]
    dataset = CLBackgroundSignalDataset(args.background_dataset, args.background_ids, args.anomaly_dataset,
        preprocess=args.scaling_filename, n_events=args.sample_size,
        divisions=[0.592,0.338,0.067,0.003],
        device=device
    )
    dataset.report_specs()

    train_data_loader = DataLoader(
        TorchCLDataset(dataset.x_train, dataset.labels_train, device),
        batch_size=args.batch_size,
        shuffle=False)

    test_data_loader = DataLoader(
        TorchCLDataset(dataset.x_test, dataset.labels_test, device),
        batch_size=args.batch_size,
        shuffle=False)

    val_data_loader = DataLoader(
        TorchCLDataset(dataset.x_val, dataset.labels_val, device),
        batch_size=args.batch_size,
        shuffle=False)

    model = SimpleDense().to(device)
    summary(model, input_size=(57,))

    # criterion = losses.SimCLRLoss()
    criterion = losses.VICRegLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler_1 = torch.optim.lr_scheduler.ConstantLR(optimizer, total_iters=5)
    scheduler_2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler_3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler_1, scheduler_2, scheduler_3], milestones=[5, 20])

    def train_one_epoch(epoch_index, tb_writer):
        running_sim_loss = 0.
        last_sim_loss = 0.

        for idx, val in enumerate(train_data_loader, 1):
            val = val[0]
            # only applicable to the final batch
            if val.shape[0] != args.batch_size:
                continue

            # embed entire batch with first value of the batch repeated
            first_val_repeated = val[0].repeat(args.batch_size, 1)

            embedded_values_orig = model(val)
            embedded_values_aug = model(first_val_repeated)

            similar_embedding_loss = criterion(embedded_values_aug.reshape((-1,1,6)), embedded_values_orig.reshape((-1,1,6)))

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
        return last_sim_loss


    def val_one_epoch(epoch_index, tb_writer):
        running_sim_loss = 0.
        last_sim_loss = 0.

        for idx, val in enumerate(val_data_loader, 1):
            val = val[0]
            if val.shape[0] != args.batch_size:
                continue

            first_val_repeated = val[0].repeat(args.batch_size, 1)

            embedded_values_aug = model(first_val_repeated)
            embedded_values_orig = model(val)

            similar_embedding_loss = criterion(embedded_values_aug.reshape((-1,1,6)), embedded_values_orig.reshape((-1,1,6)))

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
        for epoch in range(1, args.epochs+1):
            print(f'EPOCH {epoch}')
            # Gradient tracking
            model.train(True)
            avg_train_loss = train_one_epoch(epoch, writer)
            train_losses.append(avg_train_loss)

            # no gradient tracking, for validation
            model.train(False)
            avg_val_loss = val_one_epoch(epoch, writer)
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

    #Save the embedding output seperately for the background and signal part
    #dataset.save(args.output_filename, model)
    CLBackgroundDataset(args.background_dataset, args.background_ids, n_events=args.sample_size,
        preprocess=args.scaling_filename,
        divisions=[0.592,0.338,0.067,0.003],
        device=device).save(args.output_filename, model)
    CLSignalDataset(args.anomaly_dataset,n_events=args.sample_size, preprocess=args.scaling_filename, device=device).save(f'output/anomalies_embedding.npz', model)


if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()

    # If not using full data, name of smaller dataset to pull from
    parser.add_argument('background_dataset', type=str)
    parser.add_argument('background_ids', type=str)
    parser.add_argument('anomaly_dataset', type=str)

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--loss-temp', type=float, default=0.07)
    parser.add_argument('--model-name', type=str, default='output/vae.pth')
    parser.add_argument('--scaling-filename', type=str)
    parser.add_argument('--output-filename', type=str, default='output/embedding.npz')
    parser.add_argument('--sample-size', type=int, default=-1)
    parser.add_argument('--mix-in-anomalies', action='store_true')
    parser.add_argument('--train', action='store_true')

    args = parser.parse_args()
    main(args)
