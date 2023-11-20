print("Importing from 'test.py'")
import os
import sys
# Limit display of TF messages to errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import data_preprocessing
import nf_classifier
from tensorflow import keras
import models
import losses
import graphing_module
from argparse import ArgumentParser
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau)
import matplotlib.pyplot as plt

def test_main(args):
    '''Infastructure for training and plotting CVAE (background specific and with anomalies)'''
    print("=========================")
    print("PULLING DATA FOR TRAINING")

    features_dataset = np.load('../data/' + args.subset_data_name)

    features_train = features_dataset['x_train']
    features_test = features_dataset['x_test']
    features_val = features_dataset['x_val']
    labels_train = tf.reshape(features_dataset['labels_train'], (-1, 1))
    labels_test = tf.reshape(features_dataset['labels_test'], (-1, 1))
    labels_val = tf.reshape(features_dataset['labels_val'], (-1, 1))

    for name, arr in zip(['x_train','x_test','x_val'], [features_train,features_test,features_val]):
        print(name, np.unique(arr, return_counts=True))
    for name, arr in zip(['labels_train','labels_test','labels_val'], [labels_train,labels_test,labels_val]):
        print(name, np.unique(arr, return_counts=True))

    if args.train:
        # Creates CVAE and trains on training data. Saves encoder
        print("=============================")
        print("MAKING AND TRAINING THE MODEL")
        callbacks = [EarlyStopping(monitor='contrastive_loss', patience=10, verbose=1)]
        CVAE = models.CVAE(losses.SimCLRLoss, temp=args.loss_temp, latent_dim=args.latent_dim)
        CVAE.compile(optimizer='adam')
        history = CVAE.fit(features_train, labels_train, epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks,
                           validation_data=(features_val, labels_val))
        subfolder = os.path.join(os.path.dirname(__file__), '..', 'model_weights')
        os.makedirs(subfolder, exist_ok=True)
        saved_weights_path = os.path.join(subfolder, args.encoder_name)
        CVAE.encoder.save_weights(saved_weights_path)
        print(f"MODEL SAVED AT {saved_weights_path}")

        test_representation, _, _ = CVAE.encoder.predict(features_test)
        np.save('../data/zscore_rep.npy', test_representation)

    if args.plot:
        encoder = models.build_encoder(args.latent_dim, 2, [64, 32])
        encoder.load_weights("../model_weights/" + args.encoder_name)

        folder = f"Epochs_{args.epochs}_BatchSize_{args.batch_size}_LatentDim_{args.latent_dim}"

        print("==============================")
        print("MAKING RELEVANT TRAINING PLOTS")
        test_representation, _, _ = encoder.predict(features_test)
        # plt.plot(test_representation[:,0],test_representation[:,1], labels=labels_test)

        unique_labels = np.unique(labels_test)
        name_mappings = {0.0:"W-Boson", 1.0:"QCD", 2.0:"Z_2", 3.0:"tt", 4.0:"leptoquark",
            5.0:"ato4l", 6.0:"hChToTauNu", 7.0:"hToTauTau"}
        anomaly_colors = {0.0:'#00C142', 1.0:'#44CBB7', 2.0:'#4457CB', 3.0:'#8B15E4',
            4.0:'#C01E1E', 5.0:'#1f77b4', 6.0:'#ff7f0e', 7.0:'#2ca02c'}
        default_colors = {0.0:'#1f77b4', 1.0:'#ff7f0e', 2.0:'#2ca02c', 3.0:'#d62728',
            4.0:'#C01E1E', 5.0:'#1f77b4', 6.0:'#ff7f0e', 7.0:'#2ca02c'}

        # Plots representation per label. If anomaly -> uses special color pallet + opacity
        for label in unique_labels:
            name = name_mappings[label]
            indices = np.where(labels_test == label)[0]
            if label < 4.0: alpha, color = 0.025, anomaly_colors[label]
            else: alpha, color = 0.0125, anomaly_colors[label]
            plt.scatter(test_representation[indices, 0], test_representation[indices, 1], label=name, alpha=0.05, c=color, s=0.7)

        leg = plt.legend(markerscale = 3.0, loc='upper right')
        for lh in leg.legendHandles: lh.set_alpha(1)
        plt.savefig('plot.pdf')

        # graphing_module.plot_2D_pca(test_representation, folder, f'1_2D_PCA.png', labels=labels_test)
        # graphing_module.plot_3D_pca(test_representation, folder, f'1_3D_PCA.png', labels=labels_test)
        # graphing_module.plot_corner_plots(test_representation, folder, f'1_Latent_Corner_Plots.png',
        #                                   labels_test, plot_pca=False)
        # graphing_module.plot_corner_plots(test_representation, folder, f'1_PCA_Corner_Plots.png',
        #     labels_test, plot_pca=True)

    if args.train_nf:

        # Connects to GPU, rasies an error if CPU is used
        print("Connecting to GPU")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Loads the data from npz file into tensorflow dataloader
        print("Loading latent space data")
        data = np.load('../data/zscore_rep.npy')
        train_data = torch.from_numpy(data).to(dtype=torch.float32, device=device)
        train_data_loader = DataLoader(train_data, batch_size=1002, shuffle=True)

        # Creates FlowModel instance and trains model
        print("Creating Flow Model")
        # latent_size, batch_size, num_layers, device
        flow_model = nf_classifier.FlowModel(6, 1002, 10, device)
        print("Training Flow Model")
        # N epochs
        flow_model.train(train_data_loader, 10)
        print("Saving Flow Model")
        flow_model.save_model('itworkedyay.pt')

    if args.test_nf:
        # Plots ROC curves and saves file
        plot_ROC(background_test_classes, anomaly_test_classes, folder, f'3_ROC.png', anomaly)


if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()

    # If not using full data, name of smaller dataset to pull from
    parser.add_argument('--subset-data-name', type=str, default='zscore.npz')

    parser.add_argument('--latent-dim', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--learning-rate', type=float, default=0.003)
    parser.add_argument('--loss-temp', type=float, default=0.07)
    parser.add_argument('--encoder-name', type=str, default='zscore.h5')

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--train-nf', action='store_true')
    parser.add_argument('--test-nf', action='store_true')


    args = parser.parse_args()
    test_main(args)
