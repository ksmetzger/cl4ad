import os
import h5py
import numpy as np
from argparse import ArgumentParser

from sklearn.model_selection import train_test_split
import tensorflow as tf

import preprocessing

def report_file_specs(filename, divisions):
    '''
    Reports file specs: keys, shape pairs. If divisions, also reports number of samples from each label represented
    in dataset
    '''
    data = np.load(filename, mmap_mode='r')
    for key in data.keys(): print(f"Key: '{key}' Shape: '{data[key].shape}'")

    name_mappings = {0:"W-Boson", 1:"QCD", 2:"Z_2", 3:"tt", 4:"leptoquark",
        5:"ato4l", 6:"hChToTauNu", 7:"hToTauTau"}

    print('File Specs:')
    if divisions != []: # prints frequency of each label
        labels = data['labels_train'].copy()
        labels = labels.reshape((labels.shape[0],))
        label_counts = labels.astype(int)
        label_counts = np.bincount(label_counts)
        for label, count in enumerate(label_counts):
            print(f"Label {name_mappings[label]}: {count} occurances")

def main(args):
    '''
    Given Monte Carlo datasets and background IDs, make smaller dummy dataset for debugging
    If divisions, splits data to include fixed percent from each label in training data.
    Otherwise, randomly samples from points. OG split: (W 0.592, QCD 0.338, Z 0.067, tt 0.003)
    '''

    # Load the data and labels files using mmap_mode for efficiency
    data = np.load(args.background_dataset, mmap_mode='r')
    labels = np.load(args.background_ids, mmap_mode='r')

    print(f'Divisions {args.divisions}')

    if not args.divisions:
        # No divisions -> Randomly selects samples to include in smaller batch
        train_ix = np.random.choice(data['x_train'].shape[0], size=args.sample_size, replace=False)
        # Extract sample_size samples from relevant files
        np.random.shuffle(train_ix)
        x_train = data['x_train'][train_ix]
        x_test  = data['x_test'][:args.sample_size]
        x_val   = data['x_val'][:args.sample_size]
        id_train = labels['background_ID_train'][train_ix].reshape((-1, 1))
        id_test = labels['background_ID_test'][:args.sample_size].reshape((-1, 1))
        id_val  = labels['background_ID_val'][:args.sample_size].reshape((-1, 1))

    else:
        # divisions provided -> smaller batch has divisions[i] percent of sampels from ith label
        def division_indicies(dataset, labels, sample_size=-1):
            ix = []

            for label_category in range(len(args.divisions)):

                indices = np.where(labels==label_category)[0]
                label_sample_size = int(args.divisions[label_category] * labels.shape[0]) \
                    if sample_size==-1 \
                    else int(args.divisions[label_category] * sample_size)

                # If samples available < required -> use replacement
                replacement = True if len(indices) < label_sample_size else False

                indices = np.random.choice(indices, size=label_sample_size, replace=replacement)
                ix.extend(indices)

            np.random.shuffle(ix)

            return dataset[ix], labels[ix].reshape((-1,1))

        # Extract sample_size samples from relevant files
        x_train, id_train = division_indicies(data['x_train'], labels['background_ID_train'], args.sample_size)
        x_test, id_test = division_indicies(data['x_test'], labels['background_ID_test'])
        x_val, id_val = division_indicies(data['x_val'], labels['background_ID_val'])

    anomaly_dataset = np.load(args.anomaly_dataset)

    for i_key, key in enumerate(anomaly_dataset.keys()):

        anomaly_dataset_i = anomaly_dataset[key][:args.anomaly_size]
        print(f"making datasets for {key} anomaly with shape {anomaly_dataset_i.shape}")

        # Predicts anomaly_dataset_i using encoder and defines anomalous labels as 4.0
        anomaly_labels = np.empty((anomaly_dataset_i.shape[0],1))
        anomaly_labels.fill(4+i_key)

        anomaly_train, anomaly_test, anomaly_id_train, anomaly_id_test \
            = train_test_split(anomaly_dataset_i, anomaly_labels,
                test_size=0.2, random_state=1)

        anomaly_train, anomaly_val, anomaly_id_train, anomaly_id_val \
            = train_test_split(anomaly_train, anomaly_id_train,
                test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

        def shuffle_along_axis(a, a_ids, axis=0):
            idx = np.random.rand(a.shape[0]).argsort()
            return a[idx], a_ids[idx]

        # Concatenate background and anomaly to feed into plots
        x_train = np.concatenate([anomaly_train, x_train], axis=0)
        id_train = np.concatenate([anomaly_id_train, id_train], axis=0)
        x_train, id_train = shuffle_along_axis(x_train, id_train)

        x_test = np.concatenate([anomaly_test, x_test], axis=0)
        id_test = np.concatenate([anomaly_id_test, id_test], axis=0)
        x_test, id_test = shuffle_along_axis(x_test, id_test)

        x_val = np.concatenate([anomaly_val, x_val], axis=0)
        id_val = np.concatenate([anomaly_id_val, id_val], axis=0)
        x_val, id_val = shuffle_along_axis(x_val, id_val)

    # Normalizes train and testing features by x' = (x - μ) / σ, where μ, σ are predetermined constants
    x_train = preprocessing.zscore_preprocess(x_train, train=True, scaling_file=args.scaling_filename)
    x_test = preprocessing.zscore_preprocess(x_test, scaling_file=args.scaling_filename)
    x_val = preprocessing.zscore_preprocess(x_val, scaling_file=args.scaling_filename)

    # Create and save new .npz with extracted features. Reports success
    new_dataset = {'x_train': x_train, 'x_test': x_test, 'x_val': x_val,
                   'labels_train': id_train, 'labels_test': id_test, 'labels_val': id_val}

    np.savez(args.output_filename, **new_dataset)
    print(f'{args.output_filename} successfully saved')
    report_file_specs(args.output_filename, args.divisions)


if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()
    # required arguments
    parser.add_argument('background_dataset', type=str)
    parser.add_argument('background_ids', type=str)
    parser.add_argument('anomaly_dataset', type=str)

    parser.add_argument('--divisions', nargs='+', type=float, default=[])
    parser.add_argument('--scaling-filename', type=str)
    parser.add_argument('--output-filename', type=str, default='data.npz')
    parser.add_argument('--sample-size', type=int, default=120000)
    parser.add_argument('--anomaly-size', type=int, default=30000)
    args = parser.parse_args()

    main(args)
