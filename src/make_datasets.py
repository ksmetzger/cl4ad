print("Importing from 'make_datasets.py'")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from sklearn.model_selection import train_test_split
import tensorflow as tf
import h5py
import numpy as np
import data_preprocessing
from argparse import ArgumentParser

def make_montecarlo_dataset(sample_size, anomaly_size, new_filename, divisions, normalization_type):
    '''
    Given Monte Carlo datasets and background IDs, make smaller dummy dataset for debugging
    If divisions, splits data to include fixed percent from each label in training data.
    Otherwise, randomly samples from points. OG split: (W 0.592, QCD 0.338, Z 0.067, tt 0.003)
    '''
    # Load the data and labels files using mmap_mode for efficiency
    data = np.load('/eos/home-e/egovorko/kd_data/datasets_-1.npz', mmap_mode='r')
    labels = np.load('/eos/home-e/egovorko/kd_data/background_IDs_-1.npz', mmap_mode='r')

    if divisions == []:
        # No divisions -> Randomly selects samples to include in smaller batch
        train_ix = np.random.choice(data['x_train'].shape[0], size=sample_size, replace=False)
    else:
        # divisions provided -> smaller batch has divisions[i] percent of sampels from ith label
        train_ix = []
        train_labels = labels['background_ID_train']
        for label_category in range(4):
            indices = np.where(train_labels == label_category)[0]
            if sample_size==-1:
                label_sample_size = int(divisions[label_category] * labels['background_ID_train'].shape[0])
            else:
                label_sample_size = int(divisions[label_category] * sample_size)
            if len(indices) < label_sample_size: replacement = True
            else: replacement = False # If samples avaliable < required -> use replacement
            indices = np.random.choice(indices, size=label_sample_size, replace=replacement)
            train_ix.extend(indices)

    # Extract sample_size samples from relevant files
    np.random.shuffle(train_ix)
    x_train = data['x_train'][train_ix]
    x_test  = data['x_test'][:sample_size]
    x_val   = data['x_val'][:sample_size]
    id_train = tf.reshape(labels['background_ID_train'][train_ix], (-1, 1))
    id_test = tf.reshape(labels['background_ID_test'][:sample_size], (-1, 1))
    id_val  = tf.reshape(labels['background_ID_val'][:sample_size], (-1, 1))

    anomaly_dataset = np.load('/eos/home-e/egovorko/kd_data/bsm_datasets_-1.npz')

    for i_key, key in enumerate(anomaly_dataset.keys()):

        anomaly_dataset_i = anomaly_dataset[key][:anomaly_size]
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

        # Concatenate background and anomaly to feed into plots
        x_train = np.concatenate([anomaly_train, x_train], axis=0)
        id_train = tf.concat([anomaly_id_train, id_train], axis=0)

        x_test = np.concatenate([anomaly_test, x_test], axis=0)
        id_test = tf.concat([anomaly_id_test, id_test], axis=0)

        x_val = np.concatenate([anomaly_val, x_val], axis=0)
        id_val = tf.concat([anomaly_id_val, id_val], axis=0)

    if normalization_type == 'max_pt':
        # Normalizes train and testing features by dividing by max pT. Saves weights in 'configs.py' file
        data_preprocessing.save_normalization_weights(x_train, new_filename)
        x_train = data_preprocessing.maxPT_preprocess(x_train, new_filename)
        x_test = data_preprocessing.maxPT_preprocess(x_test, new_filename)
        x_val = data_preprocessing.maxPT_preprocess(x_val, new_filename)

    elif normalization_type == 'zscore':
        # Normalizes train and testing features by x' = (x - μ) / σ, where μ, σ are predetermined constants
        x_train = data_preprocessing.zscore_preprocess(x_train, train=True, scaling_file='/eos/home-e/egovorko/kd_data/zscore_scaling.npz')
        x_test = data_preprocessing.zscore_preprocess(x_test, scaling_file='/eos/home-e/egovorko/kd_data/zscore_scaling.npz')
        x_val = data_preprocessing.zscore_preprocess(x_val, scaling_file='/eos/home-e/egovorko/kd_data/zscore_scaling.npz')

    # Create and save new .npz with extracted features. Reports success
    new_dataset = {'x_train': x_train, 'x_test': x_test, 'x_val': x_val,
                   'labels_train': id_train, 'labels_test': id_test, 'labels_val': id_val}

    file_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(file_path, exist_ok=True)
    file_path = os.path.join(file_path, new_filename)
    np.savez(file_path, **new_dataset)
    print(f"{file_path} successfully saved")

def make_raw_cms_dataset(delphes_filter, new_filename, training_filename, normalization_type):
    '''
    Given raw CMS, converts to npz file with appropriate transofmrations:
    '''
    raw_cms_file = h5py.File('/eos/home-e/egovorko/kd_data/raw_cms.h5', 'r')
    dataset_np = np.array(raw_cms_file['full_data_cyl'])

    if delphes_filter:
        delphes_filter = raw_cms_file['L1_SingleMu22']
        filter_np  = np.array(delphes_filter)
        dataset_np = dataset_np[filter_np]

    # Reordering and reshaping
    dataset_np = data_preprocessing.transform_raw_cms(dataset_np)

    # Convert from computer int embedding to meaningful float rep
    dataset_np = data_preprocessing.convert_to_float(dataset_np)

    # Phi Shift so range from [0, 2π] to [-π, π]
    dataset_np = tf.reshape(dataset_np, (-1, 19, 3, 1))
    dataset_np = data_preprocessing.phi_shift(dataset_np)

    # Either max_pT or zscore normalization. Reshapes/cast so shapes compatable
    dataset_np = tf.reshape(dataset_np, (-1, 19, 3, 1))
    dataset_np = tf.cast(dataset_np, dtype=tf.float32)

    if normalization_type == 'max_pt':
        # Normalizes features by dividing by max pT. Uses training_filename to pull presaved max_pT weight
        dataset_np = data_preprocessing.maxPT_preprocess(dataset_np, training_filename)
    elif normalization_type == 'zscore':
        # Normalizes features by x' = (x - μ) / σ, where μ, σ are predetermined constants
        dataset_np = data_preprocessing.zscore_preprocess(dataset_np)

    # Saves files and reports sucess
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(file_path, exist_ok=True)
    file_path = os.path.join(file_path, new_filename)
    np.savez(file_path, dataset=dataset_np)
    raw_cms_file.close()
    print(f"{subfolder_path} successfully saved")


def report_file_specs(filename, divisions):
    '''
    Reports file specs: keys, shape pairs. If divisions, also reports number of samples from each label represented
    in dataset
    '''
    data = np.load('../data/' + filename, mmap_mode='r')
    for key in data.keys(): print(f"Key: '{key}' Shape: '{data[key].shape}'")

    name_mappings = {0:"W-Boson", 1:"QCD", 2:"Z_2", 3:"tt", 4:"leptoquark",
        5:"ato4l", 6:"hChToTauNu", 7:"hToTauTau"}

    if divisions != []: # prints frequency of each label
        labels = data['labels_train'].copy()
        labels = labels.reshape((labels.shape[0],))
        label_counts = labels.astype(int)
        label_counts = np.bincount(label_counts)
        for label, count in enumerate(label_counts):
            print(f"Label {name_mappings[label]}: {count} occurances")


if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()
    parser.add_argument('--new_filename', type=str, default='zscore.npz')
    parser.add_argument('--sample_size', type=int, default=120000)
    parser.add_argument('--anomaly_size', type=int, default=30000)
    parser.add_argument('--use_delphes_data', type=bool, default=True)
    parser.add_argument('--normalization_type', type=str, default='zscore')
    
    # Raw CMS Dataset Specific Args. Ignore if making Delphes subset 
    parser.add_argument('--delphes_filter', type=bool, default=False)
    parser.add_argument('--training_filename', type=str, default='max_pt_dataset.npz')
    args = parser.parse_args()
    
    divisions = [0.30, 0.30, 0.20, 0.20]
    
    print("Creating file now:")
    if args.use_delphes_data == True: 
        print("Assuming making a Delphes Data Subset:")
        make_montecarlo_dataset(args.sample_size, args.anomaly_size, args.new_filename, divisions, args.normalization_type)
    else: 
        print("Assuming making a Raw CMS Dataset:")
        make_raw_cms_dataset(args.delphes_filter, args.new_filename, args.training_filename, args.normalization_type)
    
    print("File Specs:")
    report_file_specs(args.new_filename, divisions)
    
    
