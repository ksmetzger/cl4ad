import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

NAME_MAPPINGS = {
    0:'W-Boson',
    1:'QCD',
    2:'Z_2',
    3:'tt',
    4:'leptoquark',
    5:'ato4l',
    6:'hChToTauNu',
    7:'hToTauTau'
}

TRAIN_TEST_VAL_MAP = {
    'x_train':1,
    'x_test':0.333,
    'x_val':0.333
}
SIGNAL_MAP = {
    'leptoquark':1,
    'ato4l':0.1643518,
    'hChToTauNu':2.232523,
    'hToTauTau': 2.029938
}

def zscore_preprocess(
        input_array,
        train=False,
        scaling_file=None
        ):
    '''
    Normalizes using zscore scaling along pT only ->  x' = (x - μ) / σ
    Assumes (μ, σ) constants determined by average across training batch
    '''
    # Loads input as tensor and (μ, σ) constants predetermined from training batch.
    if train:
        tensor = input_array.copy()
        mu = np.mean(tensor[:,:,0,:])
        sigma = np.std(tensor[:,:,0,:])
        np.savez(scaling_file, mu=mu, sigma=sigma)

        normalized_tensor = (tensor - mu) / sigma

    else:
        tensor = input_array.copy()
        scaling_const = np.load(scaling_file)
        normalized_tensor = (tensor - scaling_const['mu']) / scaling_const['sigma']

    # Masking so unrecorded data remains 0
    mask = np.not_equal(input_array, 0)
    mask = np.squeeze(mask, -1)

    # Outputs normalized pT while preserving original values for eta and phi
    outputs = np.concatenate([normalized_tensor[:,:,0,:], input_array[:,:,1,:], input_array[:,:,2,:]], axis=2)
    return np.reshape(outputs * mask, (-1, 57))


class TorchCLDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, features, labels, device):
        'Initialization'
        self.device = device
        self.features = torch.from_numpy(features).to(dtype=torch.float32, device=self.device)
        self.labels = torch.from_numpy(labels).to(dtype=torch.float32, device=self.device)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.features)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.features[index]
        y = self.labels[index]

        return X, y


class CLBackgroundDataset:
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_filename, labels_filename, preprocess=True, n_events=-1, divisions=[1,1,1,1], device=None):
        'Initialization'
        self.data = np.load(data_filename, mmap_mode='r')
        self.labels = np.load(labels_filename, mmap_mode='r')
        self.n_events = n_events if n_events!=-1 else len( self.data[ next(iter(self.data)) ] )
        self.scaled_dataset = dict()
        for k in self.data.keys():
            if 'x_' in k:
                idx = np.random.choice(self.data[k].shape[0], size=int(self.n_events*TRAIN_TEST_VAL_MAP[k]), replace=False)
                np.random.shuffle(idx)
                self.scaled_dataset[k], self.scaled_dataset[f"labels{k.replace('x','')}"] = \
                    self.division_indicies(self.data[k][idx], self.labels[k.replace('x', 'background_ID')][idx], divisions)

        if preprocess:
            self.scaled_dataset = self.preprocess(self.scaled_dataset, preprocess)

        self.x_train = self.scaled_dataset['x_train']
        self.x_test = self.scaled_dataset['x_test']
        self.x_val = self.scaled_dataset['x_val']
        self.labels_train = self.scaled_dataset['labels_train']
        self.labels_test = self.scaled_dataset['labels_test']
        self.labels_val = self.scaled_dataset['labels_val']

        self.device = device

    def save(self, filename, model):
        # Create and save new .npz with extracted features. Reports success
        reduced_x_train = np.array_split(self.scaled_dataset['x_train'],5)
        reduced_embedding_train = [model.representation(
                torch.from_numpy(data.reshape(-1,19,3)
                    ).to(dtype=torch.float32, device=self.device)).cpu().detach().numpy() for data in reduced_x_train]
        self.scaled_dataset['embedding_train'] = np.concatenate(reduced_embedding_train, axis=0)
        reduced_x_test = np.array_split(self.scaled_dataset['x_test'],3)
        reduced_embedding_test = [model.representation(
                torch.from_numpy(data.reshape(-1,19,3)
                    ).to(dtype=torch.float32, device=self.device)).cpu().detach().numpy() for data in reduced_x_test]
        self.scaled_dataset['embedding_test'] = np.concatenate(reduced_embedding_test, axis=0)
        reduced_x_val = np.array_split(self.scaled_dataset['x_val'],3)
        reduced_embedding_val = [model.representation(
                torch.from_numpy(data.reshape(-1,19,3)
                    ).to(dtype=torch.float32, device=self.device)).cpu().detach().numpy() for data in reduced_x_val]
        self.scaled_dataset['embedding_val'] = np.concatenate(reduced_embedding_val, axis=0)

        np.savez(filename, **self.scaled_dataset)
        print(f'{filename} successfully saved')

    def division_indicies(self, data, labels, divisions):
            ix = []
            for label_category in range(len(divisions)):

                indices = np.where(labels==label_category)[0]
                label_sample_size = int(divisions[label_category] * labels.shape[0])

                # If samples available < required -> use replacement
                replacement = True if len(indices) < label_sample_size else False

                indices = np.random.choice(indices, size=label_sample_size, replace=replacement)
                ix.extend(indices)

            return data[ix], labels[ix].reshape((-1,1))

    def preprocess(self, data, scaling_filename):
        # Normalizes train and testing features by x' = (x - μ) / σ, where μ, σ are predetermined constants
        data['x_train'] = zscore_preprocess(data['x_train'], train=True, scaling_file=scaling_filename)
        data['x_test'] = zscore_preprocess(data['x_test'], scaling_file=scaling_filename)
        data['x_val'] = zscore_preprocess(data['x_val'], scaling_file=scaling_filename)

        return data

    def report_specs(self):
        '''
        Reports file specs: keys, shape pairs. If divisions, also reports number of samples from each label represented
        in dataset
        '''
        print('File Specs:')
        for k in self.scaled_dataset:
            if 'label' in k:
                labels = self.scaled_dataset[k].copy()
                labels = labels.reshape((labels.shape[0],))
                label_counts = labels.astype(int)
                label_counts = np.bincount(label_counts)
                for label, count in enumerate(label_counts):
                    print(f"Label {label, NAME_MAPPINGS[label]}: {count} occurances")


class CLSignalDataset:
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_filename, n_events=-1, preprocess=None, device=None):
        'Initialization'
        self.data = np.load(data_filename, mmap_mode='r')
        self.labels = []
        self.n_events = n_events if n_events!=-1 else len( self.data[ next(iter(self.data)) ] )
        self.scaled_dataset = dict()
        for k in self.data.keys():
            idx = np.random.choice(self.data[k].shape[0], size=int(self.n_events*SIGNAL_MAP[k]), replace=False)
            np.random.shuffle(idx)
            self.scaled_dataset[k] = self.data[k][idx]
            #self.scaled_dataset[f"labels{k.replace('x','')}"] = self.labels[k][idx]

        if preprocess:
            self.scaled_dataset = self.preprocess(self.scaled_dataset, preprocess)
        
        self.device = device

    def create_labels(self):
        for i_key, key in enumerate(self.data.keys()):

            anomaly_dataset_i = self.data[key][:]
            print(f"making datasets for {key} anomaly with shape {anomaly_dataset_i.shape}")

            # Predicts anomaly_dataset_i using encoder and defines anomalous labels as 4.0
            self.labels[key] = np.empty((anomaly_dataset_i.shape[0],1))
            self.labels[key].fill(4+i_key)

    def save(self, filename, model):
        # Create and save new .npz with extracted features. Reports success
        for k in self.scaled_dataset.copy().keys():
            if 'label' not in k:
                self.scaled_dataset['embedding_'+k] = model.representation(torch.from_numpy(self.scaled_dataset[k].reshape(-1,19,3)).to(dtype=torch.float32, device=self.device)).cpu().detach().numpy()

        np.savez(filename, **self.scaled_dataset)
        print(f'{filename} successfully saved')

    def preprocess(self, data, scaling_filename):
        # Normalizes train and testing features by x' = (x - μ) / σ, where μ, σ are predetermined constants
        for k in data.keys():
            data[k] = zscore_preprocess(data[k], scaling_file=scaling_filename)

        return data


class CLBackgroundSignalDataset:
    def __init__(self, data_filename, labels_filename, data_filename_signal, preprocess=True, n_events=-1, divisions=[1,1,1,1], device=None):
        'Initialization'
        rand_number = 0
        np.random.seed(rand_number)
        background_dataset = CLBackgroundDataset(data_filename,labels_filename, preprocess=preprocess,n_events=n_events,divisions=divisions, device=None)
        signal_dataset = CLSignalDataset(data_filename_signal, n_events=n_events, preprocess=preprocess)
        self.scaled_dataset_background = background_dataset.scaled_dataset
        self.scaled_dataset_signal = signal_dataset.scaled_dataset
        #Combine the datasets for training, for now ignore labels (only self-supervised) but can be added later
        #Background part
        self.x_train = background_dataset.x_train
        self.x_test = background_dataset.x_test
        self.x_val = background_dataset.x_val
        #Signal part (Split into train:test:val)
        self.signal_dict = dict()
        for k in self.scaled_dataset_signal:
            self.signal_dict[k+"_train"], self.signal_dict[k+"_test"], self.signal_dict[k+"_val"] = np.split(shuffle(signal_dataset.scaled_dataset[k], random_state=rand_number), 
                       [int(.6*len(self.scaled_dataset_signal[k])), int(.8*len(self.scaled_dataset_signal[k]))])
        for k in self.scaled_dataset_signal:
            self.x_train = np.concatenate((self.x_train, self.signal_dict[k+"_train"]), axis=0)                                                              
            self.x_test = np.concatenate((self.x_test, self.signal_dict[k+"_test"]), axis=0)
            self.x_val = np.concatenate((self.x_val, self.signal_dict[k+"_val"]), axis=0)
        self.x_train, self.x_test, self.x_val = shuffle(self.x_train, random_state=rand_number), shuffle(self.x_test, random_state=rand_number), shuffle(self.x_val, random_state=rand_number)
        #Labels not integrated yet (filler for now)
        self.labels_train = np.ones(len(self.x_train), dtype=int)
        self.labels_test = np.ones(len(self.x_test), dtype=int)
        self.labels_val = np.ones(len(self.x_val), dtype=int)

    def report_specs(self):
        '''
        Reports file specs: keys, shape pairs. If divisions, also reports number of samples from each label represented
        in dataset
        '''
        print('File Specs:')
        for k in self.scaled_dataset_background:
            if 'label' in k:
                labels = self.scaled_dataset_background[k].copy()
                labels = labels.reshape((labels.shape[0],))
                label_counts = labels.astype(int)
                label_counts = np.bincount(label_counts)
                for label, count in enumerate(label_counts):
                    print(f"Label {label, NAME_MAPPINGS[label]}: {count} occurances")