import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

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


class CLBackgroundDataset:
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_filename, labels_filename, preprocess=None, n_events=-1, divisions=[1,1,1,1]):
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

    def save(self, filename):

        np.savez(filename,
            x_train=self.x_train,
            x_test=self.x_test,
            x_val=self.x_val,
            labels_train=self.labels_train,
            labels_test=self.labels_test,
            labels_val=self.labels_val,
            )
        print(f'{filename} successfully saved')

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
    def __init__(self, data_filename, preprocess=None):
        'Initialization'
        self.data = np.load(data_filename, mmap_mode='r')
        self.labels = self.create_labels()
        self.scaled_dataset = dict()
        for k in self.data.keys():
            idx = np.random.choice(self.data[k].shape[0], size=self.data[k].shape[0], replace=False)
            np.random.shuffle(idx)
            self.scaled_dataset[k], self.scaled_dataset[f"labels_{k}"] = self.data[k][idx], self.labels[k][idx]

        if preprocess:
            self.scaled_dataset = self.preprocess(self.scaled_dataset, preprocess)

    def create_labels(self):
        labels = dict()

        for i_key, key in enumerate(self.data.keys()):

            anomaly_dataset_i = self.data[key][:]
            print(f"making datasets for {key} anomaly with shape {anomaly_dataset_i.shape}")

            # Predicts anomaly_dataset_i using encoder and defines anomalous labels as 4.0
            labels[key] = np.empty((anomaly_dataset_i.shape[0],1))
            labels[key].fill(4+i_key)

        return labels

    def preprocess(self, data, scaling_filename):
        # Normalizes train and testing features by x' = (x - μ) / σ, where μ, σ are predetermined constants
        for k in data.keys():
            if not 'label' in k: data[k] = zscore_preprocess(data[k], scaling_file=scaling_filename)

        return data

    def save(self, filename):
        np.savez(filename, **self.scaled_dataset)
        print(f'{filename} successfully saved')

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


if __name__=='__main__':

    # Parses terminal command
    parser = ArgumentParser()

    # If not using full data, name of smaller dataset to pull from
    parser.add_argument('background_dataset', type=str)
    parser.add_argument('background_ids', type=str)
    parser.add_argument('anomaly_dataset', type=str)

    parser.add_argument('--scaling-filename', type=str)
    parser.add_argument('--output-filename', type=str, default=None)
    parser.add_argument('--output-anomaly-filename', type=str, default=None)
    parser.add_argument('--sample-size', type=int, default=-1)
    parser.add_argument('--mix-in-anomalies', action='store_true')

    args = parser.parse_args()

    # start with making the background dataset
    background_dataset = CLBackgroundDataset(
        args.background_dataset, args.background_ids,
        preprocess=args.scaling_filename,
        divisions=[0.30, 0.30, 0.20, 0.20],
    )
    background_dataset.report_specs()
    background_dataset.save(args.output_filename)

    # prepare signal datasets
    signal_dataset = CLSignalDataset(
        args.anomaly_dataset,
        preprocess=args.scaling_filename
    )
    signal_dataset.report_specs()
    signal_dataset.save(args.output_anomaly_filename)

