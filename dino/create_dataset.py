import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
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
                self.scaled_dataset[f"{k.replace('x','ix')}"], self.scaled_dataset[f"{k.replace('x','ixa')}"] = \
                    self.division_indicies(self.data[k], self.labels[k.replace('x', 'background_ID')], divisions)

        if preprocess:
            self.preprocess(preprocess)

    def division_indicies(self, data, labels, divisions):
            ix = []
            location_augmented = []
            for label_category in range(len(divisions)):

                indices = np.where(labels==label_category)[0]
                label_sample_size = int(divisions[label_category] * labels.shape[0])

                # If samples available < required -> use replacement
                replacement = True if len(indices) < label_sample_size else False

                indices = list(np.random.choice(indices, size=label_sample_size, replace=replacement))
                ix.extend(indices)

                loc_aug = np.concatenate((indices[1:], indices[0:1]))
                location_augmented.extend(loc_aug)

            ix, location_augmented = shuffle(ix, location_augmented, random_state=0)

            # return data[ix], location_augmented, labels[ix].reshape((-1,1))
            return ix, location_augmented

    def preprocess(self, scaling_filename):

        # Normalizes train and testing features by x' = (x - μ) / σ, where μ, σ are predetermined constants
        self.scaled_dataset['x_train'] = zscore_preprocess(self.data['x_train'], train=True, scaling_file=scaling_filename)
        self.scaled_dataset['x_test'] = zscore_preprocess(self.data['x_test'], scaling_file=scaling_filename)
        self.scaled_dataset['x_val'] = zscore_preprocess(self.data['x_val'], scaling_file=scaling_filename)

    def save(self, filename):

        np.savez(filename,
            x_train=self.scaled_dataset['x_train'],
            ix_train=self.scaled_dataset['ix_train'],
            ixa_train=self.scaled_dataset['ixa_train'],
            labels_train=self.labels['background_ID_train'],
            x_test=self.scaled_dataset['x_test'],
            ix_test=self.scaled_dataset['ix_test'],
            ixa_test=self.scaled_dataset['ixa_test'],
            labels_test=self.labels['background_ID_test'],
            x_val=self.scaled_dataset['x_val'],
            ix_val=self.scaled_dataset['ix_val'],
            ixa_val=self.scaled_dataset['ixa_val'],
            labels_val=self.labels['background_ID_val'],
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

class CLBackgroundSignalDataset:
    def __init__(self, data_filename, labels_filename, data_filename_signal, preprocess=True, n_events=-1, divisions=[1,1,1,1]):
        'Initialization'
        rand_number = 0
        np.random.seed(rand_number)
        background_dataset = CLBackgroundDataset(data_filename,labels_filename, preprocess=preprocess,n_events=n_events, divisions=divisions)
        signal_dataset = CLSignalDataset(data_filename_signal, preprocess=preprocess)
        self.scaled_dataset_background = background_dataset.scaled_dataset
        self.scaled_dataset_signal = signal_dataset.scaled_dataset
        #Combine the datasets for training
        #Background part
        self.x_train = self.scaled_dataset_background['x_train'][self.scaled_dataset_background['ix_train']]
        self.x_test = self.scaled_dataset_background['x_test'][self.scaled_dataset_background['ix_test']]
        self.x_val = self.scaled_dataset_background['x_val'][self.scaled_dataset_background['ix_val']]
        self.labels_train = background_dataset.labels['background_ID_train'][self.scaled_dataset_background['ix_train']].reshape((-1,1))
        self.labels_test = background_dataset.labels['background_ID_test'][self.scaled_dataset_background['ix_test']].reshape((-1,1))
        self.labels_val = background_dataset.labels['background_ID_val'][self.scaled_dataset_background['ix_val']].reshape((-1,1))
        #Signal part (Split into train:test:val)
        self.signal_dict = dict()
        for k in self.scaled_dataset_signal:
            if 'label' not in k:
                self.signal_dict[k+"_train"], self.signal_dict[k+"_test"], self.signal_dict[k+"_val"] = np.split(shuffle(signal_dataset.scaled_dataset[k], random_state=rand_number), 
                        [int(.6*len(self.scaled_dataset_signal[k])), int(.8*len(self.scaled_dataset_signal[k]))])
            if 'label' in k:
                self.signal_dict[k+"_train"], self.signal_dict[k+"_test"], self.signal_dict[k+"_val"] = np.split(shuffle(signal_dataset.scaled_dataset[k], random_state=rand_number), 
                        [int(.6*len(self.scaled_dataset_signal[k])), int(.8*len(self.scaled_dataset_signal[k]))])
        for k in self.scaled_dataset_signal:
            if 'label' not in k:
                self.x_train = np.concatenate((self.x_train, self.signal_dict[k+"_train"]), axis=0)                                                              
                self.x_test = np.concatenate((self.x_test, self.signal_dict[k+"_test"]), axis=0)
                self.x_val = np.concatenate((self.x_val, self.signal_dict[k+"_val"]), axis=0)
            if 'label' in k:
                self.labels_train = np.concatenate((self.labels_train, self.signal_dict[k+"_train"]), axis=0)                                                              
                self.labels_test = np.concatenate((self.labels_test, self.signal_dict[k+"_test"]), axis=0)
                self.labels_val = np.concatenate((self.labels_val, self.signal_dict[k+"_val"]), axis=0)
        self.x_train, self.labels_train = shuffle(self.x_train, self.labels_train, random_state=rand_number)
        self.x_test, self.labels_test = shuffle(self.x_test, self.labels_test, random_state=rand_number)
        self.x_val, self.labels_val = shuffle(self.x_val, self.labels_val, random_state=rand_number)

    def report_specs(self):
        '''
        Reports file specs: keys, shape pairs. If divisions, also reports number of samples from each label represented
        in dataset
        '''
        print('File Specs background:')
        datasets = ["Train", "Test", "Validation"]
        for i, k in enumerate([self.labels_train, self.labels_test, self.labels_val]):
            labels = k.copy()
            labels = labels.reshape((labels.shape[0],))
            label_counts = labels.astype(int)
            label_counts = np.bincount(label_counts)
            print("================")
            print(f"{datasets[i]} dataset")
            for label, count in enumerate(label_counts):
                print(f"Label {label, NAME_MAPPINGS[label]}: {count} occurances")
    
    def save(self, filename):
        np.savez(filename,
            x_train=self.x_train,
            labels_train=self.labels_train,
            x_test=self.x_test,
            labels_test=self.labels_test,
            x_val=self.x_val,
            labels_val=self.labels_val,
            )
        print(f'{filename} successfully saved')

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
    #background_dataset.save(args.output_filename)
    print("=====================")
    # prepare signal datasets
    signal_dataset = CLSignalDataset(
        args.anomaly_dataset,
        preprocess=args.scaling_filename
    )
    signal_dataset.report_specs()
    #signal_dataset.save(args.output_anomaly_filename)
    print("=====================")
    # prepare combined background + signal dataset for self-supervised training
    background_signal_dataset = CLBackgroundSignalDataset(
        args.background_dataset, args.background_ids, args.anomaly_dataset,
        preprocess=args.scaling_filename, n_events=args.sample_size,
        divisions=[0.592,0.338,0.067,0.003]
    )
    background_signal_dataset.report_specs()
    #background_signal_dataset.save(args.output_filename)