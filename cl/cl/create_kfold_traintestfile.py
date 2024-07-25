import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import random
import h5py

class ADC_Dataset():
    """ 
    Load, preprocess and save k-fold of the ADC-dataset.
    0:'W-Boson',
    1:'QCD',
    2:'Z_2',
    3:'tt', 
    """
    def __init__(self, args):
        self.data = self.load_dataset(args)
        self.preprocessed_data = self.preprocess(data=self.data)
        self.k_folded_data = self.kfold_dataset(n_folds=5,data=self.preprocessed_data)

    def load_dataset(self, args):
        dataset = np.load(args.background_dataset)
        labels = np.load(args.background_ids)
        print(f"Dataset files loaded: {dataset.files}")
        print(f"Label files loaded: {labels.files}")
        data_dict = {
            "x_train": dataset["x_train"],
            "x_val": dataset["x_val"],
            "x_test": dataset["x_test"],
            "labels_train": labels["background_ID_train"],
            "labels_val": labels["background_ID_val"],
            "labels_test": labels["background_ID_test"],
        }
        return data_dict
    def preprocess(self, data):
        data_dict_preprocessed = {
            "x_train": zscore_preprocess(data["x_train"], train=True, scaling_file="scaling_file.npz"),
            "x_val": zscore_preprocess(data["x_val"], train=False, scaling_file="scaling_file.npz"),
            "x_test": zscore_preprocess(data["x_test"], train=False, scaling_file="scaling_file.npz"),
            "labels_train": data["labels_train"],
            "labels_val":  data["labels_val"],
            "labels_test":  data["labels_test"],
            }
        return data_dict_preprocessed
    def print_stats(self):
        events_train = len(self.preprocessed_data["labels_train"])
        events_val = len(self.preprocessed_data["labels_val"])
        events_test = len(self.preprocessed_data["labels_test"])
        #Combine train and val and just kfold!
        events_train = events_train + events_val
        #Print class distribution and number of events in train/test datasets
        class_dist_train = {
            "class_0": np.sum(self.preprocessed_data['labels_train'] == 0) +np.sum(self.preprocessed_data['labels_val'] == 0),
            "class_1": np.sum(self.preprocessed_data['labels_train'] == 1) +np.sum(self.preprocessed_data['labels_val'] == 1),
            "class_2": np.sum(self.preprocessed_data['labels_train'] == 2) +np.sum(self.preprocessed_data['labels_val'] == 2),
            "class_3": np.sum(self.preprocessed_data['labels_train'] == 3) +np.sum(self.preprocessed_data['labels_val'] == 3),
        }
        class_dist_test = {
            "class_0": np.sum(self.preprocessed_data['labels_test'] == 0),
            "class_1": np.sum(self.preprocessed_data['labels_test'] == 1),
            "class_2": np.sum(self.preprocessed_data['labels_test'] == 2),
            "class_3": np.sum(self.preprocessed_data['labels_test'] == 3),
        }
        types = ["Train", "Test"]
        lengths = [events_train, events_test]
        classes = [class_dist_train, class_dist_test]
        for type, length, classe in zip(types, lengths, classes):
            print(f"===============")
            print(f"Printing statistics for the {type} dataset:")
            print(f"===============")
            print(f"Total events in the {type} dataset: {length}")
            print(f"Percentage of W-Boson events: {classe['class_0']/length*100}%")
            print(f"Percentage of QCD-multijet events: {classe['class_1']/length*100}%")
            print(f"Percentage of Z-Boson events: {classe['class_2']/length*100}%")
            print(f"Percentage of ttbar events: {classe['class_3']/length*100}%")
    def kfold_dataset(self, n_folds, data):
        #First shuffle the dataset and then split it into 5 equal sized folds
        k_folded_dict = {
            "x_test": data["x_test"],
            "labels_test": data["labels_test"]
        }
        #Combine the test and val set as that is handled by the k-folding!
        data["x_combined"] = np.concatenate((data["x_train"], data["x_val"]), axis=0)
        data["labels_combined"] = np.concatenate((data["labels_train"], data["labels_val"]), axis=0)
        data["x_combined"], data["labels_combined"] = shuffle(data["x_combined"],data["labels_combined"], random_state=args.seed)
        idx = np.random.choice(n_folds, size=len(data["labels_combined"]), replace=True)
        for i in range(n_folds):
            mask = (idx == i)
            k_folded_dict[f"x_train_fold_{i}"] = data["x_combined"][mask]
            k_folded_dict[f"labels_train_fold_{i}"] = data["labels_combined"][mask]
        return k_folded_dict 
    def save_as_h5(self):
        # Open HDF5 file and write in the data_dict structure and info
        with h5py.File(f'{args.output_filename}.hdf5', 'w') as f:
            for dataset_name in self.k_folded_data:
                dset = f.create_dataset(dataset_name, data=self.k_folded_data[dataset_name])
        #Check if it is saved
        with h5py.File(f'{args.output_filename}.hdf5', 'r') as f:
            print(list(f.keys()))

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

def main(args):
    #Seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    #Create the dataset
    dataset = ADC_Dataset(args)
    #Print stats
    dataset.print_stats()
    #Save the h5 file
    dataset.save_as_h5()

if __name__=='__main__':
    # Parses terminal command
    parser = ArgumentParser()

    # If not using full data, name of smaller dataset to pull from
    parser.add_argument('background_dataset', type=str)
    parser.add_argument('background_ids', type=str)

    parser.add_argument('--output-filename', type=str, default='ADC_Delphes_original_divisions')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)