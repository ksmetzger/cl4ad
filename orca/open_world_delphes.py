from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torchvision
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import transforms
import itertools
from torch.utils.data.sampler import Sampler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import warnings
import torch

class BACKGROUND(Dataset):

    def __init__(self, root, labeled=True, labeled_num=3, labeled_ratio=0.5, rand_number=0, transform=None, target_transform=None,
                 download=False, unlabeled_idxs=None):
        #super(OPENWORLDCIFAR100, self).__init__(root, True, transform, target_transform, download)

        #downloaded_list = self.train_list
        drive_path = ''
        background_IDs = np.load(drive_path+'background_IDs_-1.npz')
        background = np.load(drive_path+'datasets_-1.npz')
        
        self.transform = transform
        self.target_transform = transforms.ToTensor()
        self.data = background['x_train'][0:500000,...]
        self.targets = background_IDs['background_ID_train'][0:500000]
        self.targets = self.targets.astype(int)
        
   
        self.data = np.vstack(self.data).reshape(-1, 1, 19, 3)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
       

        labeled_classes = range(labeled_num)
        np.random.seed(rand_number)
        
        if labeled:
            self.labeled_idxs, self.unlabeled_idxs = self.get_labeled_index(labeled_classes, labeled_ratio)
            self.shrink_data(self.labeled_idxs)
        else:
            self.shrink_data(unlabeled_idxs)
            
        print(np.shape(self.data))
        print(np.shape(self.targets))
        print(type(self.targets[0]))
        

    def get_labeled_index(self, labeled_classes, labeled_ratio):
        labeled_idxs = []
        unlabeled_idxs = []
        for idx, label in enumerate(self.targets):
            if label in labeled_classes and np.random.rand() < labeled_ratio:
                labeled_idxs.append(idx)
            else:
                unlabeled_idxs.append(idx)
        return labeled_idxs, unlabeled_idxs

    def shrink_data(self, idxs):
        targets = np.array(self.targets)
        self.targets = targets[idxs].tolist()
        self.data = self.data[idxs, ...]


    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        image = self.transform(image)
        #label = self.target_transform(label)
        return image, label
        

class BACKGROUND_SIGNAL(Dataset):

    def __init__(self, root, datatype, rand_number=0, transform=None, target_transform=None):
        
        #Import the background + signals + labels
        drive_path = ''
        background_IDs = np.load(drive_path+'background_IDs_-1.npz')
        background = np.load(drive_path+'datasets_-1.npz')
        signals = np.load(drive_path+'bsm_datasets_-1.npz')
        
        background_data = background['x_train']
        background_targets = background_IDs['background_ID_train']
        leptoquark = signals['leptoquark']
        ato4l = signals['ato4l']
        hChToTauNu = signals['hChToTauNu']
        hToTauTau = signals['hToTauTau']
        
        #Transformations
        self.transform = transform
        self.target_transform = transforms.ToTensor()
        
        ###Random sample 1/4 (1/8) of the background for labeled data, 1/4 (1/8) of background + signals (1/8) for unlabeled data
        #Random sample the background
        np.random.seed(rand_number)
        size_fraction = 1/8
        labeled_background, unlabeled_background, labeled_targets, unlabeled_targets = train_test_split(background_data, background_targets, test_size=size_fraction, train_size =size_fraction ,random_state=rand_number)
        
        #Also random sample the signals
        unlabeled_leptoquark, _ = train_test_split(leptoquark, train_size=size_fraction, random_state=rand_number)
        unlabeled_ato4l, _ = train_test_split(ato4l, train_size=size_fraction, random_state=rand_number)
        unlabeled_hChToTauNu, _ = train_test_split(hChToTauNu, train_size=size_fraction, random_state=rand_number)
        unlabeled_hToTauTau, _ = train_test_split(hToTauTau, train_size=size_fraction, random_state=rand_number)
        
        #Shuffle in signals (and their labels for testing) with the unlabeled background
        unlabeled_data = np.concatenate((unlabeled_background, unlabeled_leptoquark, unlabeled_ato4l, unlabeled_hChToTauNu, unlabeled_hToTauTau), axis = 0)
        unlabeled_targets = np.concatenate((unlabeled_targets, np.ones(len(unlabeled_leptoquark),dtype=int)*4, np.ones(len(unlabeled_ato4l),dtype=int)*5,np.ones(len(unlabeled_hChToTauNu),dtype=int)*6,np.ones(len(unlabeled_hToTauTau),dtype=int)*7),axis=0)
        unlabeled_data_shuffled, unlabeled_targets_shuffled = shuffle(unlabeled_data, unlabeled_targets, random_state=rand_number)
        
        
        if datatype == 'train_labeled':
            self.data = labeled_background
            self.targets = labeled_targets
     
        elif datatype == 'train_unlabeled':
            self.data = unlabeled_data_shuffled
            self.targets = unlabeled_targets_shuffled
            
        elif datatype == 'test':
            self.data = unlabeled_data_shuffled
            self.targets = unlabeled_targets_shuffled
        else:
            warnings.warn('Type of dataset not available')
            return
        
        
        #Reshape the data
        self.targets = self.targets.astype(int)
        self.targets = self.targets.tolist()
        self.data = np.vstack(self.data).reshape(-1, 1, 19, 3)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
       
        #Print the shapes of data + targets
        print(np.shape(self.data))
        print(np.shape(self.targets))
        print(type(self.targets[0]))
    
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        image = self.transform(image)
        #label = self.target_transform(label)
        return image, label
        

class BACKGROUND_SIGNAL_CVAE(Dataset):

    def __init__(self, root, datatype, rand_number=0, transform=None, target_transform=None):
        
        #Import the background + signals + labels
        drive_path = ''
        background_IDs = np.load(drive_path+'background_IDs_-1.npz')
        background = np.load(drive_path+'datasets_-1.npz')
        signals = np.load(drive_path+'bsm_datasets_-1.npz')
        
        
        background_data = background['x_train']
        background_targets = background_IDs['background_ID_train']
        leptoquark = signals['leptoquark']
        ato4l = signals['ato4l']
        hChToTauNu = signals['hChToTauNu']
        hToTauTau = signals['hToTauTau']
  
             
        #Transformations
        self.transform = transform
        self.target_transform = transforms.ToTensor()
        
        ###Random sample 1/4 (1/8) of the background for labeled data, 1/4 (1/8) of background + signals (1/8) for unlabeled data
        #Random sample the background
        np.random.seed(rand_number)
        size_fraction_background = 1/4
        size_fraction_signal = 1/4
        labeled_background, unlabeled_background, labeled_targets, unlabeled_targets = train_test_split(background_data, background_targets, test_size=size_fraction_background, train_size =size_fraction_background ,random_state=rand_number)
        
        #Also random sample the signals
        unlabeled_leptoquark, _ = train_test_split(leptoquark, train_size=size_fraction_signal, random_state=rand_number)
        unlabeled_ato4l, _ = train_test_split(ato4l, train_size=size_fraction_signal, random_state=rand_number)
        unlabeled_hChToTauNu, _ = train_test_split(hChToTauNu, train_size=size_fraction_signal, random_state=rand_number)
        unlabeled_hToTauTau, _ = train_test_split(hToTauTau, train_size=size_fraction_signal, random_state=rand_number)
        
        #Shuffle in signals (and their labels for testing) with the unlabeled background
        unlabeled_data = np.concatenate((unlabeled_background, unlabeled_leptoquark, unlabeled_ato4l, unlabeled_hChToTauNu, unlabeled_hToTauTau), axis = 0)
        unlabeled_targets = np.concatenate((unlabeled_targets, np.ones(len(unlabeled_leptoquark),dtype=int)*4, np.ones(len(unlabeled_ato4l),dtype=int)*5,np.ones(len(unlabeled_hChToTauNu),dtype=int)*6,np.ones(len(unlabeled_hToTauTau),dtype=int)*7),axis=0)
        unlabeled_data_shuffled, unlabeled_targets_shuffled = shuffle(unlabeled_data, unlabeled_targets, random_state=rand_number)
   
        
        if datatype == 'train_labeled':
            self.data = labeled_background
            self.targets = labeled_targets
     
        elif datatype == 'train_unlabeled':
            self.data = unlabeled_data_shuffled
            self.targets = unlabeled_targets_shuffled
            
        elif datatype == 'test':
            self.data = unlabeled_data_shuffled
            self.targets = unlabeled_targets_shuffled
        else:
            warnings.warn('Type of dataset not available')
            return
        
        
        #Reshape the data
        self.targets = self.targets.astype(int)
        self.targets = self.targets.tolist()
        self.data = np.vstack(self.data).reshape(-1, 57)
        
       
        #Print the shapes of data + targets
        print(np.shape(self.data))
        print(np.shape(self.targets))
        print(type(self.targets[0]))
    
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        image = torch.from_numpy(image) #Turn numpy array into tensor(57)
        image = (image-2.197)/10.79  #Normalize the data to mean: 0, std: 1
        label = self.targets[idx]
        image = self.transform(image) #Still need the TransformTwice!
        #label = self.target_transform(label)
        return image, label
        
class BACKGROUND_SIGNAL_CVAE_LATENT(Dataset):

    def __init__(self, root, datatype, rand_number=0, transform=None, target_transform=None):
        
        #Import the background + signals + labels
        drive_path = ''
        
        dataset = np.load(drive_path+'unbiased_latent.npz')
        
        
        background_data = dataset['x_train']
        background_targets = dataset['labels_train']
        background_targets = background_targets.reshape(-1)
        
        leptoquark = dataset['leptoquark']
        ato4l = dataset['ato4l']
        hChToTauNu = dataset['hChToTauNu']
        hToTauTau = dataset['hToTauTau']
        
        #Transformations
        self.transform = transform
        self.target_transform = transforms.ToTensor()
        
        ###Random sample 1/4 (1/8) of the background for labeled data, 1/4 (1/8) of background + signals (1/8) for unlabeled data
        #Random sample the background
        np.random.seed(rand_number)
        size_fraction_background = 1/4
        size_fraction_signal = 1/4 
        labeled_background, unlabeled_background, labeled_targets, unlabeled_targets = train_test_split(background_data, background_targets, test_size=size_fraction_background, train_size =size_fraction_background ,random_state=rand_number)
        
        #Also random sample the signals
        unlabeled_leptoquark, _ = train_test_split(leptoquark, train_size=size_fraction_signal, random_state=rand_number)
        unlabeled_ato4l, _ = train_test_split(ato4l, train_size=size_fraction_signal, random_state=rand_number)
        unlabeled_hChToTauNu, _ = train_test_split(hChToTauNu, train_size=size_fraction_signal, random_state=rand_number)
        unlabeled_hToTauTau, _ = train_test_split(hToTauTau, train_size=size_fraction_signal, random_state=rand_number)
        
        #Shuffle in signals (and their labels for testing) with the unlabeled background
        unlabeled_data = np.concatenate((unlabeled_background, unlabeled_leptoquark, unlabeled_ato4l, unlabeled_hChToTauNu, unlabeled_hToTauTau), axis = 0)
        unlabeled_targets = np.concatenate((unlabeled_targets, np.ones(len(unlabeled_leptoquark),dtype=int)*4, np.ones(len(unlabeled_ato4l),dtype=int)*5,np.ones(len(unlabeled_hChToTauNu),dtype=int)*6,np.ones(len(unlabeled_hToTauTau),dtype=int)*7),axis=0)
        unlabeled_data_shuffled, unlabeled_targets_shuffled = shuffle(unlabeled_data, unlabeled_targets, random_state=rand_number)
       
        if datatype == 'train_labeled':
            self.data = labeled_background
            self.targets = labeled_targets
     
        elif datatype == 'train_unlabeled':
            self.data = unlabeled_data_shuffled
            self.targets = unlabeled_targets_shuffled
            
        elif datatype == 'test':
            self.data = unlabeled_data_shuffled
            self.targets = unlabeled_targets_shuffled
        else:
            warnings.warn('Type of dataset not available')
            return
        
        
        #Reshape the data
        self.targets = self.targets.astype(int)
        self.targets = self.targets.tolist()
        self.data = np.vstack(self.data).reshape(-1, 6)
        
        #Print the shapes of data + targets
        print(np.shape(self.data))
        print(np.shape(self.targets))
        print(type(self.targets[0]))

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        image = torch.from_numpy(image) #Turn numpy array into tensor(6)
        image = (image+0.3938)/3.9901  #Normalize the data to mean: 0, std: 1
        label = self.targets[idx]
        image = self.transform(image)
        #label = self.target_transform(label)
        return image, label
        

class BACKGROUND_SIGNAL_CVAE_LATENT_PYTORCH(Dataset):

    def __init__(self, root, datatype, rand_number=42, transform=None, target_transform=None):
        
        #Import the background + signals + labels
        drive_path = ''
        
        dataset = np.load(drive_path+'embedding.npz')
        
        
        #background_data = dataset['embedding_train']
        background_data = dataset['x_train']
        background_targets = dataset['labels_train']
        background_targets = background_targets.reshape(-1)
        
        #Transformations
        self.transform = transform
        self.target_transform = transforms.ToTensor()
        
        ###Random sample 1/4 (1/8) of the background for labeled data, 1/4 (1/8) of background + signals (1/8) for unlabeled data
        #Random sample the background
        np.random.seed(rand_number)
        size_fraction_background = 1/16
        
        labeled_background, unlabeled_data, labeled_targets, unlabeled_targets = train_test_split(background_data, background_targets, test_size=size_fraction_background, train_size =size_fraction_background ,random_state=rand_number)
        #Get rid of the signal data in labeled_background, labeled_targets for supervised objective
        mask = (labeled_targets < 4)
        labeled_background = labeled_background[mask]
        labeled_targets = labeled_targets[mask]
        
        
        if datatype == 'train_labeled':
            self.data = labeled_background
            self.targets = labeled_targets
     
        elif datatype == 'train_unlabeled':
            self.data = unlabeled_data
            self.targets = unlabeled_targets
            
        elif datatype == 'test':
            self.data = unlabeled_data
            self.targets = unlabeled_targets
        else:
            warnings.warn('Type of dataset not available')
            return
        
        
        #Reshape the data
        self.targets = self.targets.astype(int)
        self.targets = self.targets.tolist()
        #self.data = np.vstack(self.data).reshape(-1, 6)
        self.data = np.vstack(self.data).reshape(-1, 57)
        
        #Print the shapes of data + targets
        print(np.shape(self.data))
        print(np.shape(self.targets))
        print(type(self.targets[0]))

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        image = torch.from_numpy(image) #Turn numpy array into tensor(6)
        #image = (image+0.33405897)/1.0524634  #Normalize the data to mean: 0, std: 1
        label = self.targets[idx]
        image = self.transform(image)
        #label = self.target_transform(label)
        return image, label
    

# Dictionary of transforms
dict_transform = {
    'cifar_train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]),
    'cifar_test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]),
    'cifar_train_kyle': transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((2.197), (10.79)), #Normalization from just the background
        #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]),
    'cifar_test_kyle': transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        transforms.Normalize((2.197), (10.79)), #Normalization from just the background
    ]),
    'cifar_train_kyle_cvae': transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        #transforms.ToTensor(),
        #transforms.Normalize((2.197), (10.79)), #Normalization from just the background
        #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]),
    'cifar_test_kyle_cvae': transforms.Compose([
        #transforms.ToTensor(),
        #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        #transforms.Normalize((2.197), (10.79)), #Normalization from just the background
    ])
}
