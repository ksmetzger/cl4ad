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
import augmentations
from models.models_cl import SimpleDense, SimpleDense_small, TransformerEncoder, transformer_args_standard, SimpleDense_JetClass, transformer_args_jetclass, CVAE_JetClass

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
        drive_path = f'C:\\Users\\Kyle\\OneDrive\\Transfer Master project\\orca_fork\\cl4ad\\cl\\cl\\'
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

class BACKGROUND_SIGNAL_DENSE_LATENT(Dataset):

    def __init__(self, root, datatype, rand_number=0, transform=None, target_transform=None):
        
        #Import the background + signals + labels
        drive_path = 'C:\\Users\\Kyle\\OneDrive\\Transfer Master project\\orca_fork\\cl4ad\\cl\\cl\\output\\runs14\\'
        
        dataset = np.load(drive_path+'embedding.npz')
        dataset_signal = np.load(drive_path+'anomalies_embedding.npz')
        
        background_data = dataset['embedding_train']
        background_targets = dataset['labels_train']
        background_targets = background_targets.reshape(-1)
        
        leptoquark = dataset_signal['embedding_leptoquark']
        ato4l = dataset_signal['embedding_ato4l']
        hChToTauNu = dataset_signal['embedding_hChToTauNu']
        hToTauTau = dataset_signal['embedding_hToTauTau']
        
        #Transformations
        self.transform = transform
        self.target_transform = transforms.ToTensor()
        
        ###Random sample 1/4 (1/8) of the background for labeled data, 1/4 (1/8) of background + signals (1/8) for unlabeled data
        #Random sample the background
        np.random.seed(rand_number)
        size_fraction_background = 1/4
        size_fraction_signal = 1/4 
        labeled_background, unlabeled_background, labeled_targets, unlabeled_targets = train_test_split(background_data, background_targets, test_size=size_fraction_background, train_size =size_fraction_background ,random_state=rand_number)
        self.mean_background = np.mean(labeled_background)
        self.std_background = np.std(labeled_background)
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
        self.data = np.vstack(self.data).reshape(-1, 48)
        
        #Print the shapes of data + targets
        print(np.shape(self.data))
        print(np.shape(self.targets))
        print(type(self.targets[0]))

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        image = torch.from_numpy(image) #Turn numpy array into tensor(48)
        image = (image-self.mean_background)/self.std_background  #Normalize the data to mean: 0, std: 1
        label = self.targets[idx]
        image = self.transform(image)
        #label = self.target_transform(label)
        return image, label
    
class BACKGROUND_SIGNAL_INFERENCE_LATENT(Dataset):
    def __init__(self, root, datatype, runs, latent_dim, labeled_num = 4, rand_number=0, finetune=False ,transform=None, target_transform=None, embedding_type = 'SimpleDense'):
        '''
        Define the ORCA input dataclass without explicitly inputting the embedding files but instead doing inference on the saved model weights.
        embedding_type: Describes the architecture used to get the neural embedding choice = ('SimpleDense', 'SimpleDense_small, 'dino_transformer'), default = 'mlp'
        '''
        super(BACKGROUND_SIGNAL_INFERENCE_LATENT, self).__init__()
        #Import the background + signals + labels
        self.runs = runs
        self.latent_dim = latent_dim
        transform_augment = augmentations.Transform([])
        if finetune: #If finetuning start from orginal DELPHES input dimension (57D)
            self.latent_dim = 57
        self.labeled_num = labeled_num
        drive_path = f'C:\\Users\\Kyle\\OneDrive\\Transfer Master project\\orca_fork\\cl4ad\\cl\\cl\\'
        #drive_path_lxplus = os.path.join(os.path.abspath('../dino'), 'dataset_background_signal.npz')
        #Run inference to get the embedding of the test set for further use in ORCA
        dataset = np.load(drive_path+'dataset_background_signal.npz')
        #dataset = np.load(drive_path_lxplus)
        labels_test = dataset['labels_test'].reshape(-1)
        data_test = dataset['x_test']
        
        if embedding_type == 'dino_transformer':
            drive_path = f'C:\\Users\\Kyle\\OneDrive\\Transfer Master project\\orca_fork\\cl4ad\\dino\\'
            drive_path_lxplus = os.path.abspath('../dino/')
            data_test = data_test.reshape(-1,19,3)
        
        if finetune == False:
            embedded_test = self.inference(drive_path + f'output/{self.runs}/', data_test, labels_test, embedding_type=embedding_type, transform_augment=transform_augment)
            #embedded_test = transform_augment(data_test)
        elif finetune:
            embedded_test = data_test
        
        #Transformations
        self.transform = transform
        self.target_transform = transforms.ToTensor()
        
        ###Random sample 1/2 of the background for labeled data, 1/2 of background + signals 1/2 for unlabeled data
        #Random sample the background
        np.random.seed(rand_number)
        size_fraction_background = 1/2
        size_fraction_signal = 1/2
        labeled_background, unlabeled_data, labeled_targets, unlabeled_targets = train_test_split(embedded_test, labels_test, test_size=size_fraction_background, train_size =size_fraction_background ,random_state=rand_number)
        self.mean_background = np.mean(labeled_background)
        self.std_background = np.std(labeled_background)
        
        #Finally mask out the signal parts for the labeled background
        mask = (labeled_targets < self.labeled_num)
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
        #self.data = np.vstack(self.data).reshape(-1, self.latent_dim)
        
        #Print the shapes of data + targets
        print(np.shape(self.data))
        print(np.shape(self.targets))
        print(type(self.targets[0]))

    #Define inference if there is no embedding.npz/anomalies_embedding.npz already saved, in order to use for feeding embedding to ORCA
    def inference(self, model_name, input_data, input_labels, embedding_type, transform_augment, device=None):
        '''
        Inference for test input with dimensionality (-1, 57) using model SimpleDense()/SimpleDense_small().
        '''
        if device == None:
            device = torch.device('cpu')
        else: 
            device = device
        #Import model for embedding depending on embedding_type
        if embedding_type == 'SimpleDense':
            model = SimpleDense(self.latent_dim).to(device)
            model.load_state_dict(torch.load(model_name + 'vae.pth', map_location=torch.device(device)))
        elif embedding_type == 'SimpleDense_small':
            model = SimpleDense_small(self.latent_dim).to(device)
            model.load_state_dict(torch.load(model_name + 'vae.pth', map_location=torch.device(device)))
        elif embedding_type == 'dino_transformer':
            model = TransformerEncoder(**transformer_args_standard)
            model.load_state_dict(torch.load(model_name + '_teacher_dino_transformer.pth', map_location=torch.device(device)))

        model.eval()
        #Get output with dataloader
        data_loader = data.DataLoader(
            TorchCLDataset(input_data, input_labels, device),
            batch_size=1024,
            shuffle=False)
        with torch.no_grad():
            output = np.concatenate([model.representation(transform_augment(data).to(device)).cpu().detach().numpy() for (data, label) in data_loader], axis=0)
        return output
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        image = torch.from_numpy(image) #Turn numpy array into tensor()
        image = (image-self.mean_background)/self.std_background  #Normalize the data to mean: 0, std: 1
        label = self.targets[idx]
        image = self.transform(image)
        #label = self.target_transform(label)
        return image, label
    
class BACKGROUND_SIGNAL_INFERENCE_LATENT_JETCLASS(Dataset):
    def __init__(self, root, datatype, runs, latent_dim, labeled_num = 5, rand_number=0, finetune=False ,transform=None, target_transform=None, embedding_type = 'SimpleDense_JetClass'):
        '''
        Define the ORCA input dataclass without explicitly inputting the embedding files but instead doing inference on the saved model weights.
        embedding_type: Describes the architecture used to get the neural embedding choice = ('SimpleDense', 'SimpleDense_small, 'dino_transformer'), default = 'mlp'
        '''
        super(BACKGROUND_SIGNAL_INFERENCE_LATENT_JETCLASS, self).__init__()
        #Import the background + signals + labels
        self.runs = runs
        self.latent_dim = latent_dim
        #transform_augment = augmentations.Transform([])
        if finetune: #If finetuning start from orginal JetClass input dimension (512D)
            self.latent_dim = 512
        self.labeled_num = labeled_num
        drive_path = f'C:\\Users\\Kyle\\OneDrive\\Transfer Master project\\orca_fork\\cl4ad\\cl\\cl\\'
        #drive_path_lxplus = os.path.join(os.path.abspath('../dino'), 'dataset_background_signal.npz')
        #Run inference to get the embedding of the test set for further use in ORCA
        #dataset = np.load(drive_path+'jetclass_dataset/JetClass_background_signal_reshaped.npz')
        dataset = np.load(drive_path+'jetclass_dataset/JetClass_background_higgs_signal_testset.npz')
        #dataset = np.load(drive_path_lxplus)
        labels_test = dataset['labels_test'].reshape(-1)
        data_test = dataset['x_test'].reshape(-1,512)
        
        if embedding_type == 'dino_transformer_JetClass':
            drive_path = f'C:\\Users\\Kyle\\OneDrive\\Transfer Master project\\orca_fork\\cl4ad\\dino\\'
            drive_path_lxplus = os.path.abspath('../dino/')
            data_test = data_test.reshape(-1,128,4)
        
        if finetune == False:
            embedded_test = self.inference(drive_path + f'output/{self.runs}/', data_test, labels_test, embedding_type=embedding_type)
            #embedded_test = data_test
        elif finetune:
            embedded_test = data_test
        
        #Transformations
        self.transform = transform
        self.target_transform = transforms.ToTensor()
        
        ###Random sample 1/2 of the background for labeled data, 1/2 of background + signals 1/2 for unlabeled data
        #Random sample the background
        np.random.seed(rand_number)
        size_fraction_background = 1/2
        size_fraction_signal = 1/2
        labeled_background, unlabeled_data, labeled_targets, unlabeled_targets = train_test_split(embedded_test, labels_test, test_size=size_fraction_background, train_size =size_fraction_background ,random_state=rand_number)
        self.mean_background = np.mean(labeled_background)
        self.std_background = np.std(labeled_background)
        
        #Finally mask out the signal parts for the labeled background
        mask = (labeled_targets < self.labeled_num)
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
        #self.data = np.vstack(self.data).reshape(-1, self.latent_dim)
        
        #Print the shapes of data + targets
        print(np.shape(self.data))
        print(np.shape(self.targets))
        print(type(self.targets[0]))

    #Define inference if there is no embedding.npz/anomalies_embedding.npz already saved, in order to use for feeding embedding to ORCA
    def inference(self, model_name, input_data, input_labels, embedding_type, device=None):
        '''
        Inference for test input with dimensionality (-1, 57) using model SimpleDense()/SimpleDense_small().
        '''
        if device == None:
            device = torch.device('cpu')
        else: 
            device = device
        #Import model for embedding depending on embedding_type
        if embedding_type == 'SimpleDense_JetClass':
            model = SimpleDense_JetClass(self.latent_dim).to(device)
            #model = CVAE_JetClass(self.latent_dim).to(device)
            model.load_state_dict(torch.load(model_name + 'vae1.pth', map_location=torch.device(device)))
        elif embedding_type == 'dino_transformer_JetClass':
            model = TransformerEncoder(**transformer_args_jetclass)
            model.load_state_dict(torch.load(model_name + '_teacher_dino_transformer.pth', map_location=torch.device(device)))

        model.eval()
        #Get output with dataloader
        data_loader = data.DataLoader(
            TorchCLDataset(input_data, input_labels, device),
            batch_size=1024,
            shuffle=False)
        with torch.no_grad():
            output = np.concatenate([model.representation(data.to(device)).cpu().detach().numpy() for (data, label) in data_loader], axis=0)
        return output
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        image = torch.from_numpy(image) #Turn numpy array into tensor()
        image = (image-self.mean_background)/self.std_background  #Normalize the data to mean: 0, std: 1
        label = self.targets[idx]
        image = self.transform(image)
        #label = self.target_transform(label)
        return image, label
    
class TorchCLDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, features, labels, device):
        'Initialization'
        self.device = device
        #self.mean = np.mean(features)
        #self.std = np.std(features)
        #print(f"Mean: {self.mean} and std: {self.std}")
        self.features = torch.from_numpy(features).to(dtype=torch.float32)
        self.labels = torch.from_numpy(labels).to(dtype=torch.float32)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.features)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.features[index]
        y = self.labels[index]

        return X, y 

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
