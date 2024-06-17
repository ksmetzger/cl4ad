import numpy as np
from dataloader import read_file
import os
from sklearn.utils import shuffle
import random

def mock_dataset(random_state, filename):
    #Download with getDatasets.py (only val 5M due to size constraints)
    #and only look at particle kinematics for now: pT, eta, phi, E

    #Background: ZJetsToNuNu: 0, ZToQQ: 1, WToQQ: 2, TTBar: 3, TTBarLep: 4
    #Signal: Higgs-boson decays HToBB: 5, HToCC: 6, HToGG: 7, HToWW4Q: 8, HToWW2Q1L: 9

    #Read the JetClass files and save them as numpy npz for further processing
    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/datasets/JetClass/Pythia/val_5M/'
    files = [f for f in os.listdir('datasets/JetClass/Pythia/val_5M/')]

    background_events = []
    background_labels = []
    signal_events = []
    signal_labels = []
    for file_name in files:
        #file_name = "JetClass_Pythia_train_100M_part0.tar"
        print(file_name)
        file_path = os.path.join(dir_path, file_name)
        x_particles, x_jet, y = read_file(file_path)
        print(np.shape(x_particles))
        print(np.shape(y))
        print(y[0])
        #Convert labels from one-hot encoded -> normal
        y = np.argmax(y, axis=1)
        print(y[0])
        if "H" in file_name:
            signal_events.append(x_particles.reshape(-1,128,4))
            signal_labels.append(y)
        else:
            background_events.append(x_particles.reshape(-1,128,4))
            background_labels.append(y)
        
    print(f"Test the shape: {np.shape(np.array(background_events))}")
    background_events = np.concatenate(background_events, axis = 0)
    background_labels = np.concatenate(background_labels, axis = 0)
    signal_events = np.concatenate(signal_events, axis = 0)
    signal_labels = np.concatenate(signal_labels, axis = 0)
    #Concatenate background and signal and shuffle (5 background, 5 signal)
    mixture_events = np.concatenate((background_events, signal_events), axis=0)
    mixture_labels = np.concatenate((background_labels, signal_labels), axis=0)
    print(np.shape(mixture_events))
    print(np.shape(mixture_labels))
    mixture_events, mixture_labels = shuffle(mixture_events, mixture_labels, random_state=random_state)
    #Train-Val-Test Split
    train_events, val_events, test_events = np.split(mixture_events, 
                        [int(.6*len(mixture_events)), int(.8*len(mixture_events))])
    train_labels, val_labels, test_labels = np.split(mixture_labels, 
                        [int(.6*len(mixture_events)), int(.8*len(mixture_events))])
    #Preprocessing
    train_events = preprocess(train_events)
    val_events = preprocess(val_events)
    test_events = preprocess(test_events)

    #Save as .npz file
    np.savez(filename,
            x_train=train_events,
            labels_train=train_labels,
            x_test=test_events,
            labels_test=test_labels,
            x_val=val_events,
            labels_val=val_labels,
            )
    print(f'{filename} successfully saved')

def preprocess(array):
    mean = np.mean(array)
    std = np.std(array)
    assert std != 0.0

    return (array-mean)/std

def reshape_and_rename(filename):
    dataset = np.load(filename)
    train_events = dataset['x_train']
    test_events = dataset['x_test']
    val_events = dataset['x_val']
    train_labels = dataset['labels_train']
    test_labels = dataset['labels_test']
    val_labels = dataset['labels_val']
    print(np.shape(train_events))
    print(np.shape(test_events))
    print(np.shape(val_events))
    print(np.shape(train_labels))
    print(np.shape(test_labels))
    print(np.shape(val_labels))
    #Reshape
    train_events = train_events.reshape(-1,512)
    test_events = test_events.reshape(-1,512)
    val_events = val_events.reshape(-1,512)
    #Rename
    #before labels=['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl'])
    remap = {0: 0, 1:5, 2:6, 3:7, 4:8, 5:9, 6:1, 7:2, 8:3, 9:4}
    #To compare
    print(train_labels[432:444])
    #Replace according to the dictionary
    train_labels = [remap[x] for x in train_labels]
    test_labels = [remap[x] for x in test_labels]
    val_labels = [remap[x] for x in val_labels]
    print(train_labels[432:444])

    np.savez('JetClass_background_signal_reshaped.npz',
            x_train=train_events,
            labels_train=train_labels,
            x_test=test_events,
            labels_test=test_labels,
            x_val=val_events,
            labels_val=val_labels,
            )
    print(f'{filename} successfully saved')


def main():
    random_state = 0
    np.random.seed(random_state)
    random.seed(random_state)
    filename = 'JetClass_background_signal.npz'
    #mock_dataset(random_state=random_state, filename=filename)
    reshape_and_rename(filename=filename)

if __name__== '__main__':
    main()