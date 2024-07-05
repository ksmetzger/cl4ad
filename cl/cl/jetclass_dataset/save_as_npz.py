import numpy as np
from dataloader import read_file
import os
from sklearn.utils import shuffle
import random

def background_higgs_signal_testset(random_state, filename):
    #Download with getDatasets.py (from test 20M dataset) for downstream tasks (ORCA)
    #and only look at particle kinematics for now: pT, eta, phi, E
    #Background: ZJetsToNuNu: 0
    #Signal: Higgs-boson decays HToBB: 1, HToCC: 2, HToGG: 3, HToWW4Q: 4, HToWW2Q1L: 5
    #The output testset is shuffled and preprocessed but not divided into train/val/test -> only for test purposes.

    #Read the JetClass files and save them as numpy npz for further processing
    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/datasets/JetClass/Pythia/test_20M/'
    files = [f for f in os.listdir('datasets/JetClass/Pythia/test_20M/')]
    #List all the particle features wanted
    part_feat = ['part_pt', 'part_eta', 'part_phi', 'part_energy']
    background_events = []
    background_labels = []
    signal_events = []
    signal_labels = []

    #Get background (2M)
    for file_name in files:
        if "ZJetsToNuNu_" in file_name:
            print(file_name)
            file_path = os.path.join(dir_path, file_name)
            x_particles, x_jet, y = read_file(file_path, max_num_particles=128, particle_features=part_feat)
            print(np.shape(x_particles))
            print(np.shape(y))
            print(y[0])
            #Convert labels from one-hot encoded -> normal
            y = np.argmax(y, axis=1)
            print(y[0])
            background_events.append(x_particles.reshape(-1,128,4))
            background_labels.append(y)

    #Signal with signal ratio lambda for each higgs signal
    lmda = 0.05 #(100'000) = one file of signal
    for file_name in files:
        if "H" in file_name and "100" in file_name:
            print(file_name)
            file_path = os.path.join(dir_path, file_name)
            x_particles, x_jet, y = read_file(file_path, max_num_particles=128, particle_features=part_feat)
            print(np.shape(x_particles))
            print(np.shape(y))
            print(y[0])
            #Convert labels from one-hot encoded -> normal
            y = np.argmax(y, axis=1)
            print(y[0])
            signal_events.append(x_particles.reshape(-1,128,4))
            signal_labels.append(y)

    
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

    #Preprocess
    mixture_events = preprocess(mixture_events).reshape(-1,512)

    #Save as .npz file
    np.savez(filename,
            x_test=mixture_events,
            labels_test=mixture_labels,
            )
    print(f'{filename} successfully saved')

def mock_dataset(random_state, filename):
    #Download with getDatasets.py (only val 5M due to size constraints)
    #and only look at particle kinematics for now: pT, eta, phi, E

    #Background: ZJetsToNuNu: 0, ZToQQ: 1, WToQQ: 2, TTBar: 3, TTBarLep: 4
    #Signal: Higgs-boson decays HToBB: 5, HToCC: 6, HToGG: 7, HToWW4Q: 8, HToWW2Q1L: 9

    #Read the JetClass files and save them as numpy npz for further processing
    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/datasets/JetClass/Pythia/val_5M/'
    files = [f for f in os.listdir('datasets/JetClass/Pythia/val_5M/')]

    #List all the particle features wanted
    #part_feat = ['part_pt', 'part_eta', 'part_phi', 'part_energy', 'part_isChargedHadron', 'part_isNeutralHadron', 'part_isPhoton', 'part_isElectron', 'part_isMuon']
    part_feat = ['part_pt', 'part_eta', 'part_phi', 'part_energy']
    jet_feat=['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 
                  'jet_nparticles', 'jet_sdmass', 'jet_tau1', 'jet_tau2', 'jet_tau3', 'jet_tau4']
    background_events = []
    background_labels = []
    signal_events = []
    signal_labels = []
    background_jet_feat = []
    signal_jet_feat = []
    for file_name in files:
        #file_name = "JetClass_Pythia_train_100M_part0.tar"
        print(file_name)
        file_path = os.path.join(dir_path, file_name)
        x_particles, x_jet, y = read_file(file_path, max_num_particles=128, particle_features=part_feat, jet_features=jet_feat)
        print(np.shape(x_particles))
        print(np.shape(x_jet))
        print(np.shape(y))
        print(y[0])
        #Convert labels from one-hot encoded -> normal
        y = np.argmax(y, axis=1)
        print(y[0])
        if "H" in file_name:
            #signal_events.append(x_particles.reshape(-1,128,4))
            #signal_events.append(x_particles.reshape(-1,64,9))
            signal_jet_feat.append(x_jet.reshape(-1,10))
            signal_labels.append(y)
        else:
            #background_events.append(x_particles.reshape(-1,128,4))
            #background_events.append(x_particles.reshape(-1,64,9))
            background_jet_feat.append(x_jet.reshape(-1,10))
            background_labels.append(y)
        
    print(f"Test the shape: {np.shape(np.array(background_events))}")
    print(f"Test the shape for jet_level features: {np.shape(np.array(background_jet_feat))}")
    #background_events = np.concatenate(background_events, axis = 0)
    background_jet_feat = np.concatenate(background_jet_feat, axis = 0)
    #background_labels = np.concatenate(background_labels, axis = 0)
    #signal_events = np.concatenate(signal_events, axis = 0)
    signal_jet_feat = np.concatenate(signal_jet_feat, axis = 0)
    #signal_labels = np.concatenate(signal_labels, axis = 0)

    #Concatenate background and signal and shuffle (5 background, 5 signal)
    #mixture_events = np.concatenate((background_events, signal_events), axis=0)
    mixture_jet_feat = np.concatenate((background_jet_feat, signal_jet_feat), axis=0)
    #mixture_labels = np.concatenate((background_labels, signal_labels), axis=0)
    #print(np.shape(mixture_events))
    print(np.shape(mixture_jet_feat))
    #print(np.shape(mixture_labels))
    #mixture_events, mixture_labels = shuffle(mixture_events, mixture_labels, random_state=random_state)
    mixture_jet_feat= shuffle(mixture_jet_feat, random_state=random_state)
    #Train-Val-Test Split
    #train_events, val_events, test_events = np.split(mixture_events, 
    #                    [int(.6*len(mixture_events)), int(.8*len(mixture_events))])
    train_jet_feat, val_jet_feat, test_jet_feat = np.split(mixture_jet_feat, 
                        [int(.6*len(mixture_jet_feat)), int(.8*len(mixture_jet_feat))])
    #train_labels, val_labels, test_labels = np.split(mixture_labels, 
    #                    [int(.6*len(mixture_events)), int(.8*len(mixture_events))])
    #Preprocessing
    #train_events = preprocess(train_events)
    #val_events = preprocess(val_events)
    #test_events = preprocess(test_events)

    #Save as .npz file
    #Save the particle level features
    """ np.savez(filename,
            x_train=train_events,
            labels_train=train_labels,
            x_test=test_events,
            labels_test=test_labels,
            x_val=val_events,
            labels_val=val_labels,
            )
    print(f'{filename} successfully saved') """
    #Save the jet level features
    np.savez(filename,
            x_train_jet_feat=train_jet_feat,
            x_test_jet_feat=test_jet_feat,
            x_val_jet_feat=val_jet_feat,
            )

def preprocess(array):
    mean = np.mean(array)
    std = np.std(array)
    print(f"Mean: {mean}")
    print(f"Standard deviation: {std}")
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
    # train_events = train_events.reshape(-1,576)
    # test_events = test_events.reshape(-1,576)
    # val_events = val_events.reshape(-1,576)
    #Rename
    #before labels=['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl'])
    remap = {0: 0, 1:5, 2:6, 3:7, 4:8, 5:9, 6:1, 7:2, 8:3, 9:4}
    #To compare
    print(train_labels[432:444])
    #Replace according to the dictionary
    train_labels = np.array([remap[x] for x in train_labels])
    test_labels = np.array([remap[x] for x in test_labels])
    val_labels = np.array([remap[x] for x in val_labels])
    print(train_labels[432:444])

    """ np.savez('JetClass_background_signal_reshaped_extended.npz',
            x_train=train_events,
            labels_train=train_labels,
            x_test=test_events,
            labels_test=test_labels,
            x_val=val_events,
            labels_val=val_labels,
            ) """
    #print(f'{filename} successfully saved')


def main():
    random_state = 0
    np.random.seed(random_state)
    random.seed(random_state)
    #filename = 'JetClass_background_signal_extended.npz'
    #filename = 'JetClass_background_higgs_signal_testset.npz'
    filename = 'JetClass_jet_level_features.npz'
    mock_dataset(random_state=random_state, filename=filename)
    #reshape_and_rename(filename=filename)
    #background_higgs_signal_testset(random_state=random_state, filename=filename)


if __name__== '__main__':
    main()