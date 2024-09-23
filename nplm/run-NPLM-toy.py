import glob, h5py, math, time, os, json, argparse, datetime
import numpy as np
from FLKutils import *
from SampleUtils import *
from models import *
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser

def main(args):
    # labels identifying the signal classes in the dataset
    sig_labels=[args.signal]
    # labels identifying the background classes in the dataset
    bkg_labels=[0, 1, 2, 3]

    # hyper parameters of the NPLM model based on kernel methods
    ## number of kernels
    M = 1000 

    ## percentile of the distribution of pair-wise distances between reference-distributed points
    flk_sigma_perc=90 

    ## L2 regularization coefficient
    lam =1e-6 

    ## number of maximum iterations before the training is killed
    iterations=1000000 

    ## number of toys to simulate 
    ## (multiple experiments allow you to build a statistics for the test and quantify type I and II errors)
    Ntoys = args.n_toys

    ###Load the dataset
    #Background
    with h5py.File(f'/eos/user/k/kmetzger/test/dino/ADC_Delphes_original_divisions.hdf5', 'r') as f:
        x_test = np.array(f['x_test'][...])
        labels_test = np.array(f['labels_test'][...])
    print(f"Background dataset contains {len(labels_test)} events!")
    #Signal
    with h5py.File(f'/eos/user/k/kmetzger/test/dino/ADC_Delphes_signals.hdf5', 'r') as f:
        leptoquark = np.array(f['leptoquark'][...])
        leptoquark_labels = np.array(f['leptoquark_labels'][...])
        ato4l = np.array(f['ato4l'][...])
        ato4l_labels = np.array(f['ato4l_labels'][...])
        hChToTauNu = np.array(f['hChToTauNu'][...])
        hChToTauNu_labels = np.array(f['hChToTauNu_labels'][...])
        hToTauTau = np.array(f['hToTauTau'][...])
        hToTauTau_labels = np.array(f['hToTauTau_labels'][...])
    #Concatenate the features
    features = np.concatenate((x_test, leptoquark, ato4l, hChToTauNu, hToTauTau), axis=0)
    target = np.concatenate((labels_test, leptoquark_labels, ato4l_labels, hChToTauNu_labels, hToTauTau_labels), axis=0)
    ###
    ###Inference to get the embedding
    if args.inference:
        model_name = args.model_name
        features = inference(model_name, features, target).astype(np.float64)
    ###
    #Print shapes
    print(f'Shape of the features: {np.shape(features)}')
    print(f'and of type: {type(features)}')
    print(f'Shape of the target: {np.shape(target)}')
    print(f'and of type: {type(target)}')
    # select SIG and BKG classes
    mask_SIG = np.zeros_like(target)
    mask_BKG = np.zeros_like(target)
    for sig_label in sig_labels:
        mask_SIG += 1*(target==sig_label)
    for bkg_label in bkg_labels:
        mask_BKG += 1*(target==bkg_label)

    features_SIG = features[mask_SIG>0]
    features_BKG = features[mask_BKG>0]


    ######## standardizes data
    print('standardize')
    features_mean, features_std = np.mean(features_BKG, axis=0), np.std(features_BKG, axis=0)
    print('mean: ', features_mean)
    print('std: ', features_std)
    features_BKG = standardize(features_BKG, features_mean, features_std)
    features_SIG = standardize(features_SIG, features_mean, features_std)

    #### compute sigma hyper parameter from data
    #### sigma is the gaussian kernels width. 
    #### Following a heuristic, we set this hyperparameter to the 90% quantile of the distribution of pair-wise distances between bkg-distributed points
    #### (see below)
    #### This doesn't need modifications, but one could in principle change it (see https://arxiv.org/abs/2408.12296)
    flk_sigma = candidate_sigma(features_BKG[:2000, :], perc=flk_sigma_perc)
    print('flk_sigma', flk_sigma)

    N_ref = args.size_ref # number of reference datapoints (mixture of non-anomalous classes)
    N_bkg = args.size_back # number of backgroun datapoints in the data (mixture of non-anomalous classes present in the data)
    N_sig = 0 # number of signal datapoints in the data (mixture of anomalous classes present in the data)
    w_ref = N_bkg*1./N_ref

    xlabels = ["dim0","dim1","dim2","dim3","dim4","dim5"]
    ## run toys
    print('Start running toys')
    t0=np.array([])
    seeds = np.random.uniform(low=1, high=100000, size=(Ntoys,))
    for i in range(Ntoys):
        seed = int(seeds[i])
        rng = np.random.default_rng(seed=seed)
        N_bkg_p = rng.poisson(lam=N_bkg, size=1)[0]
        N_sig_p = rng.poisson(lam=N_sig, size=1)[0]
        rng.shuffle(features_SIG)
        rng.shuffle(features_BKG)
        features_s = features_SIG[:N_sig_p, :]
        features_b = features_BKG[:N_bkg_p+N_ref, :]
        features  = np.concatenate((features_s,features_b), axis=0)

        label_R = np.zeros((N_ref,))
        label_D = np.ones((N_bkg_p+N_sig_p, ))
        labels  = np.concatenate((label_D,label_R), axis=0)
        
        plot_reco=True
        verbose=False
        # make reconstruction plots every 20 toys (can be changed)
        #if not i%20:
        #    plot_reco=True
        #    verbose=True
        flk_config = get_logflk_config(M,flk_sigma,[lam],weight=w_ref,iter=[iterations],seed=None,cpu=False)
        t, pred = run_toy('t0', features, labels, weight=w_ref, seed=seed,
                        flk_config=flk_config, output_path='./', plot=plot_reco, savefig=plot_reco,
                        verbose=verbose, xlabels=xlabels)
        
        t0 = np.append(t0, t)


    N_ref = args.size_ref # number of reference datapoints (mixture of non-anomalous classes)
    N_bkg = args.size_back # number of backgroun datapoints in the data (mixture of non-anomalous classes present in the data)
    N_sig = args.size_sig # number of signal datapoints in the data (mixture of anomalous classes present in the data)
    w_ref = N_bkg*1./N_ref


    ## run toys
    print('Start running toys')
    t1=np.array([])
    seeds = np.random.uniform(low=1, high=100000, size=(Ntoys,))
    for i in range(Ntoys):
        seed = int(seeds[i])
        rng = np.random.default_rng(seed=seed)
        N_bkg_p = rng.poisson(lam=N_bkg, size=1)[0]
        N_sig_p = rng.poisson(lam=N_sig, size=1)[0]
        rng.shuffle(features_SIG)
        rng.shuffle(features_BKG)
        features_s = features_SIG[:N_sig_p, :]
        features_b = features_BKG[:N_bkg_p+N_ref, :]
        features  = np.concatenate((features_s,features_b), axis=0)

        label_R = np.zeros((N_ref,))
        label_D = np.ones((N_bkg_p+N_sig_p, ))
        labels  = np.concatenate((label_D,label_R), axis=0)
        
        plot_reco=True
        verbose=False
        # make reconstruction plots every 20 toys (can be changed)
        #if not i%20:
        #    plot_reco=True
        #    verbose=True
        flk_config = get_logflk_config(M,flk_sigma,[lam],weight=w_ref,iter=[iterations],seed=None,cpu=False)
        t, pred = run_toy('t1', features, labels, weight=w_ref, seed=seed,
                        flk_config=flk_config, output_path='./', plot=plot_reco, savefig=plot_reco,
                        verbose=verbose, xlabels=xlabels)
        
        t1 = np.append(t1, t)


    ## details about the save path                                                                                                                               
    folder_out = './out/'
    sig_string = ''
    if N_sig:
        sig_string+='_SIG'
        for s in sig_labels:
            sig_string+='-%i'%(s)
    NP = '%s_NR%i_NB%i_NS%i_M%i_lam%s_iter%i/'%(sig_string, N_ref, N_bkg, N_sig,
                                                    M, str(lam), iterations)
    if not os.path.exists(folder_out+NP):
        os.makedirs(folder_out+NP)

    np.save(f'{folder_out+NP}/t0.npy', t0)
    np.save(f'{folder_out+NP}/t1.npy', t1)

def inference(model_name, input_data, input_labels, device=None):
    '''
    Inference for test input with dimensionality (-1, 57) using model SimpleDense().
    '''
    if device == None:
        device = torch.device('cpu')
    else: 
        device = device
    #Import model for embedding
    #model = SimpleDense(latent_dim=12).to(device)
    #model = SimpleDense_small().to(device)
    #model = DeepSets(latent_dim=6).to(device)
    #model = CVAE(latent_dim=6).to(device)
    #model = SimpleDense_JetClass(latent_dim=6).to(device)
    #model = CVAE_JetClass(latent_dim=6).to(device)
    model = SimpleDense_ADC(latent_dim=6).to(device)
    model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))
    model.eval()
    #Get output with dataloader
    data_loader = DataLoader(
        TorchCLDataset(input_data, input_labels, device),
        batch_size=1024,
        shuffle=False)
    with torch.no_grad():
        output = np.concatenate([model.representation(data.to(device)).cpu().detach().numpy() for (data, label) in data_loader], axis=0)
    return output

class TorchCLDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, features, labels, device):
        'Initialization'
        self.device = device
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
  
if __name__=='__main__':
    # Parses terminal command
    parser = ArgumentParser()
    #Whether to do inference for embedding the dataset
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--model_name', default="models/runs247/vae2.pth", type=str)
    #TODO: Add the pars arguments for ref, back, sig sizes and kernel hyperparams
    parser.add_argument('--signal', default=4, type=int)
    parser.add_argument('--n_toys', default=100, type=int)
    parser.add_argument('--size_ref', default=1000000, type=int)
    parser.add_argument('--size_back', default=100000, type=int)
    parser.add_argument('--size_sig', default=1000, type=int)
    args = parser.parse_args()
    main(args)

