import glob, h5py, math, time, os, json, argparse, datetime
import numpy as np
from FLKutils import *
from SampleUtils import *
from models import *
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser
from transformer import TransformerEncoder
##########
#Implementation based on: https://github.com/GaiaGrosso/NPLM-embedding
##########
def main(args):
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
    with h5py.File(f'ADC_Delphes_original_divisions.hdf5', 'r') as f:
        x_test = np.array(f['x_test'][...])
        labels_test = np.array(f['labels_test'][...])
    print(f"Background dataset contains {len(labels_test)} events!")
    features = x_test
    target = labels_test
    #Blackbox
    features_BB = np.load(f"blackbox/BBsplits57/BBsplits57/split{args.slice}.npy").reshape(-1,19,3,1)
    target_BB = np.ones(len(features_BB))*4
    #Preprocess the black box with the same parameters, saved in file scaling_file.npz
    features_BB = zscore_preprocess(features_BB, train=False, scaling_file="scaling_file.npz")
    ###Inference to get the embedding
    if args.inference:
        model_name = args.model_name
        features_BKG = inference(model_name, features, target).astype(np.float64)
        features_BB = inference(model_name, features_BB, target_BB).astype(np.float64)
    ###
    #Print shapes
    print(f'Shape of the features: {np.shape(features)}')
    print(f'and of type: {type(features)}')
    print(f'Shape of the target: {np.shape(target)}')
    print(f'and of type: {type(target)}')


    ######## standardizes data
    print('standardize')
    features_mean, features_std = np.mean(features_BKG, axis=0), np.std(features_BKG, axis=0)
    print('mean: ', features_mean)
    print('std: ', features_std)
    features_mean_BB, features_std_BB = np.mean(features_BB, axis=0), np.std(features_BB, axis=0)
    print('mean: ', features_mean_BB)
    print('std: ', features_std_BB)
    features_BKG = standardize(features_BKG, features_mean, features_std)
    features_BB = standardize(features_BB, features_mean_BB, features_std_BB)

    #### compute sigma hyper parameter from data
    #### sigma is the gaussian kernels width. 
    #### Following a heuristic, we set this hyperparameter to the 90% quantile of the distribution of pair-wise distances between bkg-distributed points
    #### (see below)
    #### This doesn't need modifications, but one could in principle change it (see https://arxiv.org/abs/2408.12296)
    flk_sigma = candidate_sigma(features_BKG[:2000, :], perc=flk_sigma_perc)
    print('flk_sigma', flk_sigma)

    N_ref = args.size_ref # number of reference datapoints (mixture of non-anomalous classes)
    N_bkg = args.size_back # number of background datapoints in the data (mixture of non-anomalous classes present in the data)
    #N_sig = 0 # number of signal datapoints in the data (mixture of anomalous classes present in the data)
    w_ref = N_bkg*1./N_ref

    xlabels = [f"dim_{i}" for i in range(4)]
    ## run toys
    print('Start running toys')
    t0=np.array([])
    seeds = np.random.uniform(low=1, high=100000, size=(Ntoys,))
    for i in range(Ntoys):
        seed = int(seeds[i])
        rng = np.random.default_rng(seed=seed)
        N_bkg_p = rng.poisson(lam=N_bkg, size=1)[0]
        #N_sig_p = rng.poisson(lam=N_sig, size=1)[0]
        #rng.shuffle(features_SIG)
        rng.shuffle(features_BKG)
        #features_s = features_SIG[:N_sig_p, :]
        features = features_BKG[:N_bkg_p+N_ref, :]

        label_R = np.zeros((N_ref,))
        label_D = np.ones((N_bkg_p, ))
        labels  = np.concatenate((label_D,label_R), axis=0)
        
        plot_reco=False
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
    #N_bkg = args.size_back # number of backgroun datapoints in the data (mixture of non-anomalous classes present in the data)
    #N_sig = args.size_sig # number of signal datapoints in the data (mixture of anomalous classes present in the data)
    N_BB = len(features_BB)
    w_ref = N_BB*1./N_ref


    ## run toys
    print('Start running toys')
    t1=np.array([])
    seeds = np.random.uniform(low=1, high=100000, size=(Ntoys,))
    for i in range(Ntoys):
        seed = int(seeds[i])
        rng = np.random.default_rng(seed=seed)
        #N_bkg_p = rng.poisson(lam=N_bkg, size=1)[0]
        #N_sig_p = rng.poisson(lam=N_sig, size=1)[0]
        rng.shuffle(features_BB)
        rng.shuffle(features_BKG)
        #features_s = features_SIG[:N_sig_p, :]
        features_b = features_BKG[:N_ref, :]
        features  = np.concatenate((features_BB,features_b), axis=0)

        label_R = np.zeros((N_ref,))
        label_D = np.ones((N_BB, ))
        labels  = np.concatenate((label_D,label_R), axis=0)
        
        plot_reco=False
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
    folder_out = f'./out/{args.output_name}/'
    string = f'BB-SL{args.slice}'
    NP = '%s_NR%i_NB%i_NBB%i_M%i_lam%s_iter%i/'%(string, N_ref, N_bkg, N_BB,
                                                    M, str(lam), iterations)
    if not os.path.exists(folder_out+NP):
        os.makedirs(folder_out+NP)

    np.save(f'{folder_out+NP}/t0.npy', t0)
    np.save(f'{folder_out+NP}/t1.npy', t1)

#Run inference given a pytorch model and the model weights for the test dataset
def inference(model_name, input_data, input_labels, device=None):
    '''
    Inference for test input with dimensionality (-1, 57) using model SimpleDense().
    '''
    if device == None:
        device = torch.device('cpu')
    else: 
        device = device
    #Import model and model weights for the embedding
    transformer_args_standard = dict(
        input_dim=3, 
        model_dim=64, 
        output_dim=64, 
        embed_dim=4,
        n_heads=8, 
        dim_feedforward=256, 
        n_layers=4,
        hidden_dim_dino_head=256,
        bottleneck_dim_dino_head=64,
        pos_encoding = True,
        use_mask = False,
        mode='cls',
        dropout=0.025,
        )

    ###model = SimpleDense_ADC(latent_dim=6).to(device)
    model = TransformerEncoder(**transformer_args_standard)
    model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))
    model.eval()
    #Get output with dataloader
    data_loader = DataLoader(
        TorchCLDataset(input_data, input_labels, device),
        batch_size=1024,
        shuffle=False)
    with torch.no_grad():
        output = np.concatenate([model.representation(data.reshape(-1,19,3).to(device), data.reshape(-1,19,3).to(device)).cpu().detach().numpy() for (data, label) in data_loader], axis=0)
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
        print(f"Mu: {mu}, sigma: {sigma}")
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
  
if __name__=='__main__':
    # Parses terminal command
    parser = ArgumentParser()
    #Whether to do inference for embedding the dataset
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--model_name', default="models/runs247/vae2.pth", type=str)
    parser.add_argument('--output_name', default="runs247", type=str)
    parser.add_argument('--slice', default=0, type=int)
    parser.add_argument('--n_toys', default=100, type=int)
    parser.add_argument('--size_ref', default=1000000, type=int)
    parser.add_argument('--size_back', default=100000, type=int)
    #parser.add_argument('--size_sig', default=1000, type=int)
    args = parser.parse_args()
    main(args)

