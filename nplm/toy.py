import glob, h5py, math, time, os, json, argparse, datetime
import numpy as np
from FLKutils import *
from SampleUtils import *

# give a name to each model and provide a path to where the model's prediction for bkg and signal classes are stored
folders = {
    'r_2': '/n/home11/nswood/NonE_AD/AE_PM_Contrastive/r_2/plots/',
    'r_4': '/n/home11/nswood/NonE_AD/AE_PM_Contrastive/r_4/plots/',
    'h_2': '/n/home11/nswood/NonE_AD/AE_PM_Contrastive/h_2/plots/',
    'h_4': '/n/home11/nswood/NonE_AD/AE_PM_Contrastive/h_4/plots/',
    'r_16': '/n/home11/nswood/NonE_AD/AE_PM_Contrastive/r_16/plots/',
    'r_8_h_8': '/n/home11/nswood/NonE_AD/AE_PM_Contrastive/r_8_h_8/plots/',
}

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--manifold', type=str, help="manifold type (must be in folders.keys())", required=True)
parser.add_argument('-s', '--signal', type=int, help="signal (number of signal events)", required=True)
parser.add_argument('-b', '--background', type=int, help="background (number of background events)", required=True)
parser.add_argument('-r', '--reference', type=int, help="reference (number of reference events, must be larger than background)", required=True)
parser.add_argument('-t', '--toys', type=int, help="toys", required=True)
parser.add_argument('-l', '--signalclass', type=int, help="class number identifying the signal", required=True)
args = parser.parse_args()

manifold = args.manifold

N_ref = args.reference
N_bkg = args.background
N_sig = args.signal
w_ref = N_bkg*1./N_ref

# class number identifying the signal
sig_labels=[args.signalclass]
# class number identifying the background
bkg_labels=[0, 1, 2, 3, 4, 7, 8, 9]

# hyper parameters of the model
M=10000
flk_sigma_perc=90 #%
lam =1e-6
iterations=1000000
Ntoys = args.toys

# details about the save path
folder_out = '/n/home00/ggrosso/NonE_AD/NPLM/out/'
sig_string = ''
if N_sig:
    sig_string+='_SIG'
    for s in sig_labels:
        sig_string+='-%i'%(s)
NP = '%s%s_NR%i_NB%i_NS%i_M%i_lam%s_iter%i/'%(manifold, sig_string, N_ref, N_bkg, N_sig,
                                                  M, str(lam), iterations)
if not os.path.exists(folder_out+NP):
    os.makedirs(folder_out+NP)

############ begin load data
# This part needs to be modified according to how the predictions of your model are stored.
# Here the predictions are saved in npz files
print('Load data')
geom_list = ['Euclidean', 'PoincareBall']
data= {}
for file in glob.glob(folders[manifold]+'/*.npz'):
    filename = file.split('/')[-1].replace('.npz', '')
    data[filename]= {}
    file_load = np.load(file)
    for k2 in file_load.files:
        data[filename][k2] = file_load[k2]

features = np.array([])
for geom in geom_list:
    if geom in list(data.keys()):
        if not features.shape[0]: features = data[geom]['latent']
        else: features = np.concatenate((features, data[geom]['latent']), axis=1)

target = data['labels']['labels']
del data

mask_SIG = np.zeros_like(target)
mask_BKG = np.zeros_like(target)
for sig_label in sig_labels:
    mask_SIG += 1*(target==sig_label)
for bkg_label in bkg_labels:
    mask_BKG += 1*(target==bkg_label)

features_SIG = features[mask_SIG>0]
features_BKG = features[mask_BKG>0]
############ end load data

######## standardizes data
print('standardize')
features_mean, features_std = np.mean(features_BKG, axis=0), np.std(features_BKG, axis=0)
print('mean: ', features_mean)
print('std: ', features_std)
features_BKG = standardize(features_BKG, features_mean, features_std)
features_SIG = standardize(features_SIG, features_mean, features_std)

#### compute sigma hyper parameter from data
#### (This doesn't need modifications)
flk_sigma = candidate_sigma(features_BKG[:2000, :], perc=flk_sigma_perc)
print('flk_sigma', flk_sigma)

## run toys
print('Start running toys')
ts=np.array([])
seeds = np.arange(Ntoys)+datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
for i in range(Ntoys):
    seed = seeds[i]
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
    
    plot_reco=False
    verbose=False
    # make reconstruction plots every 20 toys (can be changed)
    if not i%20:
        plot_reco=True
        verbose=True
    flk_config = get_logflk_config(M,flk_sigma,[lam],weight=w_ref,iter=[iterations],seed=None,cpu=False)
    t, pred = run_toy(manifold, features, labels, weight=w_ref, seed=seed,
                      flk_config=flk_config, output_path='./', plot=plot_reco, savefig=plot_reco,
                      verbose=verbose)
    
    ts = np.append(ts, t)

# collect previous toys if existing
seeds_past = np.array([])
ts_past = np.array([])
if os.path.exists('%s/%s/tvalues_flksigma%s.h5'%(folder_out, NP, flk_sigma)):
    print('collecting previous tvalues')
    f = h5py.File('%s/%s/tvalues_flksigma%s.h5'%(folder_out, NP, flk_sigma), 'r')
    seeds_past = np.array(f.get('seed_toy'))
    ts_past = np.array(f.get(str(flk_sigma) ) )
    f.close()
ts = np.append(ts_past, ts)
seeds = np.append(seeds_past, seeds)

f = h5py.File('%s/%s/tvalues_flksigma%s.h5'%(folder_out, NP, flk_sigma), 'w')
f.create_dataset(str(flk_sigma), data=ts, compression='gzip')
f.create_dataset('seed_toy', data=seeds, compression='gzip')
f.close()
