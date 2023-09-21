print("Importing from 'data_preprocessing.py'")
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras

def save_normalization_weights(input_array, filename): 
    '''
    Assumes max_pt preprocessing. Saves max pT value in 'config.npy' file under 'filename' key. Returns nothing. 
    '''
    # Loads input as tensor and computes maximum pT value across all entries
    tensor = tf.convert_to_tensor(input_array, dtype = tf.float32)
    max_pt = tf.reduce_max(tensor[:, :, 0, :])

    # Creates configs.npy if dne, otherwise, updates existing dict
    try: config_dict = np.load("configs.npy", allow_pickle=True).item()
    except FileNotFoundError: config_dict = {}
        
    # Stores maximum pt value in configs.npz dict. 
    config_dict[filename] = max_pt
    np.save('configs.npy', config_dict)
    
    
def maxPT_preprocess(input_array, filename):
    '''
    Normalizes using max pT across pT only ->  x' = (x) / max(pT)
    Assumes pT saved in config.py file under filename key
    '''
    # Loads input as tensor and (μ, σ) from configs. Applies zscore scaling
    tensor = tf.convert_to_tensor(input_array, dtype = tf.float32)
    configs_dict = np.load('configs.npy', allow_pickle=True).item()
    max_pT = configs_dict[filename]
    normalized_tensor = tf.math.divide_no_nan(tensor, max_pT)

    # Masking so unrecorded data remains 0
    mask = tf.math.not_equal(tensor, 0)
    mask = tf.cast(mask, tf.float32)
    mask = tf.squeeze(mask, -1)
    
    # Outputs normalized pT while preserving original values for eta and phi. Applies mask 
    outputs = tf.concat([normalized_tensor[:,:,0,:], tensor[:,:,1,:], tensor[:,:,2,:]], axis=2)
    return tf.reshape(outputs * mask, (-1, 57))
    
    
def zscore_preprocess(input_array, train=False, scaling_file=None):
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



def linear_preprocess(input_array):
    '''
    Normalizing using linear scaling along pT only -> x'= (x-xmin) / (xmax - xmin)
    '''
    tensor = tf.convert_to_tensor(input_array, dtype = tf.float32)
    min_val = tf.reduce_min(tensor, axis=1, keepdims=True)
    max_val = tf.math.reduce_max(tensor, axis=1, keepdims=True)
    range_val = max_val - min_val
    normalized_tensor = tf.math.divide_no_nan((tensor - min_val), range_val)

    mask = tf.math.not_equal(tensor, 0)
    mask = tf.cast(mask, tf.float32)
    mask = tf.squeeze(mask, -1)

    outputs = tf.concat([normalized_tensor[:,:,0,:], tensor[:,:,1,:], tensor[:,:,2,:]], axis=2)
    return tf.reshape(outputs * mask, (-1, 57))


def log_preprocess(input_array):
    '''
    Normalizing using log scaling along pT only -> x' = log(x+1)
    '''
    tensor = tf.convert_to_tensor(input_array, dtype = tf.float32)
    normalized_tensor = tf.math.log(tensor+1) # +1 bc log(0) = nan

    mask = tf.math.not_equal(tensor, 0)
    mask = tf.cast(mask, tf.float32)
    mask = tf.squeeze(mask, -1)

    outputs = tf.concat([normalized_tensor[:,:,0,:], tensor[:,:,1,:], tensor[:,:,2,:]], axis=2)
    return tf.reshape(outputs * mask, (-1, 57))


def phi_shift(input_array):
    '''
    Shifts phi values from range [0, 2π] to [-π, π]
    '''
    # Creates mask of -2pi were input array should be shifted and 0 otherwise 
    pi_value = tf.constant(np.pi, dtype=input_array.dtype)
    mask = tf.where(input_array > pi_value, -2 * pi_value, tf.zeros_like(input_array))
    mask = tf.cast(mask, dtype=input_array.dtype)
    outputs = input_array + mask
    
    # Concatinates shifted array with original so as not to modify pT, ƞ attributes 
    outputs = tf.concat([input_array[:,:,0,:], input_array[:,:,1,:], outputs[:,:,2,:]], axis= -1)
    return outputs 


def transform_raw_cms(cms_features): 
    '''
    Given raw cms data of form (N, 33, 3) -> (1 MET, 12 jets, 8 muons, 12 electrons)
    Returns transformed data of form (N, 19, 3) -> (1 MET, 4 electrons, 4 muons, 10 jets)
    '''    
    cms_met = cms_features[:, 0:1, :]
    cms_jet = cms_features[:, 1:11, :]
    cms_muons = cms_features[:, 13:17, :]
    cms_etron = cms_features[:, 21:25, :]
    
    transformed_features = np.concatenate([cms_met, cms_etron, cms_muons, cms_jet], axis=1)
    return transformed_features 
    
    
def convert_to_float(data, inverse: bool = False, force: bool = False):
    """
    Convert pt, eta, phi from integer repr to float repr. Ratios are hard coded and taken form
    `https://github.com/thaarres/mp7_ugt_legacy/blob/anomaly_detection_trigger/firmware/hls/data_types.h`
    Assumes data is formated as [MET*1, egamma*4, muon*4, jet*10]
    """
    # Calculates constants for each feature 
    MUON_PHI_SCALER = 2 * np.pi / 576
    CALO_PHI_SCALER = 2 * np.pi / 144
    MUON_ETA_SCALER = 0.0870 / 8
    CALO_ETA_SCALER = 0.0870 / 2
    PT_CALO_SCALER  = 0.5
    PT_MUON_SCALER = 0.5

    # For each feature, creates (19, 1) array to scale along event axis 
    pt_scale = [PT_CALO_SCALER] * 5 + [PT_MUON_SCALER] * 4 + [PT_CALO_SCALER] * 10
    eta_scale = [CALO_ETA_SCALER] * 5 + [MUON_ETA_SCALER] * 4 + [CALO_PHI_SCALER] * 10
    phi_scale = [CALO_PHI_SCALER] * 5 + [MUON_PHI_SCALER] * 4 + [CALO_PHI_SCALER] * 10

    scaler = np.array([pt_scale, eta_scale, phi_scale]).T
    return data * scaler 
