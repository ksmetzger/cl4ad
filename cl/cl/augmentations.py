import numpy as np
import torch

def permutation(input_batch, device=None, rand_number=0, same_particle=False):
    '''
    Applies the augmentation "permutation" to events in a batch (torch tensor) and outputs a permutated torch tensor
    Permute constituents in the DELPHES dataset w/ structure: MET, 4x electron, 4x muon, 10x jet.
    Each constituent has 3x features: transverse mom. pT, pseudorapidity eta, azimuthal angle phi.
    Args:
        input_batch: (batch_size, 57) flattened input
        device: cuda or cpu depending on input
        same_particle: (bool) if true only shuffles particles of the same type, otherwise shuffles all constituents
    Returns:
        permutated_batch: (batch_size, 57) permutated output
    '''
    input_numpy = input_batch.cpu().detach().numpy().reshape(-1,19,3)
    #Permute the electrons, muons and jets
    np.random.seed(rand_number) #not sure if I should seed
    if same_particle:
        [np.random.shuffle(x[1:5]) for x in input_numpy] #electrons
        [np.random.shuffle(x[5:9]) for x in input_numpy] #muons
        [np.random.shuffle(x[9:19]) for x in input_numpy] #jets
    else:
        [np.random.shuffle(x[:]) for x in input_numpy] #all constituents
    #Return a torch tensor on the given device and correct shape (-1,57)
    permutated_batch = torch.from_numpy(input_numpy.reshape(-1,57)).to(dtype=torch.float32, device=device)

    return permutated_batch

def rot_around_beamline(input_batch, device=None, rand_number=0):
    '''
    Applies the augmentation "rotation around beamline" to events in a batch (torch tensor) and outputs a permutated torch tensor
    Rotates each event in the batch w/ structure: MET, 4x electron, 4x muon, 10x jet around a random angle.
    Angle around beamline is given by phi in features: transverse mom. pT, pseudorapidity eta, azimuthal angle phi.
    Args:
        input_batch: (batch_size, 57) flattened input
        device: cuda or cpu depending on input
    Returns:
        permutated_batch: (batch_size, 57) permutated output
    '''
    input_numpy = input_batch.cpu().detach().numpy().reshape(-1,19,3)
    #Rotate the whole thing around the beamline (phi) at angle
    np.random.seed(rand_number)
    #Iterate through the batch
    for x in input_numpy:
        #Angles are stored from [-pi,pi]
        angle = np.random.uniform(0,2)*np.pi
        x[:,2] = (((x[:,2]+np.pi) + angle)%(2*np.pi))-np.pi

    #Return a torch tensor on the given device and correct shape (-1,57)
    rotated_batch = torch.from_numpy(input_numpy.reshape(-1,57)).to(dtype=torch.float32, device=device)

    return rotated_batch

def gaussian_resampling_pT(input_batch, device=None, rand_number=0, std_scale=0.1):
    '''
    Applies the augmentation "gaussian resampling of pT" to events in a batch (torch tensor) and outputs a torch tensor with pT values rescaled within std.
    Resample pT of constituents with mu=pT, std=pT*std_scale in the DELPHES dataset w/ structure: MET, 4x electron, 4x muon, 10x jet.
    Each constituent has 3x features: transverse mom. pT, pseudorapidity eta, azimuthal angle phi.
    Args:
        input_batch: (batch_size, 57) flattened input
        device: cuda or cpu depending on input
        std_scale: scale multiplier for standard deviation std = pT * std_scale (default: 0.1)
    Returns:
        resampled_batch: (batch_size, 57) permutated output
    '''
    input_numpy = input_batch.cpu().detach().numpy().reshape(-1,19,3)
    #Guassian resample the pT's of each constituent with mu=pT and std = pT * std_scale
    np.random.seed(rand_number) #not sure if I should seed
    for x in input_numpy:
        x[:,0] = np.random.normal(loc=x[:,0], scale=np.absolute(x[:,0])*std_scale)

    #Return a torch tensor on the given device and correct shape (-1,57)
    resampled_batch = torch.from_numpy(input_numpy.reshape(-1,57)).to(dtype=torch.float32, device=device)

    return resampled_batch

def naive_masking(input_batch, device=None, rand_number=0, p=0.4, mask_full_particle=False):
    '''
    Applies the augmentation "naive_masking" to events in a batch (torch tensor) and outputs a torch tensor values randomly masked with probability p.
    Args:
        input_batch: (batch_size, 57) flattened input
        device: cuda or cpu depending on input
        p: probability of bernoulli trial (default: p=0.2)
        mask_full_particle: Whether to randomly mask full particles or randomly mask constistuents individually (default: False)
    Returns:
        resampled_batch: (batch_size, 57) permutated output
    '''
    np.random.seed(rand_number)
    input_numpy = input_batch.cpu().detach().numpy().reshape(-1)
    #Randomly (with prob. p) set parts of the input to 0.0 (mask/crop)
    if mask_full_particle:
        mask = np.random.choice([True, False], size=input_numpy.shape[0]/3, replace=True, p=[p, 1-p]).reshape(-1,19)
        input_numpy.reshape(-1,19,3)[mask] = 0.0
    else:
        mask = np.random.choice([True, False], size=input_numpy.shape[0], replace=True, p=[p, 1-p])
        input_numpy[mask] = 0.0
    resampled_batch = torch.from_numpy(input_numpy.reshape(-1,57)).to(dtype=torch.float32, device=device)
    return resampled_batch

def hardjet_masking(input_batch, device=None):
    '''
    Crops/masks the area of deltaR < 3.0 around the first hard jet of the event and outputs a torch tensor where the rest has been zero padded.
    The input has structure: MET, 4x electron, 4x muon, 10x jet with each constituent described by 3 features: transverse mom. pT, pseudorapidity eta, azimuthal angle phi.
        input_batch: (batch_size, 57) flattened input
        device: cuda or cpu depending on input
    Returns:
        resampled_batch: (batch_size, 57) masked output
    '''
    input_numpy = input_batch.cpu().detach().numpy().reshape(-1,19,3)
    #Calculate deltaR = (delta_phi**2 + delta_eta**2)**(1/2) from first jet
    deltaR = np.sqrt((input_numpy[:,:,1] - input_numpy[:,9,1])**2 + (input_numpy[:,:,2]-input_numpy[:,9,2])**2 + 1e-4)
    mask = np.where(deltaR <= 3.0, True, False)
    input_numpy[~mask] = 0.0
    
    resampled_batch = torch.from_numpy(input_numpy.reshape(-1,57)).to(dtype=torch.float32, device=device)
    return resampled_batch

def hardlepton_masking(input_batch, device=None):
    '''
    Crops/masks the area of deltaR < 3.0 around the first hard lepton of the event and outputs a torch tensor where the rest has been zero padded.
    The input has structure: MET, 4x electron, 4x muon, 10x jet with each constituent described by 3 features: transverse mom. pT, pseudorapidity eta, azimuthal angle phi.
        input_batch: (batch_size, 57) flattened input
        device: cuda or cpu depending on input
    Returns:
        resampled_batch: (batch_size, 57) masked output
    '''
    input_numpy = input_batch.cpu().detach().numpy().reshape(-1,19,3)
    #As there is a guaranteed lepton in each event, we have to find the highest pT one first (either first electron or muon)
    lepton_pT = np.concatenate((input_numpy[:, 0:4, 0], input_numpy[:, 4:8, 0]))  # Concatenate electron and muon pT values
    highest_pT_index = np.argmax(lepton_pT, axis=1)

    #Calculate deltaR = (delta_phi**2 + delta_eta**2)**(1/2) from first jet
    deltaR = np.sqrt((input_numpy[:,:,1] - input_numpy[:,highest_pT_index,1])**2 + (input_numpy[:,:,2]-input_numpy[:,highest_pT_index,2])**2 + 1e-4)
    mask = np.where(deltaR <= 3.0, True, False)
    input_numpy[~mask] = 0.0
    
    resampled_batch = torch.from_numpy(input_numpy.reshape(-1,57)).to(dtype=torch.float32, device=device)
    return resampled_batch

def detector_crop(input_batch, device=None, rand_number=0, crop_size = 5.0):
    '''
    Applies the augmentation "detector_crop" to events in a batch (torch tensor) and outputs a torch tensor.
    It randomly crops an area of deltaR <= crop_size of the detector and masks it by zero padding the rest.
    Args:
        input_batch: (batch_size, 57) flattened input
        device: cuda or cpu depending on input
        crop_size: region of the detector crop (default: deltaR <= 3.0)
    Returns:
        resampled_batch: (batch_size, 57) masked output
    '''
    np.random.seed(rand_number)
    input_numpy = input_batch.cpu().detach().numpy().reshape(-1,19,3)
    #Randomly crop a region of the detector with size deltaR
    #First find a random point in the angular space of the detector by uniform sampling on the unit sphere using the inverse transform method
    theta = np.arccos(1-2*np.random.rand(input_numpy.shape[0]))
    phi = 2*np.pi*np.random.rand(input_numpy.shape[0])
    eta = np.log(np.tan(theta/2+1e-4))
    #Then crop within size deltaR by setting the outside to 0.0
    deltaR = np.sqrt((input_numpy[:,:,1] - eta[:,None])**2 + (input_numpy[:,:,2]-phi[:,None])**2 + 1e-4)
    mask = deltaR <= crop_size
    input_numpy[~mask] = 0.0
    resampled_batch = torch.from_numpy(input_numpy.reshape(-1,57)).to(dtype=torch.float32, device=device)
    return resampled_batch