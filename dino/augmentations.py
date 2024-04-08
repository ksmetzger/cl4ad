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