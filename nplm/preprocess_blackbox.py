import numpy as np

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

def main():
    for i in range(10):
        blackbox_split = np.load(f"blackbox/BBsplits57D/split{i}.npy")
        blackbox_split_preprocessed = zscore_preprocess(blackbox_split.reshape(-1,19,3,1), train=False, scaling_file="scaling_file.npz")
        np.save(f"blackbox/BBsplits57D/split{i}_preprocessed.npy", blackbox_split_preprocessed)


if __name__=='__main__':
    main()