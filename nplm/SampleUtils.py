import numpy as np

def standardize(dataset, mean_all, std_all):
    dataset_new = np.copy(dataset)
    for j in range(dataset.shape[1]):
        mean, std = mean_all[j], std_all[j]
        dataset_new[:, j] = (dataset[:, j]- mean)*1./ (std+1e-6)
    return dataset_new

def inv_standardize(dataset, mean_all, std_all):
    dataset_new = np.copy(dataset)
    for j in range(dataset.shape[1]):
        mean, std = mean_all[j], std_all[j]
        vec  = dataset[:, j]
        dataset_new[:, j] = dataset[:, j] * std + mean
    return dataset_new
