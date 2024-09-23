import numpy as np

def mean_with_uncertainty(input):
    """Calculates the mean and standard deviation for an array of accuracies 
    using the corrected sample standard deviation"""
    mean = np.mean(input)
    N = len(input)
    #Sample standard deviation
    var = np.sum((input-mean)**2)/(len(input)-1)
    std = np.sqrt(var)
    mean_std = std/np.sqrt(N)
    print(f"Resulting mean and mean_std: {mean:.2f}+/-{mean_std:.2f}")
    return mean, mean_std

def save_acc_as_npz(accuracies):
    """Save the accuracies into an npz file"""
    info ={
        "array dim0": "run ID's",
        "array dim1": "latent space dimension",
        "array dim2": "mean accuracy over 5 k-folded runs",
        "array dim3": "mean standard deviation over 5 k-folded runs"
    }
    np.savez("Self-supervised SimCLR with MLP", accuracies = np.transpose(accuracies), info=info)

def main():
    input = [15.34,13.52,7.59]
    mean_with_uncertainty(input)
    #Save the self-supervised SimCLR accuracies
    #accuracies = [[249,245,250,247,265],[6,8,16,32,64],[77.77,80.88,84.31,85.99,87.72],[0.45,0.40,0.20,0.29,0.16]]
    #print(accuracies)
    #save_acc_as_npz(accuracies)
if __name__=='__main__':
    """ x = np.load("Self-supervised SimCLR with MLP.npz", allow_pickle=True)
    info = x["info"]
    acc = x["accuracies"]
    print(info)
    print(acc) """
    main()