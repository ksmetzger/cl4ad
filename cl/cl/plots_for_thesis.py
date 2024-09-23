import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd

#Accuracy plot for different latent space dimensions
def acc_versus_latent_dim(accuracies, uncertainties_acc, dimensions, baseline_acc, title, savefig, fill_uncertainty):
    fig, ax = plt.subplots()
    ax.errorbar(dimensions, accuracies[0], yerr=uncertainties_acc[0], fmt='o',markersize=4, color='blue',capsize=3,elinewidth=2,linestyle='-',linewidth=0.75, label='Self-supervised SimCLR')
    ax.errorbar(dimensions, accuracies[1], yerr=uncertainties_acc[1], fmt='s',markersize=4, color='orange',capsize=3,elinewidth=2,linestyle='-',linewidth=0.75, label='Supervised SimCLR')
    if fill_uncertainty:
        ax.fill_between(dimensions, accuracies[0] - uncertainties_acc[0], accuracies[0]+uncertainties_acc[0], color='blue', alpha=0.1)
        ax.fill_between(dimensions, accuracies[1] - uncertainties_acc[1], accuracies[1]+uncertainties_acc[1], color='orange', alpha=0.1)
    ax.axhline(y=baseline_acc, color='red', linestyle=':', linewidth=1.5, label='Baseline')
    plt.title(title, size=18)
    ax.set_xscale('log')
    ax.set_xticks(dimensions)
    ax.set_xticklabels([f'{tick}' for tick in dimensions])
    ax.legend(prop={'size': 12})
    plt.grid()
    plt.xlabel("Latent space dimension",size=12)
    plt.ylabel("Accuracy [%]", size=12)
    bottom, top = plt.ylim()
    plt.ylim((bottom-1,top+1))
    plt.show()
    if savefig:
        fig.savefig("C:\\Users\\Kyle\\OneDrive\\Transfer Master project\\figures\\"+title+".png",dpi=300)

def bar_plot_diff_arch():
    pass

def acc_heatmap(accuracies, title, savefig):
    #Make df
    df = pd.DataFrame(data=np.float_(accuracies[1:,1:]), index=accuracies[1:,0], columns=accuracies[0,1:])
    print(df)
    fig, ax = plt.subplots()
    sns.set(font_scale=1.2)
    heatmap = sns.heatmap(data=df, annot=True, fmt='.1f',cmap="flare", ax=ax)
    #plt.title(title,size=18)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    fig.tight_layout()
    plt.show()
    if savefig:
        heatmap.figure.savefig("C:\\Users\\Kyle\\OneDrive\\Transfer Master project\\figures\\"+title+".png", dpi=300)


def main():
    dimensions=[6,8,16,32,64]
    ###ADC_Delphes
    #accuracies = np.array([[77.77,80.88,84.31,85.99,87.72],[92.00,91.99,92.00,92.08,92.12]])
    #uncertainties_acc = np.array([[0.45,0.40,0.20,0.29,0.16],[0.03,0.02,0.04,0.01,0.01]])
    #baseline_acc = 85.31
    ###JetClass
    #accuracies = np.array([[22.18,22.69,23.43,24.05,24.44],[35.68,35.84,37.77,38.08,38.27]])
    #uncertainties_acc = np.array([[0.14,0.14,0.09,0.08,0.03],[0.20,0.13,0.12,0.17,0.08]])
    #baseline_acc = 22.63
    #acc_versus_latent_dim(accuracies=accuracies,uncertainties_acc=uncertainties_acc,dimensions=dimensions,title="Linear embedding evaluation", baseline_acc=baseline_acc,savefig=True, fill_uncertainty=True)
    ###Trafo
    #accuracies = np.array([['','masking False', 'masking True'],['pos. encoding False',80.7,76.8],['pos. encoding True',76.6,75.2]])
    ###Augmentations
    accuracies = np.array([['','naive masking''\n'r'$\mathrm{p}=0.5$      ', 'gaussian resampling''\n'r'$\mathrm{s}=1.5$           ','rot. around beamline', r'gaussian resampling $p_T$''\n'r'$\mathrm{s}=2.0$              ','detector crop''\n'r'$\mathrm{R}<2.5$     '],[r'naive masking' '\n' r'$\mathrm{p}=0.5$      ',22.7,21.1,21.2,21.6,20.8],['gaussian resampling''\n'r'$\mathrm{s}=1.5$           ',21.5,21.7,21.7,20.9,21.8],['rot. around beamline',22.3,22.0,19.6,16.7,18.8],[r'gaussian resampling $p_T$''\n'r'$\mathrm{s}=2.0$              ',21.5,21.1,18.8,18.5,17.0],['detector crop''\n'r'$\mathrm{R}<2.5$     ',20.1,21.6,16.2,16.6,17.5]])
    acc_heatmap(accuracies, title="heatmap different augmentations", savefig=True)
    
if __name__=='__main__':
    main()