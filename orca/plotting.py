import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os

#t-SNE Plot of given embedding colored according to given labels
def tSNE(embedding, labels, title, filename, namedir, rand_number=0, orca=False):
    
    #Define object of tSNE
    tsne = TSNE(n_components=2, random_state=rand_number)
    #Transform the embedding (N,6)
    print('okay')
    embedding_trans = tsne.fit_transform(embedding)
    print('hard part')
    #Dictionary for the targets (colors)
    dict_labels_color = {0: 'teal', 1: 'lightseagreen', 2: 'springgreen', 3: 'darkgreen', 4: 'lightcoral', 5: 'maroon', 6: 'fuchsia', 7: 'indigo'
                             }
    dict_labels_names = {0: 'W-boson', 1: 'QCD Multijet', 2: 'Z-boson', 3: 'ttbar', 4: 'leptoquark', 5: 'ato4l', 6: 'hChToTauNu', 7: 'hToTauTau'
                             }
    #Dictionary for randomly generated anomaly labels by orca
    if orca:
        dict_orca = {0: 0, 1: 1, 2: 2, 3: 3, 4: 7, 5: 5, 6: 6, 7: 4
                }
    else:
        dict_orca = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7
                }
    #Create path for the file
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, namedir)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    #Plot for each label the colored 2D tSNE dimensionality reduction
    fig, ax = plt.subplots()
    #Create array for reordering the labels
    containedlabels = []
    for label in np.unique(labels):
        print(label)
        containedlabels.append(int(label))
        idx = np.where(label == labels)[0]
        print(np.shape(embedding_trans[idx,0]))
        ax.scatter(embedding_trans[idx,0], embedding_trans[idx,1], c = dict_labels_color[dict_orca[label]], label = str(int(dict_orca[label]))+ ': ' + dict_labels_names[dict_orca[label]], s=1, zorder=(dict_orca[label]+1))
    #reordering the labels 
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    print(containedlabels)
    order = [dict_orca[i] for i in containedlabels]
    order = np.array(order)
    order = np.argsort(np.argsort(order))
    print(order)
    ax.legend([handles[i] for i in order], [labels_legend[i] for i in order],loc='lower right', markerscale=3).set_zorder(10)
    
    plt.title(title)
    plt.savefig(results_dir + filename ,dpi=300)
    plt.show()
    
#Histogram of the softmax-probability distribution of predicted labels for a given input signal class
def softmax_distribution_histogram(embedding, true_labels, filename, namedir, rand_number=0):
    
    #For each signal class (0-7) add up the softmax probabilities and normalize
    addedup_prob = np.empty((8,8))
    for true_label in np.unique(true_labels):
        idx = np.where(true_label == true_labels)[0]
        print(np.shape(embedding[idx,:]))
        addedup_prob[int(true_label),...] = (np.sum(embedding[idx,...], axis=0))/len(idx)
    
    #Dictionary for the targets
    dict_labels_color = {0: 'teal', 1: 'lightseagreen', 2: 'springgreen', 3: 'darkgreen', 4: 'lightcoral', 5: 'maroon', 6: 'fuchsia', 7: 'indigo'
        }
    dict_labels_names = {0: 'W-boson', 1: 'QCD Multijet', 2: 'Z-boson', 3: 'ttbar', 4: 'leptoquark', 5: 'ato4l', 6: 'hChToTauNu', 7: 'hToTauTau'
        }
    dict_orca = {0: 0, 1: 1, 2: 2, 3: 3, 4: 7, 5: 5, 6: 6, 7: 4
            }
    #Exchange class labeling according to ORCA dictionary
    before = np.arange(8)
    after = [dict_orca[i] for i in before]
    addedup_prob[:,before] = addedup_prob[:,after]
    #Create path for the file
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, namedir)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    #Plot the histogram for each signal class (eight total)
    for signal_class in np.unique(true_labels):
        plt.figure()
        plt.bar(np.arange(8), addedup_prob[int(signal_class),...], color = dict_labels_color[int(signal_class)] )
        plt.title(str(int(signal_class))+ ': ' + dict_labels_names[signal_class])
        plt.xlabel('Assigned class label')
        plt.ylabel('Probability (from softmax)')
        plt.savefig(results_dir + filename + '_signal_' + str(int(signal_class)), dpi=300)
        plt.show()






    
### Import embedding and targets (use unbiased_latent.npz from contrastive learning pretraining)
#Import the background + signals + labels
drive_path = ''
drive_path_pytorch = ''
dataset = np.load(drive_path+'unbiased_latent.npz')

background_data = dataset['x_train']
background_targets = dataset['labels_train']
background_targets = background_targets.reshape(-1)

print('Mean: ',np.mean(background_data),'Std: ', np.std(background_data))

leptoquark = dataset['leptoquark']
ato4l = dataset['ato4l']
hChToTauNu = dataset['hChToTauNu']
hToTauTau = dataset['hToTauTau']

###Random sample 1/4 (1/8) of the background for labeled data, 1/4 (1/8) of background + signals (1/8) for unlabeled data
#Random sample the background
rand_number = 0
np.random.seed(rand_number)
size_fraction = 1/4
labeled_background, unlabeled_background, labeled_targets, unlabeled_targets = train_test_split(background_data, background_targets, test_size=size_fraction, train_size =size_fraction ,random_state=rand_number)

#Also random sample the signals
unlabeled_leptoquark, _ = train_test_split(leptoquark, train_size=size_fraction, random_state=rand_number)
unlabeled_ato4l, _ = train_test_split(ato4l, train_size=size_fraction, random_state=rand_number)
unlabeled_hChToTauNu, _ = train_test_split(hChToTauNu, train_size=size_fraction, random_state=rand_number)
unlabeled_hToTauTau, _ = train_test_split(hToTauTau, train_size=size_fraction, random_state=rand_number)

#Shuffle in signals (and their labels for testing) with the unlabeled background
unlabeled_data = np.concatenate((unlabeled_background, unlabeled_leptoquark, unlabeled_ato4l, unlabeled_hChToTauNu, unlabeled_hToTauTau), axis = 0)
unlabeled_targets = np.concatenate((unlabeled_targets, np.ones(len(unlabeled_leptoquark),dtype=int)*4, np.ones(len(unlabeled_ato4l),dtype=int)*5,np.ones(len(unlabeled_hChToTauNu),dtype=int)*6,np.ones(len(unlabeled_hToTauTau),dtype=int)*7),axis=0)
unlabeled_data_shuffled, unlabeled_targets_shuffled = shuffle(unlabeled_data, unlabeled_targets, random_state=rand_number)


### Import targets, predictions and confidences from the training run
#Import targets + preds + confs
drive_path = 'C:\\Users\\Kyle\\OneDrive\\Transfer Semester project\\orca_newmodel\\orca\\latent\\'

dataset2 = np.load(drive_path+'target_pred_conf_kyle20240222-084122.npz')
target_pred_conf = dataset2['target_pred_conf']

#Last epoch predictions
last = target_pred_conf[-1,...]
pred_targets = last[1,:]

#Import softmax distribution
prob_softmax = dataset2['prob_softmax']

#Last epoch softmax distributions
last_softmax = prob_softmax[-1,...]

print('done loading samples')


#Indexes to plot (10% of the dataset for the t-SNE plots)
p = 0.1
idx = np.random.choice(a=[True, False], size = len(pred_targets), p=[p, 1-p])

#Run tSNE
tSNE(unlabeled_data_shuffled[idx,...], unlabeled_targets_shuffled[idx], '2D tSNE colored by true labels', 'tSNE_2D_true_labels_10.pdf', 'plots/orca_plots_v2_report/', rand_number, orca=False)
tSNE(unlabeled_data_shuffled[idx,...], pred_targets[idx], '2D tSNE colored by orca labels', 'tSNE_2D_orca_pred_labels_10.pdf', 'plots/orca_plots_v2_report/', rand_number, orca=True)

#Run softmax histograms
softmax_distribution_histogram(last_softmax, unlabeled_targets_shuffled, 'histogram of softmax_distribution test', 'plots/orca_plots_v2_report/')

#Run tSNE (for a given orca class)
for i in range(8):
    idx_class_orca = pred_targets[idx]==i
    title = '2D tSNE of orca class ' +str(i)+ ' colored by true labels'
    filename = 'tSNE_2d_true_labels_orca_class_10_'+str(i)
    print(filename)
    tSNE(unlabeled_data_shuffled[idx][idx_class_orca], unlabeled_targets_shuffled[idx][idx_class_orca], title, filename , 'plots/orca_plots_v2_report/', rand_number, orca=False)
    
#Run tSNE (for a given true class)
for i in range(8):
    idx_class_true = unlabeled_targets_shuffled[idx]==i
    title = '2D tSNE of true class ' +str(i)+ ' colored by orca labels'
    filename = 'tSNE_2d_orca_labels_true_class_10_'+str(i)
    print(filename)
    tSNE(unlabeled_data_shuffled[idx][idx_class_true], pred_targets[idx][idx_class_true], title, filename , 'plots/orca_plots_v2_report/', rand_number, orca=True)
    
    
    