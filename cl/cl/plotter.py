import os 
#import corner 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA 
from sklearn.metrics import roc_curve, auc
import torch
from models import SimpleDense, Identity
from torch.utils.data import DataLoader, Dataset
from train_with_signal import TorchCLDataset
import torch.nn as nn
import torch.nn.functional as F

#Define color and name dicts
#Dictionary for the targets (colors)
dict_labels_color = {0: 'teal', 
                     1: 'lightseagreen', 
                     2: 'springgreen', 
                     3: 'darkgreen', 
                     4: 'lightcoral', 
                     5: 'maroon', 
                     6: 'fuchsia', 
                     7: 'indigo'
                             }
dict_labels_names = {0: 'W-boson', 
                     1: 'QCD Multijet', 
                     2: 'Z-boson', 
                     3: 'ttbar', 
                     4: 'leptoquark', 
                     5: 'ato4l', 
                     6: 'hChToTauNu', 
                     7: 'hToTauTau'
                             }

#t-SNE Plot of given embedding colored according to given labels
def tSNE(embedding, labels, title, filename, namedir, dict_labels_color, dict_labels_names, rand_number=0, orca=False):
    
    #Define object of tSNE
    tsne = TSNE(n_components=2, random_state=rand_number)
    #Transform the embedding (N,6)
    print('Fitting the t-SNE')
    embedding_trans = tsne.fit_transform(embedding)

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
        #print(label)
        containedlabels.append(int(label))
        idx = np.where(label == labels)[0]
        #print(np.shape(embedding_trans[idx,0]))
        ax.scatter(embedding_trans[idx,0], embedding_trans[idx,1], c = dict_labels_color[dict_orca[label]], label = str(int(dict_orca[label]))+ ': ' + dict_labels_names[dict_orca[label]], s=1, zorder=(dict_orca[label]+1))
    #reordering the labels 
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    #print(containedlabels)
    order = [dict_orca[i] for i in containedlabels]
    order = np.array(order)
    order = np.argsort(np.argsort(order))
    #print(order)
    ax.legend([handles[i] for i in order], [labels_legend[i] for i in order],loc='lower right', markerscale=3).set_zorder(10)
    
    plt.title(title)
    plt.savefig(results_dir + filename ,dpi=300)
    plt.show()

#From graphing_module.py
def plot_ROC(predictions, labels, filename, title, folder='plots'): 
    '''
    Plots ROC Curves for the linear evaluation on-top the self-super contrastive embedding
    predicted_backgrounds: target_score for the background class (class 0)
    predicted_anomalies: target_score for the anomalous signals (class 1)
    '''
    print("Plotting ROC Plots!")   
    # Defines true class. Assumes background true class is 0, anomaly true class is 1 (pos. class)
    true = np.empty_like(labels)
    true[(labels<4)] = 0.
    true[(labels>=4)] = 1.
    predictions = predictions[:,1]
    # Calculates ROC properties and plots on same plt
    false_pos_rate, true_pos_rate, threshold = roc_curve(true, predictions)
    area_under_curve = auc(false_pos_rate, true_pos_rate)
    plt.plot(false_pos_rate, true_pos_rate, label=f'AUC: {area_under_curve*100:.2f}%', 
             linewidth=2, color='teal') 
    
    # Defines plot properties to highlight plot properties relevant to FPGA constraints
    # plt.xlim(10**(-6),1) 
    # plt.ylim(10**(-6),1)
    # plt.xscale('log') 
    # plt.yscale('log') 
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    
    plt.title(title)
    
    # Creates x=y line to compare model against random classification performance
    plt.plot(np.linspace(0, 1),np.linspace(0, 1), ':', color='0.75', linewidth=2)
    plt.legend(loc='lower right', frameon=False)
    
    # Saves plot and reports success 
    subfolder = os.path.dirname(__file__)
    subfolder = os.path.join(subfolder, folder)
    os.makedirs(subfolder, exist_ok=True)
    file_path = os.path.join(subfolder, filename)
    plt.savefig(file_path) 
    plt.close('all')
    print(f"ROC Plot saved at '{filename}'")

def plot_PCA(representations, labels, title, filename, dict_labels_color, dict_labels_names, folder='plots', rand_number=0, dimension=2, orca=False):
    '''
    Plots 2/3D PCA of representations and saves file at filename location depending on keyword dimension.
    '''
    # Perform PCA to extract the 2/3 prin. components
    print(f"Plotting {dimension}D PCA!")
    pca = PCA(n_components=dimension, random_state=rand_number)
    components = pca.fit_transform(representations)

    #Dictionary for randomly generated anomaly labels by orca
    if orca:
        dict_orca = {0: 0, 1: 1, 2: 2, 3: 3, 4: 7, 5: 5, 6: 6, 7: 4
                }
    else:
        dict_orca = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7
                }
    
    # Creates three dimensional plots 
    fig = plt.figure()
    if dimension==2:
        ax = fig.add_subplot(111)
    elif dimension==3:
        ax  = fig.add_subplot(111, projection='3d')

    #Create array for reordering the labels
    containedlabels = []
    for label in np.unique(labels):
        #print(label)
        containedlabels.append(int(label))
        idx = np.where(label == labels)[0]
        #print(np.shape(embedding_trans[idx,0]))
        if dimension==2:
            ax.scatter(components[idx,0], components[idx,1], c = dict_labels_color[dict_orca[label]], label = str(int(dict_orca[label]))+ ': ' + dict_labels_names[dict_orca[label]], s=1, zorder=(dict_orca[label]+1))
        elif dimension==3:
            ax.scatter(components[idx,0], components[idx,1], components[idx,2], c = dict_labels_color[dict_orca[label]], label = str(int(dict_orca[label]))+ ': ' + dict_labels_names[dict_orca[label]], s=1, zorder=(dict_orca[label]+1))
    #reordering the labels 
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    order = [dict_orca[i] for i in containedlabels]
    order = np.array(order)
    order = np.argsort(np.argsort(order))
    ax.legend([handles[i] for i in order], [labels_legend[i] for i in order],loc='lower right', markerscale=3).set_zorder(10)
    
    # Labels axis and creates title 
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    if dimension==3:
        ax.set_zlabel('Principal Component 3')
    
    ax.set_title(title)
    
    # Saves plot and reports success 
    subfolder = os.path.dirname(__file__)
    subfolder = os.path.join(subfolder, folder)
    os.makedirs(subfolder, exist_ok=True)
    file_path = os.path.join(subfolder, filename)
    plt.savefig(file_path) 
    plt.close('all')
    print(f"PCA Plot saved at '{filename}'")
     

#Define inference if there is no embedding.npz already saved, in order to use for plots of the embedding
def inference(model_name, input_data, input_labels, device=None):
    '''
    Inference for test input with dimensionality (-1, 57) using model SimpleDense().
    '''
    if device == None:
        device = torch.device('cpu')
    else: 
        device = device
    #Import model for embedding
    model = SimpleDense().to(device)
    model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))
    model.eval()
    #Get output with dataloader
    data_loader = DataLoader(
        TorchCLDataset(input_data, input_labels, device),
        batch_size=1024,
        shuffle=False)
    with torch.no_grad():
        output = np.concatenate([model.representation(data).cpu().detach().numpy() for (data, label) in data_loader], axis=0)

    return output

#Define classification score
def classification_score(backbone_name, head_name, input_data, input_labels, device=None, embed_dim=48, num_classes=8, mode='roc'):
    '''
    Get the classification scores (softmax) needed for roc_curvers, ...
    If  mode = 'roc' output only (softmax) probabilities for  background (class 0) vs anomaly (class 1)
        mode = 'all' output (softmax) probabilities for all the classes (4 background + 4 diff. anomalies)
    '''
    if device == None:
        device = torch.device('cpu')
    else: 
        device = device
    #Import model for embedding
    if backbone_name == 'NoEmbedding':
        backbone = Identity()
        embed_dim = 57
    else:
        backbone = SimpleDense().to(device)
        embed_dim = 48
        backbone.load_state_dict(torch.load(backbone_name, map_location=torch.device(device)))

    head = nn.Linear(embed_dim, num_classes).to(device)
    head.load_state_dict(torch.load(head_name, map_location=torch.device(device)))
    backbone.eval()
    head.eval()

    #Get logits trained from linear_eavaluation.py
    data_loader = DataLoader(
        TorchCLDataset(input_data, input_labels, device),
        batch_size=1024,
        shuffle=False)
    with torch.no_grad():
        softmax = np.concatenate([F.softmax(head(backbone.representation(data)), dim=1).cpu().detach().numpy() for (data, label) in data_loader], axis=0)
    #Get the softmax predictions
    if mode == 'roc':
        predictions = np.array([np.sum(softmax[:,:4], axis=1), np.sum(softmax[:,4:], axis=1)]).T
    elif mode == 'all':
        predictions = softmax
    return predictions

#Plot embedding (always the _test) with different methods
def main():
    rand_number = 0
    np.random.seed(rand_number)

    #Load embedding (test (train/val used for model optim))
    drive_path = 'C:\\Users\\Kyle\\OneDrive\\Transfer Master project\\orca_fork\\cl4ad\\cl\\cl\\'
    data = np.load(drive_path+'dataset_background_signal.npz')
    #embedding = np.load(drive_path+'output/runs35/embedding.npz')
    #embedded_test = embedding['embedding_test']
    labels_test = data['labels_test']
    data_test = data['x_test'].reshape(-1,57)
    embedded_test = inference('output/runs36/vae.pth', data_test, labels_test)

    #Plot t-SNE
    def plot_tsne():
        p = 0.01
        idx = np.random.choice(a=[True, False], size = len(labels_test), p=[p, 1-p]) #Indexes to plot (1% of the dataset for the t-SNE plots)
        print("===Plotting t-SNE===")
        print(f"with {np.sum(idx)} datapoints")
        tSNE(embedded_test[idx], labels_test[idx], '2D-t_SNE of 48D embedding with VICReg', '2D-t_SNE of 48D embedding with SimCLR.pdf', drive_path+'output/runs36/plots/',
            dict_labels_color, dict_labels_names, rand_number, orca=False)
        tSNE(data_test[idx], labels_test[idx], '2D-t_SNE of 57D test data', '2D-t_SNE of 57D test data.pdf', drive_path+'output/runs36/plots/',
            dict_labels_color, dict_labels_names, rand_number, orca=False)
    
    #Plot ROC curve (with AUC)
    def plot_roc():
        print("===Plotting ROC curve===")
        predictions = classification_score('output/runs36/vae.pth', 'output/runs36/head.pth', data_test, labels_test, mode='roc')
        predictions_noembedding = classification_score('NoEmbedding', 'output/NoEmbedding/head.pth', data_test, labels_test, mode='roc')
        labels = labels_test
        plot_ROC(predictions, labels, title='ROC curve with AUC for SimCLR embedding with supervised linear evaluation', filename='ROC curve with AUC for SimCLR embedding with supervised linear evaluation.pdf', folder='output/runs36/plots/')
        plot_ROC(predictions_noembedding, labels, title='ROC curve with AUC for No Embedding with supervised linear evaluation', filename='ROC curve with AUC for No Embedding with supervised linear evaluation.pdf', folder='output/runs36/plots/')
    
    def plot_pca(dimension=2):
        p = 0.01
        idx = np.random.choice(a=[True, False], size = len(labels_test), p=[p, 1-p]) #Indexes to plot (1% of the dataset for the PCA plots)
        print("===Plotting PCA===")
        print(f"with {np.sum(idx)} datapoints")
        labels = labels_test
        plot_PCA(embedded_test[idx], labels[idx], f'{dimension}D PCA of 48D embedding with SimCLR', f'{dimension}D PCA of 48D embedding with SimCLR.pdf',
                 dict_labels_color, dict_labels_names,'output/runs36/plots/' ,rand_number, dimension, orca=False)
        plot_PCA(data_test[idx], labels[idx], f'{dimension}D PCA of 57D test data', f'{dimension}D PCA of 57D test data.pdf',
                 dict_labels_color, dict_labels_names,'output/runs36/plots/' ,rand_number, dimension, orca=False)

    #plot_tsne()
    #plot_roc()
    plot_pca(dimension=2)

if __name__ == '__main__':
    main()