import os 
#import corner 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as mlines
from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA 
from sklearn.metrics import roc_curve, auc
import torch
from models import SimpleDense, Identity, SimpleDense_small, SimpleDense_JetClass, DeepSets, CVAE, CVAE_JetClass, SimpleDense_ADC
from transformer import TransformerEncoder
from torch.utils.data import DataLoader, Dataset
from train_with_signal import TorchCLDataset
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
from PIL import Image
import re
import h5py
from sklearn.model_selection import train_test_split

#Define color and name dicts
#Dictionary for the targets (colors)
#Normal background (4 classes) + signal (4 classes)
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
""" #Split datasets (just copy/paste from split.py) or if legend covers everything just plot index and color (legend seperately)
dict_labels_color = {0: 'teal', 
                     1: 'lightseagreen', 
                     2: 'springgreen', 
                     3: 'darkgreen', 
                     4: 'goldenrod', 
                     5: 'darkkhaki', 
                     6: 'olive', 
                     7: 'honeydew',
                     8: 'chocolate',
                     9: 'orange',
                     10: 'moccasin',
                     11: 'lightsalmon',
                     12: 'brown',
                     13: 'rosybrown',
                     14: 'slategrey',
                     15: 'silver',
                             }
dict_labels_names = {
                    0: '',
                    1: '',
                    2: '',
                    3: '',
                    4: '',
                    5: '',
                    6: '',
                    7: '',
                    8: '',
                    9: '',
                    10: '',
                    11: '',
                    12: '',
                    13: '',
                    14: '',
                    15: '',
                            } """
#Dicts for JetClass
dict_labels_color = {0: 'grey', 
                     1: 'teal', 
                     2: 'springgreen', 
                     3: 'darkgreen', 
                     4: 'lightcoral', 
                     5: 'maroon', 
                     6: 'fuchsia', 
                     7: 'indigo',
                     8: 'orange',
                     9: 'brown'
                             }
dict_labels_names = {0: 'QCD-background', 
                     1: 'hToBB', 
                     2: 'hToCC', 
                     3: 'hToGG', 
                     4: 'hTo4q', 
                     5: 'hTolvqq', 
                     6: 'tTobqq', 
                     7: 'tToblv',
                     8: 'WToqq',
                     9: 'ZToqq',
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
        dict_orca = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15,
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
        if label == 0:
            label_name = str(int(dict_orca[label]))+ ': ' + dict_labels_names[dict_orca[label]]
        elif label == 2:
            label_name = '1: Signal cluster'
        elif label == 3:
            label_name = '2: Signal cluster'
        ax.scatter(embedding_trans[idx,0], embedding_trans[idx,1], c = dict_labels_color[dict_orca[label]], label = label_name, s=1, zorder=(dict_orca[label]+1))
        #label = str(int(dict_orca[label]))+ ': ' + dict_labels_names[dict_orca[label]]
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
    plt.show(block=False)
    plt.close("all")

#From graphing_module.py
def plot_ROC(predictions, labels, filename, title, folder='plots'): 
    '''
    Plots ROC Curves for the linear evaluation on-top the self-super contrastive embedding for all signal vs background
    predicted_backgrounds: target_score for the background class (class 0)
    predicted_anomalies: target_score for the anomalous signals (class 1)
    '''
    print("Plotting ROC Plots!")   
    # Defines true class. Assumes background true class is 0, anomaly true class is 1 (pos. class)
    true = np.empty_like(labels)
    true[(labels<1)] = 0.
    true[(labels>=1)] = 1.
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
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid()
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

def plot_ROC_multiclass(predictions, labels, filename, title, folder='plots', num_classes = 4):
    '''
    Plots ROC Curves for the linear evaluation on-top the self-super contrastive embedding one vs all
    predictions: array with target scores for each class
    '''
    print("Plotting ROC Plots!")   
    plt.figure()
    # Defines true class. Looping over all the classes, the target class has label 1, the rest label 0.
    for idx in range(num_classes):
        true = np.empty_like(labels)
        true[(labels == idx)] = 1
        true[~(labels == idx)] = 0
        prediction = predictions[:,idx]
        # Calculates ROC properties and plots on same plt
        false_pos_rate, true_pos_rate, threshold = roc_curve(true, prediction)
        area_under_curve = auc(false_pos_rate, true_pos_rate)
        plt.plot(false_pos_rate, true_pos_rate, label=f'{dict_labels_names[idx]} AUC: {area_under_curve*100:.2f}%', 
                linewidth=2, color=dict_labels_color[idx], ds='steps')
    plt.title(title, size=13)
    plt.xlabel("FPR", size=12)
    plt.ylabel("TPR", size=12)
    plt.grid()
    # Creates x=y line to compare model against random classification performance
    plt.plot(np.linspace(0, 1),np.linspace(0, 1), ':', color='0.75', linewidth=2)
    plt.legend(loc='lower right', frameon=False,prop={'size': 10})
    
    # Saves plot and reports success 
    subfolder = os.path.dirname(__file__)
    subfolder = os.path.join(subfolder, folder)
    os.makedirs(subfolder, exist_ok=True)
    file_path = os.path.join(subfolder, filename)
    plt.savefig(file_path) 
    plt.close('all')
    print(f"ROC Plot multiclass saved at '{filename}'")

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
        dict_orca = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15,
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

def corner_plot(embedding, labels, title, filename, dict_labels_color, dict_labels_names, pca=True, normalize=True, background='one', anomalies=['leptoquark', 'ato4l', 'hChToTauNu', 'hToTauTau'], folder='plots', rand_number=0):
    '''
    Plot a corner plot visualizing the embedding
    pca: Plot 3D-PCA reduced input instead of latent space
    normalize: Upsample anomalies with replacement in order to normalize the histograms/kde plots
    background: 'detail' plots all the different SM background classes with different colors, 'one' combines all background classes as one (default: 'one')
    anomalies: array with all of the anomaly names for plotting (default: ['leptoquark', 'ato4l', 'hChToTauNu', 'hToTauTau']) 
    '''
    print("Plotting Corner Plots!")
    print(f"Mean: {np.mean(embedding)} and std: {np.std(embedding)}")
    if pca==True:
        pca = PCA(n_components=3, random_state=rand_number) 
        embedding = pca.fit_transform(embedding)
    print(f"Input shape of the embedding: {np.shape(embedding)}")
    #Create dict needed for seaborn input using the 3D-embedding and type
    def sns_dict(embedding, label_name):
        sns_dict = {}
        sns_dict["label"] = [label_name] * len(embedding)
        for i in range(np.shape(embedding)[1]): #Iterate through dimensions of the embedding
            sns_dict["Dimension {}".format(i+1)] = embedding[:,i]
        return sns_dict
    background_df = pd.DataFrame()
    anomaly_df = pd.DataFrame()
    #Background part
    if background=='one':
        #mask = (labels < 4).reshape(-1)
        mask = (labels < 1).reshape(-1)
        background_embed = embedding[mask]
        #background_df = pd.concat([background_df, pd.DataFrame(sns_dict(background_embed, "SM-background"))])
        background_df = pd.concat([background_df, pd.DataFrame(sns_dict(background_embed, "QCD-background"))])
    elif background == 'detail':
        for i in range(4): #Iterate through background classes
            mask = (labels == i).reshape(-1)
            background_embed = embedding[mask]
            background_df = pd.concat([background_df, pd.DataFrame(sns_dict(background_embed, dict_labels_names[i]))])
    else:
        assert False
    #Anomaly part
    for anomaly in anomalies:
        mask = (labels == list(dict_labels_names.keys())[list(dict_labels_names.values()).index(anomaly)]).reshape(-1) #Get label for the corr. anomaly
        anomaly_embed = embedding[mask]
        """ sns_dict["label"] = [anomaly] * np.sum(mask)
        for i in range(np.shape(embedding)[1]): #Iterate through dimensions of the embedding
                sns_dict["Dimension {}".format(i+1)] = anomaly_embed[:,i] """
        anomaly_df = pd.concat([anomaly_df, pd.DataFrame(sns_dict(anomaly_embed, anomaly))])
    
    #Scale the anomaly up with replacement if normalize = True
    if normalize:
        frac = float(background_df.shape[0]/anomaly_df.shape[0])
        anomaly_df = anomaly_df.sample(frac=frac, replace=True ,random_state=rand_number)

    #Append background and anomaly DataFrame
    df = pd.concat([background_df, anomaly_df])

    #Get color palette
    color_dict = {}
    for i in dict_labels_color.keys():
        color_dict[dict_labels_names[i]] = dict_labels_color[i]
    #color_dict["SM-background"] = 'grey'
    color_dict["QCD-background"] = 'grey'
    plot_kws ={
        's': 2,
    }

    #Create cornerplot (pairplot)
    #corner = sns.pairplot(df, hue="label", kind='kde', palette=color_dict, corner=True)
    corner = sns.pairplot(df, hue="label", palette=color_dict, corner=True, plot_kws=plot_kws)
    corner.add_legend(markerscale=5)
    #corner.figure.suptitle(title)
    plt.show()

    # Saves plot and reports success 
    subfolder = os.path.dirname(__file__)
    subfolder = os.path.join(subfolder, folder)
    os.makedirs(subfolder, exist_ok=True)
    file_path = os.path.join(subfolder, filename)
    corner.savefig(file_path) 
    plt.close('all')
    print(f"Corner Plot saved at '{filename}'")


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
    #model = SimpleDense(latent_dim=12).to(device)
    #model = SimpleDense_small().to(device)
    #model = DeepSets(latent_dim=6).to(device)
    #model = CVAE(latent_dim=6).to(device)
    model = SimpleDense_JetClass(latent_dim=6).to(device)
    #model = CVAE_JetClass(latent_dim=6).to(device)
    #model = SimpleDense_ADC(latent_dim=6).to(device)
    transformer_args_jetclass = dict(
        input_dim=4, 
        model_dim=128, 
        output_dim=64,
        embed_dim=6,   #Only change embed_dim without describing new transformer architecture
        n_heads=8, 
        dim_feedforward=256, 
        n_layers=4,
        hidden_dim_dino_head=256,
        bottleneck_dim_dino_head=64,
        pos_encoding = True,
        use_mask = True,
        )
    #model = TransformerEncoder(**transformer_args_jetclass).to(device)
    model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))
    model.eval()
    #Get output with dataloader
    data_loader = DataLoader(
        TorchCLDataset(input_data, input_labels, device),
        batch_size=1024,
        shuffle=False)
    with torch.no_grad():
        output = np.concatenate([model.representation(data.to(device)).cpu().detach().numpy() for (data, label) in data_loader], axis=0)

    return output

#Define classification score
def classification_score(backbone_name, head_name, input_data, input_labels, device=None, embed_dim=48, num_classes=10, mode='roc'):
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
        #embed_dim = 57
        embed_dim=512
    else:
        #backbone = SimpleDense(latent_dim=12).to(device)
        #embed_dim = 12
        backbone = SimpleDense_JetClass(latent_dim=6).to(device)
        embed_dim = 6
        #backbone = CVAE_JetClass(latent_dim=6).to(device)
        #backbone = SimpleDense_ADC(latent_dim=6).to(device)
        transformer_args_jetclass = dict(
        input_dim=4, 
        model_dim=128, 
        output_dim=64,
        embed_dim=6,   #Only change embed_dim without describing new transformer architecture
        n_heads=8, 
        dim_feedforward=256, 
        n_layers=4,
        hidden_dim_dino_head=256,
        bottleneck_dim_dino_head=64,
        pos_encoding = True,
        use_mask = True,
        )
        #backbone = TransformerEncoder(**transformer_args_jetclass).to(device)
        #backbone = SimpleDense_small().to(device)
        #embed_dim = 6
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

def animate(images, type, runs, method):
    fig = plt.figure(frameon=False)
    size = 5
    fig.set_size_inches(size, size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    #Set first image
    im = ax.imshow(images[0], animated=True, aspect='auto')
    #Update
    def update(i):
        im.set_array(images[i])
        return im,
    #Create the animation object
    animation_fig = animation.FuncAnimation(fig, update, frames = len(images), interval=1000, blit=True, repeat=False)
    plt.show(block=False)
    plt.close()
    animation_fig.save(f"output/{runs}/plots/{method}_{type}.gif", dpi=500)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

#Plot embedding (always the _test) with different methods
def main(runs):
    rand_number = 0
    np.random.seed(rand_number)

    #Load embedding (test (train/val used for model optim))
    drive_path = 'C:\\Users\\Kyle\\OneDrive\\Transfer Master project\\orca_fork\\cl4ad\\cl\\cl\\'
    #data = np.load(drive_path+'dataset_background_signal.npz')
    data = np.load(drive_path+'jetclass_dataset/JetClass_background_signal_reshaped.npz')
    #data = np.load(drive_path+'background_dataset_divided.npz')
    #data = np.load(drive_path+'jetclass_dataset/JetClass_background_higgs_signal_testset.npz')
    """ with h5py.File(drive_path+f'JetClass_kfolded.hdf5', 'r') as f:
        data_test = np.array(f['x_test'][...]).reshape(-1,512)
        labels_test = np.array(f['labels_test'][...]) """
    #Signal
    """ with h5py.File(drive_path+'ADC_Delphes_signals.hdf5', 'r') as f:
        leptoquark = np.array(f['leptoquark'][...])
        leptoquark_labels = np.array(f['leptoquark_labels'][...])
        ato4l = np.array(f['ato4l'][...])
        ato4l_labels = np.array(f['ato4l_labels'][...])
        hChToTauNu = np.array(f['hChToTauNu'][...])
        hChToTauNu_labels = np.array(f['hChToTauNu_labels'][...])
        hToTauTau = np.array(f['hToTauTau'][...])
        hToTauTau_labels = np.array(f['hToTauTau_labels'][...]) """
    #embedding = np.load(drive_path+'output/runs35/embedding.npz')
    #embedded_test = embedding['embedding_test']
    labels_test = data['labels_test']
    data_test = data['x_test'].reshape(-1,512)
    ###########ORCA split
    np.random.seed(0)
    size_fraction_background = 1/2
    _, data_test, _, labels_test = train_test_split(data_test, labels_test, test_size=size_fraction_background, train_size =size_fraction_background ,random_state=rand_number)
    drive_path_orca = 'C:\\Users\\Kyle\\OneDrive\\Transfer Master project\\orca_fork\\cl4ad\\orca\\latent\\thesis\\Sup-6 - large\\'
    predictions = np.load(drive_path_orca+'target_pred_conf_kyle20240920-012237.npz')
    target_pred_conf = predictions['target_pred_conf']
    #Last epoch predictions
    last = target_pred_conf[-1,...]
    pred_targets = last[1,:]
    ############
    #labels_test = data['labels_test'][data['ix_test']]
    #data_test = data['x_test'][data['ix_test']].reshape(-1,57)
    #embedded_test = inference(f'output/{runs}/vae.pth', data_test, labels_test)
    embedded_test = inference(f'output/{runs}/vae3.pth', data_test, labels_test)
    #embedded_leptoquark = inference(f'output/{runs}/vae3.pth', leptoquark, leptoquark_labels)
    #embedded_ato4l = inference(f'output/{runs}/vae3.pth', ato4l, ato4l_labels)
    #embedded_hChToTauNu = inference(f'output/{runs}/vae3.pth', hChToTauNu, hChToTauNu_labels)
    #embedded_hToTauTau = inference(f'output/{runs}/vae3.pth', hToTauTau, hToTauTau_labels)
    #data_test = data_test.reshape(-1,512)

    #Plot t-SNE
    def plot_tsne():
        p = 0.05
        idx = np.random.choice(a=[True, False], size = len(labels_test), p=[p, 1-p]) #Indexes to plot (1% of the dataset for the t-SNE plots)
        print("===Plotting t-SNE===")
        print(f"with {np.sum(idx)} background datapoints")
        #embed_points = np.concatenate((embedded_test[idx],embedded_leptoquark[:5000],embedded_ato4l[:5000],embedded_hChToTauNu[:5000],embedded_hToTauTau[:5000]),axis=0)
        #test_points = np.concatenate((data_test[idx],leptoquark[:5000],ato4l[:5000],hChToTauNu[:5000],hToTauTau[:5000]),axis=0)
        #labels_points = np.concatenate((labels_test[idx],leptoquark_labels[:5000], ato4l_labels[:5000],hChToTauNu_labels[:5000],hToTauTau_labels[:5000]),axis=0)
        # tSNE(embedded_test[idx], labels_test[idx], '2D-t_SNE of 6D embedding with self-supervised SimCLR', '2D-t_SNE of 6D signal embedding with SimCLR.pdf', drive_path+f'output/{runs}/plots/',
        #     dict_labels_color, dict_labels_names, rand_number, orca=False)
        # tSNE(data_test[idx], labels_test[idx], '2D-t_SNE of 57D source input (No Embedding)', '2D-t_SNE of 57D signal test data.pdf', drive_path+f'output/{runs}/plots/',
        #     dict_labels_color, dict_labels_names, rand_number, orca=False)
        # tSNE(embedded_test[idx], labels_test[idx], '2D-t_SNE of 6D embedding colored by true labels', '2D-t_SNE of 6D embedding colored by true labels.pdf', drive_path+f'output/{runs}/plots/',
        #     dict_labels_color, dict_labels_names, rand_number, orca=False)
        tSNE(embedded_test[idx], pred_targets[idx], '2D-t_SNE of 6D embedding colored by ORCA labels', '2D-t_SNE of 6D embedding colored by ORCA labels.pdf', drive_path+f'output/{runs}/plots/',
            dict_labels_color, dict_labels_names, rand_number, orca=False)
    
    #Plot ROC curve (with AUC)
    def plot_roc():
        print("===Plotting ROC curve===")
        labels = labels_test
        #All signal vs background class ROC
        #predictions = classification_score(f'output/{runs}/vae3.pth', f'output/{runs}/head3.pth', data_test, labels_test, mode='roc')
        #predictions_noembedding = classification_score('NoEmbedding', 'output/NoEmbeddingADC_Delphes/head_kfold.pth', data_test, labels_test, mode='roc')
        #plot_ROC(predictions, labels, title='ROC curve with AUC for SimCLR embedding with supervised linear evaluation', filename='ROC curve with AUC for SimCLR embedding with supervised linear evaluation.pdf', folder=f'output/{runs}/plots/')
        #plot_ROC(predictions_noembedding, labels, title='ROC curve with AUC for No Embedding with supervised linear evaluation', filename='ROC curve with AUC for No Embedding with supervised linear evaluation.pdf', folder=f'output/{runs}/plots/')
        #One vs All ROC
        predictions = classification_score(f'output/{runs}/vae1.pth', f'output/{runs}/head1.pth', data_test, labels_test, mode='all')
        predictions_noembedding = classification_score('NoEmbedding', 'output/NoEmbeddingJetClass_kfolded/head.pth', data_test, labels_test, mode='all')
        plot_ROC_multiclass(predictions, labels, title='ROC curves with AUC for supervised SimCLR embedding', filename='OnevsAll ROC curve with AUC for embedding with supervised linear evaluation.pdf', folder=f'output/{runs}/plots/', num_classes=10)
        plot_ROC_multiclass(predictions_noembedding, labels, title='ROC curves with AUC for No Embedding', filename='OnevsAll ROC curve with AUC for balanced No Embedding with supervised linear evaluation.pdf', folder=f'output/{runs}/plots/', num_classes=10)

    def plot_pca(dimension=2):
        p = 0.01
        idx = np.random.choice(a=[True, False], size = len(labels_test), p=[p, 1-p]) #Indexes to plot (1% of the dataset for the PCA plots)
        print("===Plotting PCA===")
        print(f"with {np.sum(idx)} datapoints")
        labels = labels_test
        plot_PCA(embedded_test[idx], labels[idx], f'{dimension}D PCA of 48D embedding with SimCLR', f'{dimension}D PCA of 48D embedding with SimCLR.pdf',
                 dict_labels_color, dict_labels_names,f'output/{runs}/plots/' ,rand_number, dimension, orca=False)
        plot_PCA(data_test[idx], labels[idx], f'{dimension}D PCA of 57D test data', f'{dimension}D PCA of 57D test data.pdf',
                 dict_labels_color, dict_labels_names,f'output/{runs}/plots/' ,rand_number, dimension, orca=False)

    def plot_corner(embedded_test, subfolder="", iteration=""):
        p = 0.05
        idx = np.random.choice(a=[True, False], size = len(labels_test), p=[p, 1-p]) #Indexes to plot (1% of the dataset for the PCA plots)
        print("===Plotting Cornerplot===")
        print(f"with {np.sum(idx)} datapoints")
        labels = labels_test
        """ corner_plot(embedded_test[idx], labels[idx], f'Corner plot of (3D-PCA) of the embedding with anomaly leptoquark {iteration}', 
                    f'Corner plot of (3D-PCA) of the embedding with anomaly leptoquark{iteration}.png',dict_labels_color, dict_labels_names, pca=False, normalize=True, background='one', anomalies=['leptoquark'], folder=f'output/{runs}/plots/{subfolder}')
        corner_plot(embedded_test[idx], labels[idx], f'Corner plot of (3D-PCA) of the embedding with anomaly ato4l {iteration}', 
                    f'Corner plot of (3D-PCA) of the embedding with anomaly ato4l{iteration}.png',dict_labels_color, dict_labels_names, pca=False, normalize=True, background='one', anomalies=['ato4l'], folder=f'output/{runs}/plots/{subfolder}')
        corner_plot(embedded_test[idx], labels[idx], f'Corner plot of (3D-PCA) of the embedding with anomaly hChToTauNu {iteration}', 
                    f'Corner plot of (3D-PCA) of the embedding with anomaly hChToTauNu{iteration}.png',dict_labels_color, dict_labels_names, pca=False, normalize=True, background='one', anomalies=['hChToTauNu'], folder=f'output/{runs}/plots/{subfolder}')
        corner_plot(embedded_test[idx], labels[idx], f'Corner plot of (3D-PCA) of the embedding with anomaly hToTauTau {iteration}', 
                    f'Corner plot of (3D-PCA) of the embedding with anomaly hToTauTau{iteration}.png',dict_labels_color, dict_labels_names, pca=False, normalize=True, background='one', anomalies=['hToTauTau'], folder=f'output/{runs}/plots/{subfolder}') """
        
        """ for i, key in enumerate(dict_labels_names.values()):
            print(f"Corner plot for anomaly {key}")
            if i != 0:
                corner_plot(embedded_test[idx], labels[idx], f'Corner plot of (3D-PCA) of the embedding with anomaly {key}{iteration}', 
                            f'Corner plot of (3D-PCA) of the embedding with anomaly {key}{iteration}.png',dict_labels_color, dict_labels_names, pca=False, normalize=True, background='one', anomalies=[key], folder=f'output/{runs}/plots/{subfolder}') """
        # corner_plot(embedded_test[idx], labels[idx], f'Corner plot of (3D-PCA) of the embedding with anomaly all{iteration}', 
        #                     f'Corner plot of (3D-PCA) of the embedding with anomaly all{iteration}.png',dict_labels_color, dict_labels_names, pca=False, normalize=False, background='one', anomalies=['hToBB','hToCC','hToGG','hTo4q','hTolvqq','tTobqq','tToblv','WToqq','ZToqq'], folder=f'output/{runs}/plots/{subfolder}')
        corner_plot(embedded_test[idx], labels[idx], f'Corner plot of (3D-PCA) of the embedding with anomaly htocc_htobb{iteration}', 
                            f'Corner plot of (3D-PCA) of the embedding with anomaly htocc_htobb{iteration}.png',dict_labels_color, dict_labels_names, pca=False, normalize=False, background='one', anomalies=['hToBB','hToCC'], folder=f'output/{runs}/plots/{subfolder}')
    def animate_method(data_test, labels_test, method='corner'):
        """
            methods implemented so far: 'corner', ...
        """
        if method == 'corner':
            print(f"Animating corner plots for all anomalies")
            function = plot_corner
            types = ["leptoquark", "ato4l", "hChToTauNu", "hToTauTau"]

        subfolder = 'steps/'
        #Load all of the checkpoints (steps) of the animation
        files = [f for f in os.listdir(f'output/{runs}/checkpoints')]
        #Sort the files
        files.sort(key=natural_keys)
        #Run the method and get the output in steps folder
        for i, file in enumerate(files):
            path = f"output/{runs}/checkpoints/" + file
            embed = inference(path, data_test, labels_test)
            function(embed, subfolder, iteration = i)
        #Get all the images
        images_files = [image_file for image_file in os.listdir(f'output/{runs}/plots/steps')]
        #Sort the images
        images_files.sort(key=natural_keys)
        #Load and make gif for different anomaly types
        for type in types:
            images = [] #Empty array to store images for gif
            for image_file in images_files:
                if type in image_file:
                    image = Image.open(f'output/{runs}/plots/steps/{image_file}')
                    images.append(image)
            print(f"Image array length: {len(images)}")
            #Make the gif and save it
            animate(images, type, runs=runs, method=method)

    #plot_tsne()
    #plot_roc()
    #plot_pca(dimension=2)
    plot_corner(embedded_test=embedded_test)
    #animate_method(data_test, labels_test, method='corner')

if __name__ == '__main__':
    main('runs303')