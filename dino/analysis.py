import numpy as np
from plotter import inference, transformer_args_jetclass
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
#from interpretation import get_gradient_JetClass, standardize_gradients, get_subspace_eigenvectors, dict_labels_color_JetClass, dict_labels_names_JetClass
import torch
from transformer import TransformerEncoder

def get_embedding(runs):
    """ 
    Get the embedding + labels from JetClass dataset """

    #Load embedding (test (train/val used for model optim))
    drive_path = 'C:\\Users\\Kyle\\OneDrive\\Transfer Master project\\orca_fork\\cl4ad\\cl\\cl\\'
    data = np.load(drive_path+'jetclass_dataset/JetClass_background_signal_reshaped.npz')
    labels_test = data['labels_test']
    data_test = data['x_test'].reshape(-1,128,4)
    embedded_test = inference(f'output/{runs}/_teacher_dino_transformer.pth', data_test, labels_test)
    return embedded_test, labels_test

def get_jet_feat(jet_feat_name, jet_feat_dict):
    """
    For each collision event get the corresponding jet feature in an array (n_event,)"""

    #Load the jet_level feature dataset
    drive_path = 'C:\\Users\\Kyle\\OneDrive\\Transfer Master project\\orca_fork\\cl4ad\\cl\\cl\\'
    data = np.load(drive_path+'jetclass_dataset/JetClass_jet_level_features.npz')
    if jet_feat_name=='tau2overtau1':
        jet_feat = data['x_test_jet_feat'][:, 7] / data['x_test_jet_feat'][:, 6]
    elif jet_feat_name == 'tau3overtau2':
         jet_feat = data['x_test_jet_feat'][:, 8] / data['x_test_jet_feat'][:, 7]
         #print(np.shape(data['x_test_jet_feat'][:, 8]))
         #print(np.shape(jet_feat))
    elif jet_feat_name == 'tau4overtau2':
         jet_feat = data['x_test_jet_feat'][:, 9] / data['x_test_jet_feat'][:, 7]
    else:
        label = jet_feat_dict[jet_feat_name]
        jet_feat = data['x_test_jet_feat'][:,label]
    
    #mask = jet_feat > 1
    #print(jet_feat[mask])
    return jet_feat


def plot_latent_gradient(embedding, labels, jet_feat, classes, jet_feat_name, title, rand_state, runs, taudivisor=False):
    """
    Plot a cornerplot of the selected embedding classes with a gradient coloring dependent on the jet level feature given in jet_feat.
    
    Input:
        embedding: Array of embedded events: array(n_events, latent_dim)
        labels: Array with truth labels for each event (classes 0-9): array(n_events,)
        jet_feat: Array with the jet level feature for each event: array(n_events,)
        classes: List of classes (labels 0-9) that should be included: List[Int]
        jet_feat_name: Name of the selected jet_level feature: String
        title: Title of the plot: String
        rand_state: Random number for setting seed when sampling from embedding: Int
        runs: Which run/embedding to use: String
    """
    #Concatenate the embedding with the jet_feature and convert to dataFrame
    sns_dict = {}
    for i in range(np.shape(embedding)[1]): #Iterate through dimensions of the embedding
            sns_dict["Dimension {}".format(i+1)] = embedding[:,i]
    sns_dict[jet_feat_name] = jet_feat
    sns_dict["Label"] = labels
    df = pd.DataFrame(sns_dict)
    #Only select the classes given by the input
    df = df[df["Label"].isin(classes)]
    print(df)
    #For taux/tauy split into two bins with value taudivisor
    if taudivisor:
        df[jet_feat_name] = pd.cut(df[jet_feat_name], bins=[0,taudivisor,1])
    else:
         #Bin the jet_feat with pd.qcut in order to pass it to hue
        df[jet_feat_name] = pd.qcut(df[jet_feat_name], q=5, precision=0)
    #Downsample for plotting
    df = df.sample(frac=0.005, replace=False, random_state=rand_state)
    print(f"Plotting {df.shape} events")
    print(df)
    #Plot
    if len(classes) == 2:
         marker_styles = ['s', 'D']
         type_to_marker = {classes[0]: 's', classes[1]: 'D'}
         plot_kws = {
            's': 5,
            'alpha': 0.8,
            'style': df['Label'],
            'markers': type_to_marker
         }
    else:
        plot_kws = {
            's': 2,
        }
    corner = sns.pairplot(data=df.drop("Label", axis=1), hue=jet_feat_name, kind='scatter', palette="rocket", corner=True, plot_kws=plot_kws)
    corner.figure.suptitle(title)
    #Set alpha and markersize in legend seperately
    corner.add_legend(markerscale=5)
    plt.show()

    # Saves plot and reports success 
    filename = title + '.png'
    subfolder = os.path.dirname(__file__)
    subfolder = os.path.join(subfolder, f'output/{runs}/plots/')
    os.makedirs(subfolder, exist_ok=True)
    file_path = os.path.join(subfolder, filename)
    corner.savefig(file_path) 
    #plt.close('all')
    print(f"Corner Plot saved as '{filename}'")

def plot_onetile_from_cornerplot(tile, embedding, labels, jet_feat, classes, jet_feat_name, title, rand_state, runs):
    """
    Plot just one tile of the cornerplot from: plot_latent_gradient()
    
    Input:
        embedding: Array of embedded events: array(n_events, latent_dim)
        labels: Array with truth labels for each event (classes 0-9): array(n_events,)
        jet_feat: Array with the jet level feature for each event: array(n_events,)
        classes: List of classes (labels 0-9) that should be included: List[Int]
        jet_feat_name: Name of the selected jet_level feature: String
        title: Title of the plot: String
        rand_state: Random number for setting seed when sampling from embedding: Int
        runs: Which run/embedding to use: String
        tile: Which tile to choose: Tuple
    """
    #Concatenate the embedding with the jet_feature and convert to dataFrame
    sns_dict = {}
    for i in range(np.shape(embedding)[1]): #Iterate through dimensions of the embedding
            sns_dict["Dimension {}".format(i+1)] = embedding[:,i]
    sns_dict[jet_feat_name] = jet_feat
    sns_dict["Label"] = labels
    df = pd.DataFrame(sns_dict)
    #Only select the classes given by the input
    df = df[df["Label"].isin(classes)]
    #Bin the jet_feat with pd.qcut in order to pass it to hue
    df[jet_feat_name] = pd.qcut(df[jet_feat_name], q=5, precision=0)
    #Downsample for plotting
    df = df.sample(frac=0.005, replace=False, random_state=rand_state)
    print(f"Plotting {df.shape} events")
    print(df)

    #Plot the tile
    scatter = sns.scatterplot(data=df, x="Dimension {}".format(tile[0]), y="Dimension {}".format(tile[1]), hue=jet_feat_name, palette='rocket')
    #plt.show()

    # Saves plot and reports success 
    filename = title + '.png'
    subfolder = os.path.dirname(__file__)
    subfolder = os.path.join(subfolder, f'output/{runs}/plots/')
    os.makedirs(subfolder, exist_ok=True)
    file_path = os.path.join(subfolder, filename)
    #scatter.savefig(file_path) 
    #plt.close('all')
    print(f"Scatter tile Plot saved as '{filename}'")

    return scatter

""" def plot_onetile_with_subspace_first_eigenvector(tile,embedding,labels,jet_feat,classes,jet_feat_name,title,rand_state,runs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    signal = dict_labels_names[classes[-1]] #Treat the last class input as the signal to calculate the gradients from the classifier for.
    latent_dim = 6
    num_classes = 10
    model = SimpleDense_JetClass(latent_dim)
    head = torch.nn.Linear(latent_dim, num_classes)
    model.load_state_dict(torch.load(f'output/{runs}/vae.pth', map_location=torch.device(device)))
    head.load_state_dict(torch.load(f'output/{runs}/head.pth', map_location=torch.device(device)))

    #Get gradients from interpretation.py using the subspace inter. from CHAKRAVARTI et al.
    input_data, input_labels, gradients = get_gradient_JetClass(signal=signal, model=model, classification_head=head, drive_path='C:\\Users\\Kyle\\OneDrive\\Transfer Master project\\orca_fork\\cl4ad\\cl\\cl\\', device=device)
    std_gradients = standardize_gradients(gradients)
    #Compute the eigenvectors
    eigenvectors, eigenvalues = get_subspace_eigenvectors(std_gradients)

    #Plot the tile with the vector
    fig, (ax1, ax2) = plt.subplots(1,2)
    plt.sca(ax1)
    scatter = plot_onetile_from_cornerplot(tile, embedding, labels, jet_feat, classes, jet_feat_name, title, rand_state=rand_state, runs=runs)
    plt.sca(ax2)
    first_eigen = eigenvectors[0]
    ax2.arrow(0, 0, first_eigen[tile[0]]*10, first_eigen[tile[1]]*10, head_width=.1, color=dict_labels_color_JetClass[dict_labels_names_JetClass[signal]])
    plt.xlabel(f'Dimension {tile[0]}')
    plt.ylabel(f'Dimension {tile[1]}')
    plt.show()

    # Saves plot and reports success 
    filename = title + '.png'
    subfolder = os.path.dirname(__file__)
    subfolder = os.path.join(subfolder, f'output/{runs}/plots/')
    os.makedirs(subfolder, exist_ok=True)
    file_path = os.path.join(subfolder, filename)
    plt.savefig(file_path)
    #plt.close('all')
    print(f"Scatter tile Plot with first subspace eigenvector saved as '{filename}'")
 """




jet_feat_dict = {
    'jet_pt': 0,
    'jet_eta': 1,
    'jet_phi': 2,
    'jet_energy': 3,
    'jet_nparticles': 4,
    'jet_sdmass': 5,
    'jet_tau1': 6,
    'jet_tau2': 7,
    'jet_tau3': 8,
    'jet_tau4': 9,
}
dict_labels_names = {
    0: 'QCD-background', 
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
jet_level_feat = {
    0: 'jet_pt',
    1: 'jet_eta',
    2: 'jet_phi',
    3: 'jet_energy',
    4:'jet_nparticles',
    5: 'jet_sdmass',
    6: 'jet_tau1',
    7: 'jet_tau2',
    8: 'jet_tau3',
    9: 'jet_tau4',
    10: 'tau2overtau1',
    11: 'tau3overtau2',
    12: 'tau4overtau2',
}
def main():
    rand_number = 0
    np.random.seed(rand_number)
    #Which embedding to use
    runs = 'runs93'
    #And select which jet_level feature to visualize in latent space
    """ The available jet-level features are:
            jet_pt
            jet_eta
            jet_phi
            jet_energy
            jet_nparticles
            jet_sdmass
            jet_tau1
            jet_tau2
            jet_tau3
            jet_tau4
            tau2overtau1 (2 prong) 
            tau3overtau2 (3 prong)
            tau4overtau2 (4 prong)
            """
    #jet_feat_name = 'jet_phi'
    taudivisor = False
    classes = [0,1,2,3,4,5,6,7,8,9] #Classes to include, see corresponding class names in dict_labels_names
    #classes = [0,1,2,3,4,5]
    #classes = [0,6,7,8,9]
    #classes = [6]
    #Getters
    embedding, labels = get_embedding(runs)
    #Loop in order to do inference only once! (Takes a while with DINO transformer encoder)
    for jet_feat_name in jet_level_feat.values():
        print(f'Plotting the jet-level feature: {jet_feat_name}')
        jet_feat = get_jet_feat(jet_feat_name, jet_feat_dict)
        #Plot
        plot_latent_gradient(embedding, labels, jet_feat, classes, jet_feat_name, title=f'Latent embedding colored by {jet_feat_name}', rand_state=rand_number, runs=runs, taudivisor=taudivisor)
        tile = (1,2) #Choose which dimensions to plot
        #plot_onetile_from_cornerplot(tile, embedding, labels, jet_feat, classes, jet_feat_name, title=f'Latent embedding tile {tile} colored by {jet_feat_name}', rand_state=rand_number, runs=runs)
        #plot_onetile_with_subspace_first_eigenvector(tile, embedding, labels, jet_feat, classes, jet_feat_name, title=f'Latent embedding tile {tile} with eigenvector colored by {jet_feat_name}', rand_state=rand_number, runs=runs)

if __name__ == '__main__':
    main()
         