import numpy as np
from plotter import inference
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from interpretation import get_gradient_JetClass, standardize_gradients, get_subspace_eigenvectors, dict_labels_color_JetClass, dict_labels_names_JetClass
import torch
from models import SimpleDense_JetClass
import matplotlib.text as mtext
import h5py

class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle, usetex=False, **self.text_props)
        handlebox.add_artist(title)
        return title
    
def get_embedding(runs):
    """ 
    Get the embedding + labels from JetClass dataset """

    #Load embedding (test (train/val used for model optim))
    drive_path = 'C:\\Users\\Kyle\\OneDrive\\Transfer Master project\\orca_fork\\cl4ad\\cl\\cl\\'
    #data = np.load(drive_path+'jetclass_dataset/JetClass_background_signal_reshaped.npz')
    #labels_test = data['labels_test']
    #data_test = data['x_test'].reshape(-1,512)
    with h5py.File(drive_path+f'JetClass_kfolded.hdf5', 'r') as f:
        data_test = np.array(f['x_test'][...]).reshape(-1,512)
        labels_test = np.array(f['labels_test'][...])
    embedded_test = inference(f'output/{runs}/vae1.pth', data_test, labels_test)
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


def plot_latent_gradient(embedding, labels, jet_feat, classes, jet_feat_name, title, rand_state, runs, taudivisor=False, palette='rocket'):
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
    df = df.sample(frac=0.05, replace=False, random_state=rand_state)
    #Rename for the legend
    legend_name = r'jet $\phi$'
    df = df.rename(columns={jet_feat_name: legend_name})
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
            'markers': type_to_marker, 
         }
    else:
        plot_kws = {
            's': 2,
        }
    corner = sns.pairplot(data=df.drop("Label", axis=1), hue=legend_name, kind='scatter', palette=palette, corner=True, plot_kws=plot_kws)
    #corner.figure.suptitle(title)
    #Set alpha and markersize in legend seperately
    #corner.legend.set_title("test")
    corner.add_legend(markerscale=5)
    #leg = corner._legend
    #leg.set_title("test")
    plt.show()
    plt.close("all")
    # Saves plot and reports success 
    filename = title + '.png'
    subfolder = os.path.dirname(__file__)
    subfolder = os.path.join(subfolder, f'output/{runs}/plots/')
    os.makedirs(subfolder, exist_ok=True)
    file_path = os.path.join(subfolder, filename)
    corner.savefig(file_path, dpi=300) 
    plt.close('all')
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

def plot_tau2overtau1_histogram(embedding, labels, jet_feat, classes, jet_feat_name, title, rand_state, runs, taudivisor=False):
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
    df = df.sample(frac=0.05, replace=False, random_state=rand_state)
    print(f"Plotting {df.shape} events")
    print(df)
    #Plot the cornerplot seperately for every class and overlay them
    for dim in range(np.shape(embedding)[1]):
        fig, ax = plt.subplots()
        plt.grid(zorder=-1)
        palette_list = ['Greys', 'GnBu', 'Greens', 'Greens_d', 'Reds', 'Purples', 'RdPu', 'Oranges', 'Blues']
        handles = []
        labels = []
        for num in classes:
            palette = palette_list[num]
            df_class = df[df["Label"]==num]
            print(df_class)
            cbar_kws = {
                 'alpha': 0.1
            }
            sns.histplot(data=df_class, x=f'Dimension {dim+1}', hue=jet_feat_name, palette=palette, legend=True, ax=ax, multiple="dodge")
            legend = ax.get_legend()
            handle = legend.legend_handles
            text = legend.texts
            for t in text:
                labels.append(t.get_text())
            handles.extend(handle)
        ax.legend([dict_labels_names[classes[0]], handles[0], handles[1], dict_labels_names[classes[1]], handles[2], handles[3]], ["", labels[0], labels[1], "", labels[2], labels[3]], title=r'$\tau_4/\tau_2$', 
                handler_map ={str: LegendTitle({'fontsize': 8})}, title_fontsize = 12)
        #plt.xlim(-0.1,14)
        #ax.figure.suptitle(title)
        #plt.show()
        
        # Saves plot and reports success 
        filename = title + f'_dimension{dim+1}' + '.png'
        subfolder = os.path.dirname(__file__)
        subfolder = os.path.join(subfolder, f'output/{runs}/plots/')
        os.makedirs(subfolder, exist_ok=True)
        file_path = os.path.join(subfolder, filename)
        plt.savefig(file_path, dpi=300) 
        #plt.close('all')
        print(f"Tau2overTau1 histogram saved as '{filename}'")
     

def plot_onetile_with_subspace_first_eigenvector(tile,embedding,labels,jet_feat,classes,jet_feat_name,title,rand_state,runs):
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
    0: 'QCD', 
    1: 'hToBB', 
    2: r'$\mathrm{H} \rightarrow \mathrm{c\overline{c}}$', 
    3: 'hToGG', 
    4: r'$\mathrm{H}\rightarrow \mathrm{4q}$', 
    5: 'hTolvqq', 
    6: r"$\mathrm{t}\rightarrow \mathrm{bqq'}$", 
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
    runs = 'runs333'
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
    jet_feat_name = 'tau4overtau2'
    #jet_feat_name = 'jet_phi'
    taudivisor = 0.5
    #classes = [0,1,2,3,4,5,6,7,8,9] #Classes to include, see corresponding class names in dict_labels_names
    #classes = [0,1,2,3,4,5]
    #classes = [0,6,7,8,9]
    classes = [0,4]
    #Getters
    embedding, labels = get_embedding(runs)
    jet_feat = get_jet_feat(jet_feat_name, jet_feat_dict)
    #for jet_feat_name in jet_level_feat.values():
        #print(f'Plotting the jet-level feature: {jet_feat_name}')
        #jet_feat = get_jet_feat(jet_feat_name, jet_feat_dict)
        #Plot
        #plot_latent_gradient(embedding, labels, jet_feat, classes, jet_feat_name, title=f'Latent embedding colored by {jet_feat_name}', rand_state=rand_number, runs=runs, taudivisor=taudivisor)
        #tile = (1,2) #Choose which dimensions to plot
        #plot_onetile_from_cornerplot(tile, embedding, labels, jet_feat, classes, jet_feat_name, title=f'Latent embedding tile {tile} colored by {jet_feat_name}', rand_state=rand_number, runs=runs)
        #plot_onetile_with_subspace_first_eigenvector(tile, embedding, labels, jet_feat, classes, jet_feat_name, title=f'Latent embedding tile {tile} with eigenvector colored by {jet_feat_name}', rand_state=rand_number, runs=runs)
    plot_tau2overtau1_histogram(embedding, labels, jet_feat, classes, jet_feat_name, title=f'Histogram of embedding colored by {jet_feat_name}', rand_state=rand_number, runs=runs, taudivisor=taudivisor)
    #plot_latent_gradient(embedding, labels, jet_feat, classes, jet_feat_name, title=f'Latent embedding colored by {jet_feat_name}', rand_state=rand_number, runs=runs, taudivisor=taudivisor)
if __name__ == '__main__':
    main()
         