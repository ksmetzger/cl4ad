print("Importing from 'nf_classifier.py'")
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader

import numpy as np
from nflows.nn.nets import ResidualNet
from nflows import transforms, distributions, flows

from argparse import ArgumentParser

class FlowModel: 
    def __init__(self, latent_size, batch_size, num_layers, device): 
        self.device = device
        self.latent_size = latent_size 
        self.batch_size = batch_size 
        
        # Assumes base distribution is standard normal
        base_dist = distributions.StandardNormal(shape=[latent_size])
        
        transform_layers = []
        for _ in range(num_layers):
            # Permutes ordering across the input dimensions 
            transform_layers.append(transforms.permutations.ReversePermutation(features=latent_size))
            # Applies affine transformation through invertible NN with specified hidden features
            transform_layers.append(transforms.autoregressive.MaskedAffineAutoregressiveTransform(
                features=latent_size, hidden_features=latent_size+2))
        
        # Creates normalizing flow model and initializes optimizer 
        transform_obj = transforms.CompositeTransform(transform_layers)
        self.flow_obj = flows.Flow(transform_obj, base_dist).to(device)
        self.optimizer = torch.optim.Adam(self.flow_obj.parameters(), lr=1e-2)
        # Adjust learning rate during training to use an exponential decay schedule
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        
        print(f"Num of trainable params: {sum(p.numel() for p in self.flow_obj.parameters() if p.requires_grad)}")
        
    def train(self, dataloader, epochs):
        for epoch in range(epochs): 
            for ix, data in enumerate(dataloader): 
                # if number of samples in current batch != batch size, skips for dimension concerns
                if data.shape[0] != self.batch_size: continue 
                    
                # Performs a foward pass and backprop to update model
                self.optimizer.zero_grad()
                loss = -self.flow_obj.log_prob(inputs=data).mean()
                loss.backward()
                self.optimizer.step()
                
            print(f"Iteration {i+1}: Avg. log prob = {-loss.item():.3f}")
            self.sheduler.step()
    
    def save_model(self, filename): 
        # Saves model weights to specified file 
        subfolder = os.path.join(os.path.dirname(__file__), '..', 'model_weights')
        os.makedirs(subfolder, exist_ok=True)
        saved_weights_path = os.path.join(subfolder, filename)
        torch.save(self.flow_obj.state_dict(), saved_weights_path)
        
    def load_model(self, filename, device): 
        # Load the mdoel weights from a file 
        subfolder = os.path.join(os.path.dirname(__file__), '..', 'model_weights')
        os.makedirs(subfolder, exist_ok=True)
        saved_weights_path = os.path.join(subfolder, filename)
        self.flow_obj.load_state_dict(torch.load(saved_weights_path, map_location=device))
    
    
if __name__ == "__main__": 
    parser = ArgumentParser()
    parser.add_argument('--latent_size', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=1002)
    parser.add_argument('--num_layers', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--latent_filename', type=str, default='zscore_rep.npy')
    parser.add_argument('--classifier_filename', type=str, default='itworkedyay.pt')
    args = parser.parse_args()
    
    # Connects to GPU, rasies an error if CPU is used
    print("Connecting to GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Loads the data from npz file into tensorflow dataloader 
    print("Loading latent space data")
    data = np.load('../data/' + args.latent_filename)
    train_data = torch.from_numpy(data).to(dtype=torch.float32, device=device)
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    
    # Creates FlowModel instance and trains model 
    print("Creating Flow Model")
    flow_model = FlowModel(args.latent_size, args.batch_size, args.num_layers, device)
    print("Training Flow Model")
    flow_model.train(train_data_loader, args.epochs)
    print("Saving Flow Model")
    flow_model.save_model(args.classifier_filename)
    