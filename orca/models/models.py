import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np


#Inspired by https://github.com/katyagovorkova/cl4ad/blob/main/src/cl/models.py and extended to fit orca
#five-layer model working directly from Delphes input
class CVAE_direct(nn.Module):
    def __init__(self, num_classes = 8):
        super(CVAE_direct, self).__init__()
        self.num_classes = num_classes
        
        self.encoder = torch.nn.Sequential(
            nn.Linear(57,32),          #Dimensionality of Delphes input (57)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(),              
            nn.Linear(32,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(),              
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(),              
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(),             
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(),             
            )
        self.linear = NormedLinear(512, self.num_classes)
        
        
    def forward(self, x):
        out = self.encoder(x)
        out_linear = self.linear(out)
        return out_linear, out

#Inspired by https://github.com/katyagovorkova/cl4ad/blob/main/src/cl/models.py and extended to fit orca
#five-layer model working from the embedding
class CVAE_latent(nn.Module):
    def __init__(self, num_classes = 8):
        super(CVAE_latent, self).__init__()
        self.num_classes = num_classes
        
        self.encoder = torch.nn.Sequential(
            nn.Linear(6,32),          #Dimensionality of input from the embedding (6)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),              
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),             
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),             
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),            
            )
        self.linear = NormedLinear(512, self.num_classes)
        
        
    def forward(self, x):
        out = self.encoder(x)
        out_linear = self.linear(out)
        return out_linear, out

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return 10 * out
      
    
#Simpler two-layer model starting from embedding input
class CVAE_latent_simple(nn.Module):
    def __init__(self, num_classes = 8):
        super(CVAE_latent_simple, self).__init__()
        self.num_classes = num_classes
        
        self.encoder = torch.nn.Sequential(
            nn.Linear(6,12),  
            nn.BatchNorm1d(12),
            nn.LeakyReLU(),
            nn.Linear(12,24),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(),
            
            )
        self.linear = NormedLinear(24, self.num_classes)
        

    def forward(self, x):
        out = self.encoder(x)
        out_linear = self.linear(out)
        return out_linear, out

#Simpler two-layer model starting directly from the Delphes dataset
class CVAE_direct_simple(nn.Module):
    def __init__(self, num_classes = 8):
        super(CVAE_direct_simple, self).__init__()
        self.num_classes = num_classes
        
        self.encoder = torch.nn.Sequential(
            nn.Linear(57,48),  
            nn.BatchNorm1d(48),
            nn.LeakyReLU(),
            nn.Dropout(),          
            nn.Linear(48,24),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(),
            nn.Dropout()
            
            )
        self.linear = NormedLinear(24, self.num_classes)

    def forward(self, x):
        out = self.encoder(x)
        out_linear = self.linear(out)
        return out_linear, out
