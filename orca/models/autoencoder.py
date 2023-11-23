import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter



#Inspired by https://github.com/katyagovorkova/cl4ad/blob/main/src/cl/models.py and extended to fit orca
class CVAE(nn.Module):
    def __init__(self, num_classes = 8):
        super(CVAE, self).__init__()
        self.num_classes = num_classes
        
        self.encoder = torch.nn.Sequential(
            nn.Linear(57,32),       #Change 6 <-> 57 depending on the input (Delphes: 57, latent: 6)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            #Add additional two layers
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256,512)
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