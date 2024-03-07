import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CVAE(torch.nn.Module):
    '''
    Creates fully supervised CVAE Class
    Training architecture: input -> latent space μ representation -> Proj(μ) -> contrastive loss
    '''
    def __init__(self, latent_dim=6, layer_size_projection=16, **kwargs):
        super().__init__(**kwargs)

        self.mlp = nn.Sequential(
            nn.Linear(57, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        )

        self.z_mean = nn.Linear(64, latent_dim)
        self.z_log_var = nn.Linear(64, latent_dim)

        self.proj_head = nn.Sequential(
            nn.Linear(latent_dim, layer_size_projection),
            nn.LeakyReLU(),
            nn.Linear(layer_size_projection, latent_dim)
        )


    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        if type(mu)==np.ndarray:
            mu = torch.from_numpy(mu).to(dtype=torch.float32, device=DEVICE)
            logvar = torch.from_numpy(logvar).to(dtype=torch.float32, device=DEVICE)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def representation(self, x):
        x = self.mlp(x)
        mu, logvar = self.z_mean(x), self.z_log_var(x)
        z = self.reparameterize(mu, logvar)

        return z

    def forward(self, x):
        z = self.representation(x)
        z_proj = self.proj_head(z)

        return z_proj

class SimpleDense(torch.nn.Module):
    def __init__(self, latent_dim = 48, expanded_dim = 192):
        super(SimpleDense, self).__init__()
        self.latent_dim = latent_dim
        self.expanded_dim = expanded_dim

        self.encoder = torch.nn.Sequential(
            nn.Linear(57,52),
            nn.BatchNorm1d(52),
            nn.LeakyReLU(),
            nn.Linear(52,self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.LeakyReLU()

        )
        self.expander = torch.nn.Sequential(
            nn.Linear(self.latent_dim,96),
            nn.BatchNorm1d(96),
            nn.LeakyReLU(),
            nn.Linear(96,self.expanded_dim)
        )
    def representation(self, x):
        y = self.encoder(x)
        return y
    
    def forward(self, x):
        y = self.representation(x)
        z = self.expander(y)
        return z