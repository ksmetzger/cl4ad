import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    