
# Description: This file contains the implementation of the Autoencoder 
# model with 4 layers for the latent dimensionality reduction.
#
# Author: Grégory Sainton 
# Date: 2025-01-06

import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_channels=4):
        """
        Encoder: Extracts latent features from input images.
        Args:
            latent_channels (int): Number of channels in the latent space.
        """
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(128, latent_channels,
                      kernel_size=3,
                      stride=1, padding=1)  # -> (B, latent_channels, 16, 16)
        )
        self.latent_norm = nn.Tanh()
        
    def forward(self, x):
        z = self.encoder(x)
        return self.latent_norm(z)

class Decoder(nn.Module):
    def __init__(self, latent_channels=4):
        """
        Decoder: Reconstructs images from spatial latent features.
        Args:
            latent_channels (int): Number of channels in the latent space.
        """
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            # Input: (B, latent_channels, 16, 16)
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4,
                               stride=2, padding=1),  # -> (B, 128, 32, 32)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4,
                               stride=2, padding=1),  # -> (B, 64, 64, 64)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # -> (B, 3, 64, 64)
            nn.Sigmoid()  # Ensure outputs are in [0, 1]
        )


    def forward(self, x):
        return self.decoder(x)
    
class Autoencoder(nn.Module):
    def __init__(self, latent_channels):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded