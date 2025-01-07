
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        """
        Encoder: Extracts latent features from input images.
        Args:
            latent_dim (int): Dimensionality of the latent space.
        """
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, latent_dim),
        )
        self.latent_norm = nn.Tanh()
        
    def forward(self, x):
        z = self.encoder(x)
        return self.latent_norm(z)

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        """
        Decoder: Reconstructs images from latent features.
        Args:
            latent_dim (int): Dimensionality of the latent space.
        """
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),  # Match encoder's downsampled size
            nn.LeakyReLU(0.1),

            nn.Unflatten(1, (512, 4, 4)),  

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output values between [0, 1]
        )

    def forward(self, x):
        return self.decoder(x)
    
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

