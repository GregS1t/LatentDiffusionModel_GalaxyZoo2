
# File with the some models to predict the noise in the latent space
# 1. NoisePredictor: A simple feedforward neural network to predict the noise in the latent space

# 2. UNet: A U-Net like architecture to predict the noise in the latent space


import torch.nn as nn

#
# Simple feed forward neural network to predict the noise in the latent space
# Tested on the GalaxyZoo dataset -> Poor results
#-------------------------------------------------------------------------------

class NoisePredictor(nn.Module):
            def __init__(self, latent_dim):
                super(NoisePredictor, self).__init__()
                self.net = nn.Sequential(
                    nn.Linear(latent_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, latent_dim)
                )

            def forward(self, z_t, t):
                # z_t is [batch_size, latent_dim]
                return self.net(z_t)
            

#
# U-Net like architecture to predict the noise in the latent space
#-------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetNoisePredictor(nn.Module):
    def __init__(self, in_channels, base_channels=64, time_emb_dim=256,
                    out_channels=None):
        """
        UNet pour la prédiction du bruit dans un LDM.
        
        Args:
            in_channels (int): channel number of the input tensor (equal to the latent_dim).
            base_channels (int): channel number of the first layer (default = 64).
            time_emb_dim (int): dimension de l'embedding du temps.
            out_channels (int): nombre de canaux en sortie (par défaut = in_channels).
        """
        super(UNetNoisePredictor, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # MLP pour encoder l'indice de temps
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        # Projection du time embedding pour l'injecter dans la couche bottleneck
        self.time_proj = nn.Linear(time_emb_dim, base_channels * 4)
        
        # Chemin descendant (encodeur)
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)  # Downsample
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1)
        
        # Chemin montant (décodeur)
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)  # après concaténation avec skip connection
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.conv6 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        
        # Couche finale pour ramener au nombre de canaux souhaité
        self.conv_final = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        self.relu = nn.ReLU()

    def forward(self, x, t):
        """
        Args:
            x (torch.Tensor): latent bruité de forme (B, in_channels, H, W)
            t (torch.Tensor): tenseur d'indices de temps, de forme (B,)
        Returns:
            torch.Tensor: prédiction du bruit de même forme que x.
        """
        # Encode l'indice de temps
        t = t.float().unsqueeze(1)  # (B, 1)
        time_emb = self.time_mlp(t)  # (B, time_emb_dim)
        
        # Chemin descendant
        x1 = self.relu(self.conv1(x))   # (B, base_channels, H, W)
        x1 = self.relu(self.conv2(x1))    # (B, base_channels, H, W)
        x2 = self.relu(self.down1(x1))    # (B, base_channels*2, H/2, W/2)
        x2 = self.relu(self.conv3(x2))    # (B, base_channels*2, H/2, W/2)
        x3 = self.relu(self.down2(x2))    # (B, base_channels*4, H/4, W/4)
        x3 = self.relu(self.conv4(x3))    # (B, base_channels*4, H/4, W/4)
        
        # Bottleneck avec injection du time embedding
        bottleneck = self.relu(self.bottleneck(x3))  # (B, base_channels*4, H/4, W/4)
        time_emb_proj = self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)  # (B, base_channels*4, 1, 1)
        bottleneck = bottleneck + time_emb_proj  # injection par addition (broadcast)
        
        # Chemin montant
        x_up1 = self.relu(self.up1(bottleneck))  # (B, base_channels*2, H/2, W/2)
        x_cat1 = torch.cat([x_up1, x2], dim=1)     # concaténation avec le skip connection
        x_dec1 = self.relu(self.conv5(x_cat1))     # (B, base_channels*2, H/2, W/2)
        x_up2 = self.relu(self.up2(x_dec1))         # (B, base_channels, H, W)
        x_cat2 = torch.cat([x_up2, x1], dim=1)       # concaténation avec le skip connection
        x_dec2 = self.relu(self.conv6(x_cat2))       # (B, base_channels, H, W)
        
        out = self.conv_final(x_dec2)  # (B, out_channels, H, W)
        return out
