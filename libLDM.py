# Description: Latent Diffusion Model for image generation  

# Author: Grégory Sainton
# Institution: Observatoire de Paris - PSL University
# Date: 2025-01-06


import torch
import torch.nn as nn

class LatentDiffusionModel(nn.Module):
    """
    Latent Diffusion Model for image generation

    Args:
        network (nn.Module): Neural network for predicting noise
        n_steps (int): Number of steps in the diffusion process
        min_beta (float): Minimum value of beta
        max_beta (float): Maximum value of beta
        device (str): Device to run the model on
        latent_dim (int): Dimension of the latent space
        
    """

    def __init__(self, network, n_steps=1000, min_beta=1e-4, max_beta=0.02, 
                 device=None, latent_dim=128):
        super(LatentDiffusionModel, self).__init__()
        self.network = network.to(device)
        self.n_steps = n_steps
        self.latent_dim = latent_dim
        self.device = device

        # Diffusion hyper param'
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) \
                                        + 1e-6 for i in range(len(self.alphas))]).to(device)

    def q_t(self, z_0, t, eps=None):
        """
        Compute the diffusion process at time t 
        on the latent space : 
        q(z_t| z_0) = N(z_t | mu_t, sigma_t) with 

        beta_i = betas[i]
        alpha_bar_t = prod_{i=1}^{t} (1 - beta_i)

        mu_t = sqrt(1 - alpha_bar_t) * z_0
        sigma_t = sqrt(alpha_bar_t) * I

        z_t = mu_t + sigma_t * eps

        Args:
            z_0 (torch.Tensor): Initial latent vector
            t (int): Time step
            eps (torch.Tensor): Noise tensor
        """
        # alpha_bar_t: [batch_size] after indexing
        alpha_bar_t = self.alpha_bars[t]  # [batch_size]
        alpha_bar_t = alpha_bar_t.unsqueeze(1)  # [batch_size, 1]
        # Broadcast alpha_bar_t to match z_0: [batch_size, latent_dim]
        # Since latent_dim is the same for all samples, we can expand:
        alpha_bar_t = alpha_bar_t.expand(-1, self.latent_dim)  # [batch_size, latent_dim]

        z_t = (alpha_bar_t.sqrt() * z_0) + ((1 - alpha_bar_t).sqrt() * eps)
        # z_t: [batch_size, latent_dim]
        return z_t

    def predict_eps(self, z_t, t):
        """
        Predict the noise added at timestep t

        """
        return self.network(z_t, t)

    def compute_loss(self, z_0, t):
        """
        Loss calculation for the diffusion model
        In comments, the dimensions of the tensors are indicated
        
        Formula:
        L = ||eps_theta - eps||^2   with eps ~ N(0, I)

        Args:
            z_0 (torch.Tensor): Initial latent vector
            t (int): Time step
        """
        eps = torch.randn_like(z_0)        # [batch_size, latent_dim]
        z_t = self.q_t(z_0, t, eps)        # [batch_size, latent_dim]
        eps_theta = self.predict_eps(z_t, t)  # [batch_size, latent_dim]
        
        loss_fn = nn.MSELoss()
        return loss_fn(eps_theta, eps)
    
    @torch.no_grad()
    def p_sample(self, z_t, t):
        """
        Reverse diffusion process
        p(z_0 | z_t) = N(z_0 | mu_t, sigma_t) with

        beta_i = betas[i]
        alpha_bar_t = prod_{i=1}^{t} (1 - beta_i)

        Args:
            z_t (torch.Tensor): Latent vector at time t
            t (int): Time step

        """

        # Extract scalars per batch element
        beta_t = self.betas[t]          # [batch_size]
        alpha_t = self.alphas[t]        # [batch_size]
        alpha_bar_t = self.alpha_bars[t]# [batch_size]

        # alpha_bar_prev (for t>0), else 1.0
        alpha_bar_prev = torch.where(
            t > 0,
            self.alpha_bars[(t-1).clamp(min=0)],
            torch.ones_like(alpha_bar_t)
        )  # [batch_size]

        # Expand all to [batch_size, latent_dim]
        beta_t = beta_t.unsqueeze(1).expand(-1, self.latent_dim)
        alpha_t = alpha_t.unsqueeze(1).expand(-1, self.latent_dim)
        alpha_bar_t = alpha_bar_t.unsqueeze(1).expand(-1, self.latent_dim)
        alpha_bar_prev = alpha_bar_prev.unsqueeze(1).expand(-1, self.latent_dim)

        eps_theta = self.predict_eps(z_t, t)  # [batch_size, latent_dim]

        noise = torch.randn_like(z_t)         # [batch_size, latent_dim]
        sigma_t = ((1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t).sqrt() # [batch_size, latent_dim]

        z_prev = (z_t - (1 - alpha_t).sqrt() * eps_theta) / alpha_t.sqrt()

        # Add noise if t>0
        mask = (t > 0).unsqueeze(1).expand(-1, self.latent_dim)  # [batch_size, latent_dim]
        z_prev = z_prev + torch.where(mask, sigma_t * noise, torch.zeros_like(z_t))
        return z_prev
    

    @torch.no_grad()
    def generate_sample(self, num_samples):
        """
        Generate samples from the model
    
        """
        z_t = torch.randn(num_samples, self.latent_dim).to(self.device)

        for t in reversed(range(self.n_steps)):
            t_tensor = torch.full((num_samples, 1), t,
                                  dtype=torch.long).to(self.device)
            z_t = self.p_sample(z_t, t_tensor)

        return z_t
    
    @torch.no_grad()
    def decode_latent(self, z_0, decoder):
        """
        Decode the latent vector to an image

        Args:
            z_0 (torch.Tensor): Latent vector
            decoder (nn.Module): pre-trained decoder

        Returns:
            torch.Tensor: Decoded image
        """
        return decoder(z_0)
    
# ----------------------------------------------------------
# Example of a noise predictor network
# ----------------------------------------------------------
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