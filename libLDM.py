# Description: Latent Diffusion Model for image generation  

# Author: Grégory Sainton
# Institution: Observatoire de Paris - PSL University
# Date: 2025-01-06


import torch
import torch.nn as nn

class LatentDiffusionModel_MLP(nn.Module):
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
        super(LatentDiffusionModel_MLP, self).__init__()
        self.network = network.to(device)
        self.n_steps = n_steps
        self.latent_dim = latent_dim
        self.device = device

        # Diffusion hyper param
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
        eps = torch.randn_like(z_0)           # [batch_size, latent_dim]
        z_t = self.q_t(z_0, t, eps)           # [batch_size, latent_dim]
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

class LatentDiffusionModel_UNET(nn.Module):
    """
    Latent Diffusion Model for image generation adapted for spatial tensors.
    
    Args:
        network (nn.Module): Noise prediction network (e.g. UNet).
        n_steps (int): Number of steps in the diffusion process.
        min_beta (float): Minimum value for beta.
        max_beta (float): Maximum value for beta.
        device (str): Device on which to run the model.
        latent_shape (tuple): Shape of the latent tensor, e.g. (C, H, W)
                              (C: number of channels, H: height, W: width).
    """

    def __init__(self, network, n_steps=1000, min_beta=1e-4, max_beta=0.02, 
                 device=None, latent_shape=(4, 16, 16)):
        super(LatentDiffusionModel_UNET, self).__init__()
        self.network = network.to(device)
        self.n_steps = n_steps
        self.latent_shape = latent_shape  # e.g. (C, H, W)
        self.device = device

        # Diffusion hyperparameters
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) + 1e-6 
                                         for i in range(len(self.alphas))]).to(device)


    def q_t(self, z_0, t, eps=None):
        """
        Computes the forward diffusion process in latent space.

        q(z_t| z_0) = N(z_t | mu_t, sigma_t) with 

        beta_i = betas[i]
        alpha_bar_t = prod_{i=1}^{t} (1 - beta_i)

        mu_t = sqrt(1 - alpha_bar_t) * z_0
        sigma_t = sqrt(alpha_bar_t) * I

        z_t = mu_t + sigma_t * eps
        
        The process is defined as:
            z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * eps,
        where alpha_bar_t is the cumulative product of (1 - beta_i) for i=1,...,t.
        
        Args:
            z_0 (torch.Tensor): Initial latent tensor of shape (B, C, H, W).
            t (torch.Tensor): Time step indices of shape (B,) (or (B, 1)).
            eps (torch.Tensor, optional): Noise tensor to add (if None, generated randomly).
            
        Returns:
            torch.Tensor: Noisy latent tensor z_t of shape (B, C, H, W).
        """
        if eps is None:
            eps = torch.randn_like(z_0)
        # Reshape alpha_bar_t for diffusion over all dimensions: (B, 1, 1, 1)
        # Note : This is the difference with the MLP version. It also as to be
        # done for the other variables in the other functions like p_sample
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        z_t = (alpha_bar_t.sqrt() * z_0) + ((1 - alpha_bar_t).sqrt() * eps)
        return z_t

    def predict_eps(self, z_t, t):
        """
        Predicts the noise added at time step t using the network (e.g. UNet).
        
        Args:
            z_t (torch.Tensor): Noisy latent tensor of shape (B, C, H, W).
            t (torch.Tensor): Time step indices.
        
        Returns:
            torch.Tensor: Noise prediction of shape (B, C, H, W).
        """
        return self.network(z_t, t)

    def compute_loss(self, z_0, t):
        """
        Computes the loss for the diffusion model.
        
        The loss is defined as:
            L = ||eps_theta - eps||^2,
        where eps ~ N(0, I) and eps_theta is the network's prediction.
        
        Args:
            z_0 (torch.Tensor): Initial latent tensor of shape (B, C, H, W).
            t (torch.Tensor): Time step indices.
        
        Returns:
            torch.Tensor: The MSE loss.
        """
        eps = torch.randn_like(z_0)  # (B, C, H, W)
        z_t = self.q_t(z_0, t, eps)    # (B, C, H, W)
        eps_theta = self.predict_eps(z_t, t)  # (B, C, H, W)
        loss_fn = nn.MSELoss()
        return loss_fn(eps_theta, eps)   

    @torch.no_grad()
    def p_sample(self, z_t, t):
        """
        Performs one step of the reverse diffusion process.
        
        Args:
            z_t (torch.Tensor): Latent tensor at time t of shape (B, C, H, W).
            t (torch.Tensor): Time step indices of shape (B,).
            
        Returns:
            torch.Tensor: "Denoised" latent tensor z_prev of shape (B, C, H, W).
        """
        # Reshape the scalars to have shape (B, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        
        # For t > 0, use the previous cumulative alpha; otherwise, use 1.0
        alpha_bar_prev = torch.where(
            t > 0,
            self.alpha_bars[(t - 1).clamp(min=0)].view(-1, 1, 1, 1),
            torch.ones_like(alpha_bar_t)
        )
        
        eps_theta = self.predict_eps(z_t, t)  # (B, C, H, W)
        noise = torch.randn_like(z_t)
        sigma_t = (((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t).sqrt()
        
        # Compute the previous latent state
        z_prev = (z_t - (1 - alpha_t).sqrt() * eps_theta) / alpha_t.sqrt()
        # Add noise if t > 0
        mask = (t > 0).view(-1, 1, 1, 1)
        z_prev = z_prev + torch.where(mask, sigma_t * noise, torch.zeros_like(z_t))
        return z_prev

    @torch.no_grad()
    def generate_sample(self, num_samples):
        """
        Generates samples from the model via the reverse diffusion process.
        
        Args:
            num_samples (int): Number of samples to generate.
            
        Returns:
            torch.Tensor: Final generated latent tensor of shape (num_samples, C, H, W).
        """
        # Generate an initial random latent tensor of shape (num_samples, C, H, W)
        z_t = torch.randn(num_samples, *self.latent_shape).to(self.device)
        for t in reversed(range(self.n_steps)):
            # Create a tensor for the current time step of shape (num_samples,)
            t_tensor = torch.full((num_samples,), t, dtype=torch.long).to(self.device)
            z_t = self.p_sample(z_t, t_tensor)
        return z_t

    @torch.no_grad()
    def decode_latent(self, z_0, decoder):
        """
        Decodes the latent tensor into an image using the pretrained decoder.
        
        Args:
            z_0 (torch.Tensor): Latent tensor of shape (B, C, H, W).
            decoder (nn.Module): Pretrained decoder.
            
        Returns:
            torch.Tensor: Reconstructed image.
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

