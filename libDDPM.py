
# Import libs
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import imageio
import einops


# Define the model
class DDPModel(nn.Module):
    def __init__(self, network, n_steps, min_beta = 10**-4, max_beta = 0.02,
                 device = None, image_chw = (1, 28, 28)):
        """
            network (_type_):
            n_steps (_type_):
            min_beta (_type_, optional): Defaults to 10**-4 (from Ho et al, 2015)
            max_beta (float, optional): Defaults to 0.02. (from Ho et al, 2015)
            device (_type_, optional):  Defaults to None. (GPU or CPU)
            image_chw (tuple, optional):  Defaults to (1, 28, 28). Dim of image
        """
        super(DDPModel, self).__init__()
        self.network = network.to(device)
        self.device = device
        self.n_steps = n_steps
        self.image_chw = image_chw


        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1. - self.betas
        #self.alpha_bars = torch.cumprod(self.alphas, 0).to(device)   # From equation 4
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) + 1e-6 for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eps=None):

        n, c, h, w = x0.shape                   # Extract dimension of the image

        alf_bar = self.alpha_bars[t].view(-1, 1, 1, 1) # Get the alpha bar for each image in the batch

        if eps is None :
            eps = torch.randn(n, c, h, w).to(self.device) # Generate noise

        #noise = alf_bar.sqrt().reshape(n, 1, 1, 1)*x0 \
        #    + (1 - alf_bar).sqrt().reshape(n, 1, 1, 1)*eps # Ho et al, 2020
        noise = alf_bar.sqrt() * x0 + (1 - alf_bar).sqrt() * eps

        return noise

    def backward(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x, t)

    def compute_loss(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0).to(self.device)  # Generate noise if not provided

        # Forward process: add noise to the original image
        x_noisy = self.forward(x0, t, eps)

        # Backward process: the network predicts the noise (eps_theta)
        eps_theta = self.backward(x_noisy, t)

        # Compute Mean Squared Error (MSE) loss between true noise and predicted noise
        return torch.mean((eps - eps_theta) ** 2)



######## PLOT IMAGES FUNCTION ########

def show_images(images, label=None, classes=None, show=True,
                    suptitle="Images of the first batch", save2file=False,
                    filename="first_batch.png"):
    # Convert images to NumPy if they are PyTorch tensors
    if isinstance(images, torch.Tensor):
        images = images.cpu().detach().numpy()

    # If images are a list, convert each to a NumPy array and check for consistency
    elif isinstance(images, list):
        # Convert each element to a NumPy array if it is a tensor
        converted_images = []
        for img in images:
            if isinstance(img, torch.Tensor):
                converted_images.append(img.cpu().detach().numpy())
            else:
                converted_images.append(img)

        # Check for consistent shapes
        shapes = [img.shape for img in converted_images]
        if len(set(shapes)) > 1:
            raise ValueError(f"Inconsistent image shapes found: {shapes}")

        images = np.array(converted_images)

    # Ensure images are in shape (B, H, W, C) for displaying with matplotlib
    if images.ndim == 4:
        if images.shape[1] == 1:  # Grayscale images (B, 1, H, W)
            images = np.transpose(images, (0, 2, 3, 1))  # Convert to (B, H, W, 1)
        elif images.shape[1] == 3:  # RGB images (B, 3, H, W)
            images = np.transpose(images, (0, 2, 3, 1))  # Convert to (B, H, W, C)
        else:
            raise ValueError(f"Unexpected number of channels: {images.shape[1]}")
    elif images.ndim == 3:  # Single image grayscale (H, W) or (H, W, 1)
        if images.shape[0] == 1:  # Grayscale image
            images = np.squeeze(images, axis=0)  # Convert (1, H, W) to (H, W)

    # Defining number of images to show
    fig = plt.figure(figsize=(10, 10))
    rows = int(len(images) ** 0.5)
    cols = round(len(images) / rows)

    idx = 0
    for i in range(rows):
        for j in range(cols):
            ax = fig.add_subplot(rows, cols, idx + 1)
            if label is not None:
                if idx < len(label) and classes is not None:
                    ax.set_title(classes[label[idx]], fontsize=8)
            ax.axis("off")
            if idx < len(images):
                # Display color image or grayscale depending on the last dimension
                if images[idx].shape[-1] == 3:
                    plt.imshow(images[idx].clip(0, 1))  # Ensure image values are between 0 and 1
                else:
                    plt.imshow(images[idx].squeeze(), cmap="gray")
                idx += 1
    plt.suptitle(suptitle, fontsize=20)
    plt.tight_layout()
    if show:
        plt.show()
    if save2file:
        fig.savefig(filename)
        plt.close(fig)

def show_first_batch(loader, classes=None, save2file=False, show=True,
                     filename="first_batch.png",):
    # Iterate through the loader and plot the first batch

    for images, label, img_type in loader:
        show_images(images, label, classes=classes, show=show,
                    suptitle="Images of the first batch", save2file=save2file,
                     filename=filename)
        break

def show_forward(ddpm, loader, device):
    # Showing forward process
    for batch in loader:
        imgs = batch[0]
        show_images(imgs, "Original images")
        for percent in [0.25, 0.5, 0.75, 1]:
            show_images(ddpm(imgs.to(device), [int(percent*ddpm.n_steps)-1 for _ in range(len(imgs))]),
                        f"DDPM Noisy images {int(percent * 100)} %")
        break

def show_forward_same_image(ddpm, loader, device, n_noise=20):
    # Showing forward process of the same image in the same grid of plots
    # Select one image in the batch
    imgs = next(iter(loader))[0][:1]

    # Create a grid of n_steps plots with 10 columns
    n_col = 10
    n_row = n_noise // n_col
    fig, axs = plt.subplots(n_row, n_col, figsize=(20, 2 * n_row))

    for i in range(n_noise):
        ax = axs[i // n_col, i % n_col]

        # Apply the DDPM forward diffusion process
        noisy_img = ddpm(imgs.to(device), [i])

        # Convert the tensor to numpy and transpose to (H, W, C)
        noisy_img_np = noisy_img.cpu().detach().numpy().squeeze()
        if noisy_img_np.ndim == 2:  # If after squeezing, we have [H, W], add back a dummy channel
            ax.imshow(noisy_img_np, cmap='gray')  # For grayscale images
            #noisy_img_np = noisy_img_np[..., np.newaxis]
        else:
            # If still 3D, make sure it's in the right format [H, W, C]
            noisy_img_np = noisy_img_np.transpose(1, 2, 0).squeeze()
            ax.imshow(noisy_img_np.clip(0, 1))
            #noisy_img_np = noisy_img_np.transpose(1, 2, 0)

        # Clip values to be between 0 and 1 for visualization
        ax.imshow(noisy_img_np.clip(0, 1))

        ax.axis("off")
        ax.set_title(f"Step {i}")
    plt.suptitle(
            "DDPM Forward process of a Single Image \n"
            + f"noise level={n_noise} | $\\beta_{{min}}$={ddpm.betas[0]:.2e}"
            + f"| $\\beta_{{max}}$={ddpm.betas[-1]:.2e}",
            fontsize=20)

    plt.show()

def generate_new_images(ddpm, n_samples=16, device=None, frames_per_gif=100,
                        gif_name="sampling.gif", c=1, h=28, w=28):
    import imageio
    import einops

    """

    Given a DDPM model, a number of samples to be generated and a device,
    returns some newly generated samples

    """
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn(n_samples, c, h, w).to(device)

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                # Option 1: sigma_t squared = beta_t
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                # Option 2: sigma_t squared = beta_tilda_t
                #prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                #beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                #sigma_t = beta_tilda_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z

            # Adding frames to the GIF
            if idx in frame_idxs or t == 0:
                # Putting digits in range [0, 255]
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)

                # Rendering frame
                frames.append(frame)

    # Storing the gif
    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            rgb_frame = np.repeat(frame, 3, axis=2)
            writer.append_data(rgb_frame)

            # Showing the last frame for a longer time
            if idx == len(frames) - 1:
                last_rgb_frame = np.repeat(frames[-1], 3, axis=2)
                for _ in range(frames_per_gif // 3):
                    writer.append_data(last_rgb_frame)
    return x
