
# Purpose : Training of a Latent Diffusion Model (LDM) on the Galaxy Zoo 2 dataset.

# The LDM is a generative model that learns to generate images by diffusing a latent space.
# The model is trained on a subset of spiral galaxies from the Galaxy Zoo 2 dataset. 
# The training is done in two steps:
# 1. Training an autoencoder to learn a latent representation of the images.
# 2. Training the Latent Diffusion Model (LDM) using the learned latent representation.
# The LDM is trained to generate samples from the latent space and decode them into images.

# Author: Grégory Sainton
# Institution: Observatoire de Paris - PSL University
# Date: 2025-01-06


import os, sys
import datetime
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import PIL
from tqdm import tqdm

# Scikit-learn
from sklearn.manifold import TSNE

# PyTorch
import torch
from torchvision.transforms import (Compose, ToTensor,
                                    Lambda, Resize, CenterCrop,
                                    RandomHorizontalFlip,
                                    RandomRotation, Normalize)
from packaging import version

torch_version = version.parse(torch.__version__)

if version.parse(torch.__version__) >= version.parse("2.5.0"):
    from torch.amp import autocast, GradScaler
else:
    from torch.cuda.amp import autocast, GradScaler

from torch.utils.data import random_split
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
import torch.nn.functional as F

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Custom Libraries
from libGalaxyZooDataset import *
from libDDPM import *
from libGPU_torch_utils import *
from libAutoEncoder4LDM import *
from libNoisePredictors import *
from libLDM import *

## Side functions   

# Custom loss function combining MSE and SSIM
# SSMI = Structural Similarity Index Measure 
# (https://en.wikipedia.org/wiki/Structural_similarity)

def loss_with_ssmi(reconstructed, original):
    mse_loss = F.mse_loss(reconstructed, original)
    ssim_loss = 1 - ssim(reconstructed, original,
                         data_range=1, size_average=True)  # SSIM loss (inverted)
    return mse_loss + 0.1 * ssim_loss  # Weight SSIM loss by a factor, adjust as needed

def version_aware_autocast(device):
    """
    Create an autocast context manager based on the PyTorch version.
    Depending on the version, the `device_type` argument may or 
    may not be supported.
    
    Args:
        device (str): Device to use (cpu or cuda).

    Returns:
        autocast: Context manager for mixed precision training.
    """
    if torch_version >= version.parse("2.5.0"):
        return autocast(device_type="cuda" if device == "cuda" else "cpu")
    else:
        return autocast()

def visualize_latent_space(latent_vectors, labels, epoch, output_dir, strdate):
    """
    Visualizes the latent space using t-SNE.

    Args:
        latent_vectors (np.array): Latent vectors from the encoder.
        labels (list): Corresponding labels for each vector.
        epoch (int): Current epoch number (for saving).
        output_dir (str): Directory to save the plot.
        strdate (str): Timestamp string for filenames.
    """
    
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(latent_vectors)

    df = pd.DataFrame({
        'x': reduced_vectors[:, 0],
        'y': reduced_vectors[:, 1],
        'label': labels
    })

    # Plot using seaborn
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='x', y='y', hue='label', palette='viridis', data=df, alpha=0.8)
    plt.title(f'Latent Space Visualization at Epoch {epoch}')
    plt.legend(title='Labels', loc='best')
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{strdate}_latent_space_epoch_{epoch:03}.png"))
    plt.close()

# Define the path to the parameters file
parameters_file = 'param_GZ2.json'
SEED = 42


## Parameters for training ###
autoencoder_training = True
diffusion_model_training = True


# ------------------------------------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    if os.path.exists(parameters_file):
        # Load the parameters from the JSON file
        with open(parameters_file, 'r') as json_file:
            parameters = json.load(json_file)

        # Now you can access the parameters using dictionary-style access
        no_train = parameters['no_train']
        verbose = parameters['verbose']
        test_model = parameters['test_model']
        batch_size = parameters['batch_size']
        latent_dim = parameters['latent_dim']
        n_epochs = parameters['n_epochs']
        n_epochs_diffusion = parameters['n_epochs_diffusion']
        lr = parameters['lr']
        n_steps = parameters['n_steps']
        min_beta = parameters['min_beta']
        max_beta = parameters['max_beta']
        nb_gal4training = parameters['nb_gal4training']
        nb_gal2plot = parameters['nb_gal2plot']
        plot_subset = parameters['plot_subset']
        output_model = parameters['output_model']
        ref_column_file = parameters['ref_column_file']
        mapping_file = parameters['mapping_file']
        DATALOCATION_DIR = parameters['DATALOCATION_DIR']
        DDPM_dir = parameters['DDPM_dir']
        output_dir = parameters['output_dir']
        plot_reconstruction = parameters['plot_reconstruction']        
        save_freq = parameters['save_freq']
        val_freq = parameters['val_freq'] 

        # Carbon followup
        carbon_estimation = parameters['carbon_estimation']
        carbon_log_dir = parameters['carbon_log_dir']
        carbon_log_file = parameters['carbon_log_file']
        training_log_file = parameters['training_log_file']

    else:
        print(f"File {parameters_file} does not exist !!!")
        sys.exit(1)

    ## DATASET

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    mydevice = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = setup_device(mydevice)

    if not os.path.exists('tensorboard_logs'):
        os.makedirs('tensorboard_logs')
    tensorboard_log_dir = os.path.join(DDPM_dir, 'tensorboard_logs_LDM')
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    ### Load the data ###
    print("\n"+"-"*50)
    print("Read the file with all the galaxies...")
    print("-"*50+"\n")

    # Check if the file exists and is not empty
    if not os.path.exists(os.path.join(DATALOCATION_DIR, ref_column_file)):
        print(f"File {ref_column_file} does not exist in {DATALOCATION_DIR}")
        sys.exit(1)

    if os.path.getsize(os.path.join(DATALOCATION_DIR, ref_column_file)) <= 0:
        print(f"File {ref_column_file} is empty")
        sys.exit(1)

    df_galaxy_zoo_header = pd.read_csv(os.path.join(DATALOCATION_DIR,
                                                    ref_column_file), delimiter=',',
                                                    header=0)
    
    morpho = 'S'
    morpho_gal = {'S': 'spiral', 'E': 'elliptical', 'A': 'erregular'}

    # Display the spiral galaxies (gz2_class starting with 'S')
    print("Selection of spiral galaxies from the dataset:")
    df_sub_morpho = df_galaxy_zoo_header[df_galaxy_zoo_header['gz2_class'].str.startswith(morpho)]

    # Load the mapping file
    df_mapping = pd.read_csv(os.path.join(DATALOCATION_DIR, mapping_file),
                                delimiter=',',
                                header=0)

    # Match where dr7objid in df_spiral = objid in df_mapping to add asset_id to the df_spiral
    df_sub_morpho = pd.merge(df_sub_morpho, df_mapping, left_on='dr7objid', right_on='objid')

    print(f"Number of {morpho_gal[morpho]} galaxies in the dataset: {df_sub_morpho.shape[0]}")
    if nb_gal4training < 0 or nb_gal4training > df_sub_morpho.shape[0]:
        nb_gal2training = df_sub_morpho.shape[0]
    else:   
        nb_gal4training = min(nb_gal4training, df_sub_morpho.shape[0])

    print(f"Number of {morpho_gal[morpho]} galaxies for training: {nb_gal4training}")
    # Create a subset of galaxies to plot if needed
    df_sub_morpho_subset = df_sub_morpho.sample(n=nb_gal4training, random_state=SEED)

    # Exit if the subset is empty
    if df_sub_morpho_subset.empty:
        print("The subset of galaxies is empty")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # PREPARE DATA LOADERS
    # --------------------------------------------------------------------------

    transform = Compose([
        Resize((64, 64)),
        #CenterCrop(40),
        RandomHorizontalFlip(),
        RandomRotation(20),
        ToTensor(),
        Lambda(lambda x: x.float())
    ])

    transform_tanh = Compose([
        Resize((64, 64)),
        #CenterCrop(40),
        RandomHorizontalFlip(),
        RandomRotation(20),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])

    # Define the dataset and the dataloader
    dataset = GalaxyZooDataset(df_sub_morpho_subset, DATALOCATION_DIR,
                            transform=transform)

    dataset_size = len(dataset)
    validation_split = 0.2  # 20% for validation
    validation_size = int(validation_split * dataset_size)
    training_size = dataset_size - validation_size

    train_dataset, val_dataset = random_split(dataset,
                                            [training_size, validation_size])

    # Custom collate function to handle the variable image sizes or empty tensors
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=custom_collate)

    # Free up memory
    del df_galaxy_zoo_header, df_sub_morpho, df_mapping, df_sub_morpho_subset
    del dataset
    del train_dataset
    del val_dataset

    # ---------------------------------------------------------------------------
    if verbose:
        print("Show the first batch of images...")
        show_first_batch(train_loader, save2file=True,
                            filename='output/first_batch_GZ2.png')
        plt.show()

    # --------------------------------------------------------------------------
    # TRAINING OF THE AUTOENCODER ITSELF
    # --------------------------------------------------------------------------

    if autoencoder_training:
        print("\n"+"-"*50)
        print("Training the autoencoder...")
        print("-"*50+"\n")
        autoencoder = Autoencoder(latent_dim=latent_dim).to(device)

        optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=1e-4)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            patience=5, factor=0.5)


        # The following code block is used to handle mixed precision training
        # It could be put into a function to avoid code duplication TODO
        if torch_version < version.parse("2.5.0"):
            scaler = torch.amp.GradScaler(init_scale=65536.0, growth_interval=2000)
        else:
            scaler = GradScaler()

        num_epochs = n_epochs
        best_val_loss = float('inf')
        patience = 10
        trigger_times = 0

        # For plotting
        train_losses = []
        val_losses = []

        # To visualize latent space
        latent_vectors = []
        labels = []
        reconstruction_errors = []

        datenow = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        strdate = f"{datenow}_autoencoder_training"

        for epoch in range(num_epochs):
            autoencoder.train()
            running_loss = 0.0
            step_bar = tqdm(train_loader, leave=False,
                                desc=f"Epoch {epoch + 1}/{num_epochs}",
                                colour="#005500")
            for _, batch in enumerate(step_bar):
                if batch is None:  # Skip batches with missing files
                    continue

                x0 = batch[0].to(device)   # Input images
                batch_labels = batch[2]
                #with autocast(device_type="cuda" if device == "cuda" else "cpu"):
                with version_aware_autocast(device):
                    encoded = autoencoder.encoder(x0)
                    decoded = autoencoder.decoder(encoded)
                    loss = loss_with_ssmi(decoded, x0)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
            
            avg_train_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

            # Estimate validation loss every `val_freq` epochs
            if (epoch + 1) % val_freq != 0: 
                autoencoder.eval()
                val_loss = 0.0

                latent_vectors_list = []
                labels_list = []

                with torch.no_grad():
                    for _, batch in enumerate(val_loader):
                        if batch is None:
                            continue

                        x0 = batch[0].to(device)
                        batch_labels = batch[2]

                    #with autocast(device_type="cuda" if device == "cuda" else "cpu"):
                    with version_aware_autocast(device):
                        encoded = autoencoder.encoder(x0)
                        decoded = autoencoder.decoder(encoded)
                        loss = loss_with_ssmi(decoded, x0)

                        val_loss += loss.item()

                        # Compute reconstruction error per sample
                        errors = torch.mean((x0 - decoded) ** 2, dim=[1, 2, 3])  # MSE per sample
                        reconstruction_errors.extend(errors.detach().cpu().numpy())


                        # Collect latent vectors and labels
                        latent_vectors_list.append(encoded.detach().cpu().numpy())
                        labels_list.extend(batch_labels)

                avg_val_loss = val_loss / len(val_loader)
                print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

            # Plot reconstruction error distribution
            if plot_reconstruction:
                
                plt.figure(figsize=(8, 6))
                plt.hist(reconstruction_errors, bins=50, color='skyblue', edgecolor='black')
                plt.title(f'Histogram of Reconstruction Errors at Epoch {epoch}')
                plt.xlabel(f'Reconstruction Error at Epoch {epoch}')
                plt.ylabel('Frequency')
                plt.grid(True)
                plt.savefig(os.path.join(output_dir,f'{strdate}_reconstruction_errors_{epoch:03}.png'))
                #plt.show()

                # Visualize original and reconstructed images
                autoencoder.eval()
                with torch.no_grad():
                    sample_batch = next(iter(val_loader))[0].to(device)
                    with autocast(device_type="cuda" if device == "cuda" else "cpu"):
                        reconstructed_batch = autoencoder(sample_batch)

                sample_batch = sample_batch.cpu()
                reconstructed_batch = reconstructed_batch.cpu()

                n = 6  # Number of images to display
                plt.figure(figsize=(12, 4))
                for i in range(n):
                    # Original images
                    ax = plt.subplot(2, n, i + 1)
                    plt.imshow(np.transpose(sample_batch[i].numpy(), (1, 2, 0)))
                    plt.title("Original")
                    plt.axis('off')

                    # Reconstructed images
                    ax = plt.subplot(2, n, i + 1 + n)
                    plt.imshow(np.transpose(reconstructed_batch[i].numpy(), (1, 2, 0)))
                    plt.title("Reconstructed")
                    plt.suptitle(f"Original and Reconstructed Images at Epoch {epoch}")
                    plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir,
                                        f"{strdate}_reconstructed_images_{epoch:03}.png"))
                #plt.show()

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)

            # Early Stopping Check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(autoencoder.state_dict(),
                            os.path.join(output_dir,
                                        f"{strdate}_best_autoencoder.pth"))
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print('Early stopping!')
                    break

            # Save the model every save_freq epochs
            if (epoch + 1) % save_freq == 0:
                torch.save(autoencoder.state_dict(),
                            os.path.join(output_dir,
                                        f"{strdate}_autoencoder_epoch_{epoch+1}.pth"))
                torch.cuda.empty_cache()

            # Concatenate all latent vectors
            latent_vectors = np.concatenate(latent_vectors_list, axis=0)
            labels = np.array(labels_list)
            
            subset_size = 1000
            if len(latent_vectors) > subset_size:
                indices = np.random.choice(len(latent_vectors), size=subset_size,
                                        replace=False)
                latent_vectors_subset = latent_vectors[indices]
                labels_subset = labels[indices]
            else:
                latent_vectors_subset = latent_vectors
                labels_subset = labels

            # Visualize latent space
            visualize_latent_space(latent_vectors_subset, labels_subset, epoch, 
                                output_dir, strdate)
            
        if verbose:
            # After training
            print("Training complete! Plotting training and validation loss...")
            plt.figure(figsize=(8, 6))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.yscale('log')
            plt.legend()
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir,
                                        f"{strdate}_training_validation_loss.png"))
            plt.show()
            #plt.close()
        print("AUTOENCODER: Training complete!")

    # --------------------------------------------------------------------------
    # TRAINING OF THE LATENT DIFFUSION MODEL
    # --------------------------------------------------------------------------
    if diffusion_model_training:
        print("\n"+"-"*50)
        print("Training the Latent Diffusion Model...")
        print("-"*50+"\n")
        #best_ae_path = os.path.join(output_dir, "20241128_084322_autoencoder_training_best_autoencoder.pth")
        best_ae_path = os.path.join(output_dir,
                                        f"{strdate}_best_autoencoder.pth")
        # Chargement de l'autoencodeur pré-entraîné
        autoencoder = Autoencoder(latent_dim=latent_dim).to(device)
        autoencoder.load_state_dict(torch.load(best_ae_path, weights_only=True))
        autoencoder.eval()  # Mode évaluation pour éviter toute modification des poids
        print(f"Best autoencoder reloaded from file {best_ae_path}.")
        
        # Version with simple feedforward neural network
        noise_predictor = NoisePredictor(latent_dim=latent_dim).to(device)
        diffusion_model_mlp = LatentDiffusionModel_MLP(
            network=noise_predictor,
            n_steps=n_steps,
            min_beta=min_beta,
            max_beta=max_beta,
            latent_dim=latent_dim,
            device=device
        )

        # Version with U-net architecture
        latent_channels = latent_dim
        
        unet_noise_predictor = UNetNoisePredictor(
        in_channels=latent_channels,  # nombre de canaux en entrée
        base_channels=64,             # vous pouvez ajuster
        time_emb_dim=256,             # dimension de l'embedding du temps
        out_channels=latent_channels  # pour que la sortie ait le même nombre de canaux
        ).to(device)
        
        diffusion_model_w_unet = LatentDiffusionModel_UNET(
            network=unet_noise_predictor,
            n_steps=n_steps,
            min_beta=min_beta,
            max_beta=max_beta,
            latent_dim=latent_dim,
            device=device
        )

        diffusion_model = diffusion_model_w_unet

        optimizer_diffusion = torch.optim.Adam(diffusion_model.network.parameters(), lr=lr)
        scheduler_diffusion = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_diffusion, mode='min',
                                                            patience=5, factor=0.5)

        if torch_version < version.parse("2.5.0"):
            scaler = torch.amp.GradScaler(init_scale=65536.0, growth_interval=2000)
        else:
            scaler = GradScaler()

        # Diffusion model training
        best_val_loss = float('inf')

        for epoch in range(n_epochs_diffusion):
            diffusion_model.train()
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs_diffusion}",
                                            leave=False)
            
            for images in progress_bar:
                x0 = images[0].to(device)

                with torch.no_grad():  # L'autoencodeur n'est pas modifié
                    z_0 = autoencoder.encoder(x0)

                t = torch.randint(0, n_steps, (z_0.size(0),), device=device)
                with version_aware_autocast(device):
                    loss = diffusion_model.compute_loss(z_0, t)

                optimizer_diffusion.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer_diffusion)
                scaler.update()

                running_loss += loss.item() * z_0.size(0)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            epoch_loss = running_loss / len(train_loader.dataset)
            tqdm.write(f"Epoch [{epoch+1}/{n_epochs_diffusion}], Training Loss: {epoch_loss:.4f}")
            progress_bar.set_postfix({"Train loss": f"{epoch_loss:.4f}"})

            scheduler_diffusion.step(epoch_loss)

            diffusion_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images in val_loader:
                    x0 = images[0].to(device)
                    z_0 = autoencoder.encoder(x0)
                    t = torch.randint(0, n_steps, (z_0.size(0),), device=device)

                    with autocast(device_type="cuda" if device == "cuda" else "cpu"):
                        loss = diffusion_model.compute_loss(z_0, t)
                    val_loss += loss.item() * z_0.size(0)

            val_loss /= len(val_loader.dataset)
            tqdm.write(f"Epoch [{epoch+1}/{n_epochs_diffusion}], Validation Loss: {val_loss:.4f}")
            progress_bar.set_postfix({"Val loss": f"{val_loss:.4f}"})

            # Log the losses to TensorBoard
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)

            # Save the model if the validation loss is the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(diffusion_model.state_dict(), os.path.join(output_dir, "LDM_UNET_diffusion_model_best.pth"))
                print("Best model saved for validation loss: {:.4f}".format(val_loss) + " at epoch " + str(epoch + 1))

    print("LATENT DIFFUSION MODEL: Training complete!")