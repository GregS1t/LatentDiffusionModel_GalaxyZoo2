# Latent Diffusion Model applied on the GalaxyZoo2 Dataset

This repository is a sandbox for testing the Latent Diffusion Model (LDM) on the Galaxy Zoo 2 dataset. The project is still in development and may not work properly. Use it freely, but at your own risk!

## About LDM
Latent Diffusion Models (LDMs) introduce an efficient way to train and sample from diffusion-based generative models by operating on a learned latent space rather than directly on high-resolution data (e.g. raw images).
The training is following the steps : 

### Learn a Latent Representation

    First, an autoencoder (often a variational autoencoder, or VAE) is trained to encode high-dimensional data (images) into a lower-dimensional latent space.

    This autoencoder typically consists of an encoder, which compresses the original data into latent codes, and a decoder, which reconstructs the data from these codes.
    
    The goal of the autoencoder is to retain as much semantic information as possible in the latent representation while significantly reducing dimensionality (e.g., compressing a 256×256×3 image into a much smaller latent vector).

### Diffusion in the Latent Space

    Once the autoencoder is trained, its encoder is used to convert training images into latent codes.
    A denoising diffusion process is then trained on these latent codes rather than on the original images. This involves:
        - Forward (Noising) Process: Gradually adding noise to a latent code in a series of discrete steps, ultimately transforming the latent code into random noise.
        - Reverse (Denoising) Process: Training a model (often a U-Net-like network) to iteratively remove the noise at each step and recover the clean latent code.
    
    By working in this compressed space, the diffusion model has fewer parameters to learn and can more effectively capture global structure. It also reduces the computational overhead significantly compared to diffusion on raw pixel data.

### Sampling / Generation

    To generate new samples, the diffusion model starts with random noise in the latent space and runs the reverse diffusion process, step by step, until it obtains a final (denoised) latent code.

    This latent code is then decoded by the autoencoder’s decoder to produce a full-resolution image. In my example, as large as the input images

    Because the model has learned to handle global structure in the latent space, the generated images often exhibit high fidelity and sharp details.


## Purpose

The main purpose of this project is to test the Latent Diffusion Model. The notebook `GZ2_Latent_Diffusion_Models_v1.0.ipynb` contains all the code necessary to train and test the Latent Diffusion Model.

The file `GZ2_LatentDiffusionModel_training_v1.py` requires all files starting with `lib*`.

## Hyperparameters and Other Information

All hyperparameters for training both the autoencoder and the diffusion model are saved in the file `param_GZ2.json`. This file also contains the paths to the data, outputs, and more.

You can also define the frequency to save plots and models during training.

## Applications

This model has only been tested on the Galaxy Zoo 2 dataset. But 



## Galaxy Zoo Dataset

The Galaxy Zoo 2 dataset is a collection of images and classifications of galaxies, created through a citizen science project. It includes detailed morphological classifications for nearly 300,000 galaxies, derived from the [Sloan Digital Sky Survey (SDSS)](https://data.galaxyzoo.org/). Volunteers contribute by answering a series of questions about each galaxy's features, such as its shape, the presence of spiral arms, and the existence of bars. More information about the dataset [Here](https://data.galaxyzoo.org/)


All images are saved in a single directory `your/location/DATA/GALAXYZOO2/`. You need to set this path in the file `param_GZ2.json` in the field `DATALOCATION_DIR`.

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To train the model, run:

```bash
python GZ2_LatentDiffusionModel_training_v1.py
```
Using some flags, you can either choose to train the autoencoder, the latent diffusion model or both.


## Contribution

Contributions are welcome! Feel free to open an issue or submit a pull request.
License
This project is licensed under the MIT License. See the LICENSE file for more details.