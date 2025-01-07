# LatentDiffusionModel_GalaxyZoo2

Clearly a sandbox. Not yet working properly.
Free to use but at your own risk !

## Purpose 

Mainly to test the Latent Diffusion Model 
The notebook *GZ2_Latent_Diffusion_Models_v1.0.ipynb * contains all the code necessary to train and test the Latent Diffusion Model

The file GZ2_LatentDiffusionModel_training_v1.py needs all the files starting by _lib*_


## Hyperparameters and other informations

All the hyperparameters for training both the autoencoder and the diffusion model are save in the file *param_GZ2.json*

This param file also contains the paths to the data, the outputs and so on !

You can also define the frequency to save the plots, the models during the training.

## Applications
 
Only tested on GalaxyZoo 2 dataset

## GALAXY ZOO Dataset

All the images are saved in a single directory your/location/DATA/GALAXYZOO2/
You need to set this path in the file *param_GZ2.json* in the field *DATALOCATION_DIR*


