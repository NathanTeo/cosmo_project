# cosmo_project
Generating CMB foregrounds from simulations using GANs

Simulations of CMB foregrounds are an key resource in understanding and removing CMB foregrounds from the CMB map. However, simulations are computationally expensive and time consuming. To produce CMB foreground maps, generative adversarial networks can be trained on an initial batch of 2D CMB foreground simulation maps to generate indepedent 2D maps.

This repository contains code for training a GAN model to generate N gaussian blobs.

Main Folders
1. scripts:
Contains parameters for model training/testing. Run file to start model training/testing.

2. code_model:
Contains all code that runs training or testing. All code is imported to scripts and does not need to be run. Model architecture, GAN training module, plotting tools can be found here.

3. code_data:
Code used to generate real images for model training.

4. data:
Contains real images as npy files. Created when data is generated.

5. checkpoints:
Contains saved models. Created when trianing is initiated.

6. logs:
Contains logged training data --- plots and losses. Created when training is initiated.

7. lightning_logs:
Contains logged training data to be viewed from tensorboard. Created when training is initiated.

8. plots:
Contains output from model testing. Created when model is tested.
