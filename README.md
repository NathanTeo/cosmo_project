# cosmo_project
Checking how well generative model replicate statistics of simple datasets. In the context of cosmology, simulation maps can be generated using generative models. In this research, we attempt to learn, through increasingly more complex datasets, how well these generative models learn fundamental statistics of the training dataset.

We test on datasets that contain
1. Exactly 10 randomly placed Gaussian blobs of the same size and amplitude at a 32 by 32 resolution
2. Poisson 10 randomly placed Gaussian blobs of the same size and amplitude at a 32 by 32 resolution
3. Poisson 500 randomly placed Gaussian blobs of the same size and amplitude at a 64 by 64 resolution
4. Poisson 500 clustered Gaussian blobs of the same size and amplitude at a 64 by 64 resolution

This repository contains code for training and testing both GANs and diffusion models.

Main Folders
1. code_model:
Contains all code that runs training or testing. All code is imported to scripts and does not need to be run. Model architecture, GAN training module, plotting tools can be found here.
2. code_data:
Code used to create the real datasets for model training.
3. utils:
QOL scripts for moving or clearing files or job submition
4. workspace_tests:
Contains miscellaneous scripts for plotting and testing outside the scope of individual model testing handled by the code_model scripts. Includes tests for comparing two models, checking the power spectrum etc.

Requirements
Install python
Create a virtual env (venv comes preinstalled in later versions of python) and pip install the requirements_cpu.txt or requirements_gpu.txt. 

Creating a run
1. In a selected folder (not the same as the repo), create a folder in the same format as the example run found in the repo, the config file contains the sbatch file for submitting a job to a slurm system and a model config file that contains all parameters for training and testing and code to execute training and testing.
2. Create the real dataset using code_data.blob_realization and transfer the data to the run folder with utils.import_data

Starting a run
On a slurm system, sbatch {run_folder}/{run_name}/config/job.sh submits a job for training. sbatch cosmo_project/utils/test.sh {run_name} and sbatch cosmo_project/utils/generate.sh {run_name} tests and generates samples respectively.

On a local system, run the config file directly using python {run_folder}/{run_name}/config/model_params.py {run_name} {task}. The task variable accepts 'train', 'test' and 'generate' as inputs.
