"""
Author: Nathan Teo

This script checks if the dataModule prepares the data correctly
"""

model_run = 'diffusion_7e'

#############################################################################################
# Import
import sys
import os
import torch
import matplotlib.pyplot as plt

project_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_project'
sys.path.append(project_path)
from code_model.data_modules import *
from code_model.testers.eval_utils import init_param

root_path = f'C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_runs/{model_run}'
sys.path.append(root_path)
from config import model_params 
# model_config = __import__(f'{model_run}.config.model_params')
training_params = model_params.training_params

# Params
batch_size = training_params['batch_size']
num_workers = training_params['num_workers']
data_transforms = init_param(training_params, 'data_transforms')

# Paths
root_path = root_path
data_path = f'{root_path}/data'

# Get data file name
filenames = os.listdir(data_path)
if len(filenames)>1:
        raise Exception('More than 1 data file found')
else:
        data_file_name = filenames[0]

data = BlobDataModule(
        data_file=f'{data_path}/{data_file_name}',
        batch_size=batch_size, num_workers=num_workers,
        transforms=data_transforms
        )

data.setup()

samples = data.samples

print(f'dataset size: {data.num_samples}')
print(f'scaling factor: {data.scaling_factor}')

print()
print('check data')
print('-----------')
print('test 1')
print(f'min {torch.min(samples)} | max {torch.max(samples)}')

print()
print('test 2')
print('plotting...')
for i in range(4):
    plt.imshow(samples[i].squeeze())
    plt.colorbar()
    plt.show()
    plt.close()

print()
print('check complete')
