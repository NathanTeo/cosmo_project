"""
Author: Nathan Teo

This script generates and saves real samples for the GAN.
The samples are saved in a single .npy file along with a single sample plot.
"""

import os

"""Params"""
params = {
    'image_size': 32,
    'seed': 70,
    'blob_size': 5,
    'sample_num': 5000,
    'blob_num': 10,
    'num_distribution': 'uniform',
    'pad': 0,
    'noise': 0
}

root_path = "C:/Users/Idiot/Desktop/Research/OFYP/cosmo"
save_path = f"{root_path}/cosmo_data/{params['blob_num']}_blob"

import sys
sys.path.append(f'{root_path}/cosmo_project')

from code_data.utils import *

# Create save directory
if not os.path.exists(save_path):
    os.makedirs(save_path)

"""Create dataset"""
dataset = blobDataset(**params)
dataset.generate()
dataset.save(save_path)

"""Plot"""
dataset.plot_example(dataset.samples[0], save_path)
if params['num_distribution']!='uniform':
    dataset.plot_distribution(save_path)
