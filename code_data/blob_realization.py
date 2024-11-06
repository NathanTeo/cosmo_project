"""
Author: Nathan Teo

This script generates and saves real samples for the GAN.
The samples are saved in a single .npy file along with a single sample plot.
"""

import os

"""Params"""
params = {
    'image_size': 64,
    'seed': 70,
    'blob_size': 5,
    'blob_amplitude': 1,
    'amplitude_distribution': 'delta',
    'sample_num': 100000,
    'blob_num': 500,
    'num_distribution': 'poisson',
    'clustering': None,
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

if __name__=="__main__":
    """Create dataset"""
    dataset = blobDataset(**params)
    dataset.realize(mode='multi', temp_root_path=save_path)
    dataset.save(save_path)

    """Plot"""
    dataset.plot_example(dataset.samples[0], save_path)
    if params['num_distribution']!='delta':
        dataset.plot_count_distribution(save_path)
        
    if params['amplitude_distribution']!='delta':
        dataset.plot_amplitude_distribution(save_path)
