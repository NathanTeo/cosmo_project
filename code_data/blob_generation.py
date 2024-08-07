"""
Author: Nathan Teo

This script generates and saves real samples for the GAN.
The samples are saved in a single .npy file along with a single sample plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os
from tqdm.auto import tqdm

"""Params"""
image_size = 40
seed = 70
blob_size = 0
sample_num = 50000
blob_num = 5
distribution = 'uniform'
pad = 0
noise = False
noise_scale = 0.05
root_path = "C:/Users/Idiot/Desktop/Research/OFYP/cosmo_project"
save_path = f"{root_path}/Data/{blob_num}_blob"
file_name = f'bn{blob_num}{distribution[0]}-is{image_size}-bs{blob_size}-sn{sample_num}-sd{seed}-ns{int(noise)}'

import sys
sys.path.append(root_path)

from code_data.utils import *

"Initialize"
# Initialize random seed
random.seed(seed)

# Create save directory
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Padding
if pad == 'auto':
    pad = blob_size 
generation_matrix_size = image_size + pad*2

"""Create blob"""
x, y = np.mgrid[0:generation_matrix_size:1, 0:generation_matrix_size:1]
pos = np.dstack((x, y))

# Create samples
samples = []
sample_counts = []
for i in tqdm(range(sample_num)):
    if distribution=='uniform':
        sample = create_blob_sample(pos, blob_num, blob_size, generation_matrix_size)
    elif distribution=='poisson':
        current_blob_num = np.random.poisson(blob_num)
        sample_counts.append(current_blob_num)
        sample = create_blob_sample(pos, current_blob_num, blob_size, generation_matrix_size)
            
    # Add noise
    if noise==True:
        noise_img = np.random.normal(0, noise_scale, (image_size, image_size))
        sample = np.add(sample, noise_img)
        
    # Unpad
    pad_sample = sample
    if pad != 0:
        sample = sample[pad:-pad,pad:-pad]
    
    # Rescale
    sample = sample/blob_num
    
    # Add sample to list
    samples.append(sample)

print('Completed!')

# Save samples
np.save(f'{save_path}/{file_name}', samples)

"""Plots"""
if distribution=='uniform':
    plt.title(f'num of blobs: {blob_num} | image size: {image_size} | blob size: {blob_size}')
    plt.imshow(sample)
    plt.colorbar()
    plt.savefig(f'{save_path}/sample_{file_name}')
    plt.close()

elif distribution=='poisson':
    plt.title(f'mean num of blobs: {blob_num} | image size: {image_size} | blob size: {blob_size}')
    plt.imshow(sample)
    plt.colorbar()
    plt.savefig(f'{save_path}/sample_{file_name}')
    plt.close()
    
    fig = plt.figure()
    fig.suptitle(f'num of blobs: {blob_num} | image size: {image_size} | blob size: {blob_size}')
    plt.hist(sample_counts)
    plt.savefig(f'{save_path}/distr_{file_name}')
    plt.close()