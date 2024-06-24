"""
Author: Nathan Teo

This script plots a stacked image of the real blob samples.
It checks if the generation is done well and if the sample size is sufficiently large.
"""

import numpy as np
import matplotlib.pyplot as plt

# Params
blob_num = 1
generation_seed = 70
sample_num = 10000
image_size = 28
blob_size = 5
noise = False

root_path = "C:/Users/Idiot/Desktop/Research/OFYP/cosmo_project"
data_dir=f'{root_path}/Data/{blob_num}_blob'
file_name = f'bn{blob_num}-is{image_size}-bs{blob_size}-sn{sample_num}-sd{generation_seed}-ns{noise}'

# Load samples
samples = np.load(f'{data_dir}/{file_name}.npy')

# Stack samples
for i, sample in enumerate(samples):
    if i == 0:
        sample_sum = sample
    else:
        sample_sum = np.add(sample_sum, sample)

plt.imshow(sample_sum)
plt.show()