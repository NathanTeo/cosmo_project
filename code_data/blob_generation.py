import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import random
import os
from tqdm.auto import tqdm

"""Params"""
image_size = 28
seed = 70
blob_size = 5
sample_num = 10000
blob_num = 10
pad = 0
noise = True
noise_range = (-0.1, 0.1)
root_path = "C:/Users/Idiot/Desktop/Research/OFYP/cosmo_project"
save_path = f"{root_path}/Data/{blob_num}_blob"
file_name = f'bn{blob_num}-is{image_size}-bs{blob_size}-sn{sample_num}-sd{seed}-ns{noise}'

"Initialize"
random.seed(seed)

if not os.path.exists(save_path):
    os.makedirs(save_path)

if pad == 'auto':
    pad = blob_size
    
generation_matrix_size = image_size + pad*2

"""Create blob"""
def normalize_2d(matrix):
    return (matrix-np.min(matrix))/(np.max(matrix)-np.min(matrix)) 

x, y = np.mgrid[0:generation_matrix_size:1, 0:generation_matrix_size:1]
pos = np.dstack((x, y))

samples = []
for i in tqdm(range(sample_num)):
    for j in range(blob_num):
        mean_coords = [random.randint(0, generation_matrix_size-1), random.randint(0, generation_matrix_size-1)]
        if j==0:
            sample = multivariate_normal(mean_coords, [[blob_size, 0], [0, blob_size]]).pdf(pos)
        if j!=0:
            sample_next = multivariate_normal(mean_coords, [[blob_size, 0], [0, blob_size]]).pdf(pos)
            sample = np.add(sample, sample_next)
    
    # Normalize
    sample = normalize_2d(sample)
    
    # Add nosie
    if noise==True:
        noise_img = np.random.uniform(*noise_range, (image_size, image_size))
        sample = sample + noise_img
        
    # Unpad
    pad_sample = sample
    if pad != 0:
        sample = sample[pad:-pad,pad:-pad]
    
    samples.append(sample)

np.save(f'{save_path}/{file_name}', samples)

# Plot to check
plt.title(f'num of blobs: {blob_num} | image size: {image_size} | blob size: {blob_size}')
plt.imshow(sample)
plt.colorbar()
plt.savefig(f'{save_path}/{file_name}')