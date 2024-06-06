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
blob_num = 1
pad = 0
root_path = "C:\\Users\\Idiot\\Desktop\\Research\\OFYP\\cosmo_project"
save_path = f"{root_path}\\Data\\{blob_num}_blob"
file_name = f'{blob_num}blob_imgsize{image_size}_blobsize{blob_size}_samplenum{sample_num}_seed{seed}'

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
    
    # normalize
    sample = normalize_2d(sample)
    
    # unpad
    pad_sample = sample
    if pad != 0:
        sample = sample[pad:-pad,pad:-pad]
    
    samples.append(sample)

np.save(f'{save_path}\\{file_name}', samples)

# Check
plt.title(f'size: {blob_size}, coords: {mean_coords}')
plt.imshow(sample)
plt.colorbar()
plt.show()

# plt.title(f'size: {blob_size}, coords: {mean_coords}')
# plt.imshow(pad_sample)
# plt.colorbar()
# plt.show()