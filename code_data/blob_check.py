"""
Author: Nathan Teo

"""

import numpy as np
import matplotlib.pyplot as plt

blob_num = 1
generation_seed = 70
sample_num = 1000
image_size = 28
blob_size = 5

root_path = "C:\\Users\\Idiot\\Desktop\\Research\\OFYP\\cosmo_project"
data_dir=f'{root_path}\\Data\\{blob_num}_blob'
file_name = f'{blob_num}blob_imgsize{image_size}_blobsize{blob_size}_samplenum{sample_num}_seed{generation_seed}'

samples = np.load(f'{data_dir}\\{file_name}.npy')
    
for i, sample in enumerate(samples):
    if i == 0:
        sample_sum = sample
    else:
        sample_sum = np.add(sample_sum, sample)

plt.imshow(sample_sum)
plt.show()

# sample_sum = sample_sum[5:-5,5:-5]

# plt.imshow(sample_sum)
# plt.show()