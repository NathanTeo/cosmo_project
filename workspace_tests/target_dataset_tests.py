"""
Author: Nathan Teo

This script checks statistics of the target dataset
"""

run = "diffusion_8d"

#########################################################################################

# Import
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from utils.load_model import modelLoader

project_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo'
sys.path.append(f'{project_path}/cosmo_project')
from code_model.testers.eval_utils import *
from code_model.testers.plotting_utils import *
from code_data.utils import *

# Paths
save_path = f'{project_path}/misc_plots/inspect_checkpoints/{run}'
chkpt_path = f"C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_runs/{run}/checkpoints"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load samples
loader = modelLoader(run)
print('loading samples...', end='\t')
samples = loader.load_real_samples()
print('complete')
print(samples.shape)

total_fluxes = find_total_fluxes(samples)

batched_total_fluxes = np.array_split(total_fluxes, int(len(samples)/5000))

print(f'batches: {len(batched_total_fluxes)}')
print([len(batch) for batch in batched_total_fluxes])

plt.hist(total_fluxes, histtype='step', color='r')
plt.show()
plt.close()

for batch in batched_total_fluxes:
    plt.hist(batch, histtype='step', color=('r',0.3))    
plt.show()
plt.close()

# for i, batch in enumerate(batched_total_fluxes):
#     plt.hist(batch, histtype='step', color=('r',0.3))
#     plt.title(i)
#     plt.show()
#     plt.close()

min_flux_idx = np.argmin(total_fluxes)
print(min_flux_idx)
print(f'minimum flux: {total_fluxes[min_flux_idx]}')

plt.imshow(samples[min_flux_idx])
plt.show()
plt.close()