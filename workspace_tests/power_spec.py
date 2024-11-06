import numpy as np
import matplotlib.pyplot as plt
import os

project_path = "C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_project"
import sys
sys.path.append(project_path)
from code_data.utils import blobDataset
from code_model.testers.eval_utils import power_spectrum

root_path = "C:/Users/Idiot/Desktop/Research/OFYP/cosmo/misc_plots"

name_1 = '10k'
name_2 = '1k'
folder_name = 'big_image'

params_1 = {
    'image_size': 256,
    'seed': 50,
    'blob_size': 5,
    'sample_num': 10,
    'blob_amplitude': 1,
    'amplitude_distribution': 'delta',
    'blob_num': 10000,
    'num_distribution': 'delta',
    'clustering': None,
    'pad': 0,
    'noise': 0
}

params_2 = {
    'image_size': 256,
    'seed': 71,
    'blob_size': 5,
    'sample_num': 10,
    'blob_amplitude': 1,
    'amplitude_distribution': 'delta',
    'blob_num': 1000,
    'num_distribution': 'delta',
    'clustering': None,
    'pad': 0,
    'noise': 0
}

def detrend(sample, method='constant'):
    if method=='constant':
        sample = sample-np.mean(sample)
    return sample
        

save_path = f'{root_path}/power_spec/{folder_name}'
if not os.path.exists(save_path):
    os.mkdir(save_path)

dataset = blobDataset(**params_1)
dataset.realize(mode='single', batch=5, temp_root_path=save_path)
samples_1 = dataset.samples

dataset = blobDataset(**params_2)
dataset.realize(mode='single', batch=5, temp_root_path=save_path)
samples_2 = dataset.samples


"""Plot"""
print('plotting...', end='    ')
sample_1 = samples_1[0]
sample_2 = samples_2[0]

fig, axs = plt.subplots(1, 2)
axs[0].imshow(sample_1)
axs[1].imshow(sample_2)

axs[0].set_title(name_1)
axs[1].set_title(name_2)

plt.savefig(f'{save_path}/samples')
plt.close()

sample_1_detrend = detrend(sample_1)
sample_2_detrend = detrend(sample_2)

fig, axs = plt.subplots(1, 2, figsize=(5,3))

bnd_1 = np.max([-np.min(sample_1_detrend), np.max(sample_1_detrend)])
bnd_2 = np.max([-np.min(sample_2_detrend), np.max(sample_2_detrend)])
bnd = np.max([bnd_1, bnd_2])

im = axs[0].imshow(sample_1_detrend, vmin=-bnd, vmax=bnd, cmap='RdBu')
axs[1].imshow(sample_2_detrend, vmin=-bnd, vmax=bnd, cmap='RdBu')

axs[0].set_title(name_1)
axs[1].set_title(name_2)

fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.2, 0.03, 0.6])
fig.colorbar(im, cax=cax, orientation='vertical')

plt.savefig(f'{save_path}/detrend_samples')
plt.close()

fig, ax = plt.subplots()

cls_2 = []   
for sample in samples_2:
    bins, cl = power_spectrum(sample, sample, 300, 15000, taper=True)
    ax.plot(bins, cl, color=('blue', 0.2), label=name_2)
    cls_2.append(cl)

cls_1 = []
for sample in samples_1:
    bins, cl = power_spectrum(sample, sample, 300, 15000, taper=True)
    ax.plot(bins, cl, linestyle='dashed', color=('red', 0.2), label=name_1)
    cls_1.append(cl)

plt.yscale('log')
plt.xscale('log')

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    
plt.ylabel(r'$C_l$')
plt.xlabel(r'$l$')

legend_without_duplicate_labels(ax)
plt.savefig(f'{save_path}/{name_1}-vs-{name_2}')

"""
plt.xlim((0,3000))
plt.ylim((1e-12,1e-6))
plt.savefig(f'{save_path}/{name_1}-vs-{name_2}_large-scales')

plt.xlim((3000,5000))
plt.ylim((1e-14,1e-7))
plt.savefig(f'{save_path}/{name_1}-vs-{name_2}_medium-scales')

plt.xlim((5000,10000))
plt.ylim((1e-18,1e-12))
plt.savefig(f'{save_path}/{name_1}-vs-{name_2}_blob-scales')
"""
plt.close()

cl_2_mean = np.mean(cls_2, axis=0)
cl_1_mean = np.mean(cls_1, axis=0)

fig, axs = plt.subplots(2, sharex=True)

fig.suptitle(f'mean Cls of {params_1["sample_num"]} samples')
axs[0].plot(bins, cl_2_mean, color='blue', label=name_2)
axs[0].plot(bins, cl_1_mean, color='red', linestyle='dashed', label=name_1)
axs[0].set_yscale('log')
axs[0].set_xscale('log')
axs[0].set_ylabel(r'$C_l$')
axs[0].legend()

axs[1].plot(bins, cl_1_mean-cl_2_mean, color='red', label=f'{name_1}-{name_2}')
axs[1].plot(bins, cl_2_mean-cl_1_mean, color='blue', label=f'{name_2}-{name_1}')
axs[1].plot(bins, np.repeat(0, len(bins)), color='blue')
axs[1].set_title('difference')
axs[1].set_yscale('log')
axs[1].set_xscale('log')
axs[1].set_ylabel(r'$\Delta C_l$')
axs[1].set_xlabel(r'$l$')
axs[1].legend()


plt.tight_layout()
plt.savefig(f'{save_path}/{name_1}-vs-{name_2}_mean')
plt.close()


fig, axs = plt.subplots(2, sharex=True)

fig.suptitle(f'mean Cls of {params_1["sample_num"]} samples')
axs[0].plot(bins, cl_2_mean, color='blue', label=name_2)
axs[0].plot(bins, cl_1_mean, color='red', linestyle='dashed', label=name_1)
axs[0].set_xscale('log')
axs[0].set_ylabel(r'$C_l$')
axs[0].legend()

axs[1].plot(bins, cl_1_mean-cl_2_mean, color='red', label=f'{name_1}-{name_2}')
axs[1].plot(bins, np.repeat(0, len(bins)), color='blue')
axs[1].set_title('difference')
axs[1].set_xscale('log')
axs[1].set_ylabel(r'$\Delta C_l$')
axs[1].set_xlabel(r'$l$')
axs[1].legend()

plt.tight_layout()
plt.savefig(f'{save_path}/{name_1}-vs-{name_2}_mean_semilog')
plt.close()
print('complete')