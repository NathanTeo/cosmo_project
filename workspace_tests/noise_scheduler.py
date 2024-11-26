"""
Author: Nathan Teo

This script makes a plot of the noise scheduler implemented at time steps of set intervals
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os

load_fname = 'ds2.jpg'
save_fname = 'cos_sched_rev'
reverse = True

################################################################################
def cosine_schedule(noise_steps, s=0.008):
    """Prepares cosine scheduler for adding noise"""
    def f(t):
        return torch.cos((t / noise_steps + s) / (1 + s) * 0.5 * torch.pi) ** 2
    x = torch.linspace(0, noise_steps, noise_steps + 1)
    alpha_cumprod = f(x) / f(torch.tensor([0]))
    betas = 1 - alpha_cumprod[1:] / alpha_cumprod[:-1]
    betas = torch.clip(betas, 0.0001, 0.999)
    return betas

def add_noise(alpha_hats, x, t):
    """Adds gaussian noise to images"""
    sqrt_alpha_hats = torch.sqrt(alpha_hats[t])[:,None,None,None]
    sqrt_one_minus_alpha_hats = torch.sqrt(1. - alpha_hats[t])[:,None,None,None]
    noise = torch.randn_like(x)

    return sqrt_alpha_hats*x + sqrt_one_minus_alpha_hats*noise, noise

project_path = "C:/Users/Idiot/Desktop/Research/OFYP/cosmo"
save_path = f'{project_path}/misc_plots/noise_sched'
plot_file_format = 'png'

if not os.path.exists(save_path):
    os.mkdir(save_path)

noise_steps = 1000
s = 0.008
t_step = 100

betas = cosine_schedule(noise_steps, s)
alphas = 1. - betas
alpha_hats = torch.cumprod(alphas, dim=0)

# Load the Image  
img = Image.open(f'C:/Users/Idiot/Desktop/Research/OFYP/misc/images/{load_fname}')  
# Convert the PIL image 
img = torch.tensor(np.array(img))/255

t = np.concatenate([[0], np.arange(99, noise_steps, 100)])

noised_imgs, _ = add_noise(alpha_hats, img, t)

fig, axs = plt.subplots(1,11, figsize=(7,1))

if reverse:
    noised_imgs = noised_imgs.flip(dims=(0,))
    t = t[::-1]
    
for ax, step, img in zip(axs, t, noised_imgs):
    ax.imshow(img)
    ax.set_title(step, fontsize='medium')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_axis_off()

fig.text(.06, .7, 'time\nstep', ha='left')
plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig(f'{save_path}/{save_fname}.{plot_file_format}')