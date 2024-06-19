"""
Author: Nathan Teo
"""

import os

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from code_model.data_modules import *
from code_model.GAN_modules import *
from code_model.plotting_utils import *

def run_testing(training_params, generation_params, checkpoint='last', grid_row_num=2, plot_num=25, stack_num=10000):
    """Initialize variables"""
    gan_version = training_params['gan_version']
    gen_version = training_params['generator_version']
    dis_version = training_params['discriminator_version']
    training_seed = training_params['random_seed']
    lr = training_params['lr']

    blob_num = generation_params['blob_num']
    generation_seed = generation_params['seed']
    blob_size = generation_params['blob_size']
    sample_num = generation_params['sample_num']
    image_size = training_params['image_size']

    latent_dim = training_params['latent_dim']
    gen_img_w = training_params['gen_img_w']
    gen_upsamp = training_params['generator_upsamp_size']
    dis_conv = training_params['discriminator_conv_size']
    dis_lin = training_params['discriminator_linear_size']

    batch_size = training_params['batch_size']
    num_workers = training_params['num_workers']
    max_epochs = training_params['max_epochs']
    avail_gpus = training_params['avail_gpus']
    
    model_name = '{}-g{}-d{}-bn{}-bs{}-sn1e{}-is{}-ts{}-lr{}-ld{}-gw{}-gu{}-dc{}-dl{}'.format(
        gan_version,
        gen_version, dis_version,
        blob_num, blob_size, int(np.log10(sample_num)), image_size,
        training_seed, str(lr)[2:],
        latent_dim, gen_img_w, gen_upsamp, dis_conv, dis_lin
        )

    """Paths"""
    root_path = training_params['root_path']
    data_path = f'Data\\{blob_num}_blob'
    data_file_name = f'{blob_num}blob_imgsize{image_size}_blobsize{blob_size}_samplenum{sample_num}_seed{generation_seed}.npy'
    chkpt_path = f'checkpoints\\{blob_num}_blob\\{model_name}'

    log_path = f'logs\\{model_name}'

    training_params['model_name'] = model_name
    
    save_path = f'{root_path}\\plots\\{model_name}\\images'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    """Initialize seed"""
    torch.manual_seed(training_seed)
    
    """Load data"""
    data = np.load(f'{root_path}\\{data_path}\\{data_file_name}')
    
    """Load model"""
    model = gans[gan_version].load_from_checkpoint(
        f'{root_path}\\{chkpt_path}\\{model_name}\\{checkpoint}.ckpt',
        **training_params
        )

    """Plot generated images"""
    # Get images
    real_imgs = data[:grid_row_num*grid_row_num]
    
    for n in tqdm(range(plot_num), desc='Plotting'):
        z = torch.randn(grid_row_num*grid_row_num, latent_dim)
        gen_imgs = model(z).cpu().detach().squeeze()
        
        # Plotting grid of images
        fig = plt.figure(figsize=(6,3))
        subfig = fig.subfigures(1, 2, wspace=0.2)
        
        plot_img_grid(subfig[0], real_imgs, grid_row_num, title='Real Imgs')
        plot_img_grid(subfig[1], gen_imgs, grid_row_num, title='Generated Imgs')
        
        # Save
        plt.savefig(f'{save_path}\\gen-imgs_{model_name}_{n}.png')
        plt.close()
    
        # Plotting marginal sums
        fig = plt.figure(figsize=(4,6))
        subfig = fig.subfigures(1, 2)
                
        plot_marginal_sums(real_imgs, subfig[0], grid_row_num, title='Real')
        plot_marginal_sums(gen_imgs, subfig[1], grid_row_num, title='Generated')
        
        fig.suptitle('Marginal Sums')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path}\\marg-sums_{model_name}_{n}.png')
        plt.close()
    
    """Stacking"""
    # Generate images
    z = torch.randn(stack_num, latent_dim)
    
    print('generating images...')
    gen_imgs = model(z).cpu().detach().squeeze()
    
    # Stack images
    stacked_real_img = stack_imgs(data)
    stacked_gen_img = stack_imgs(gen_imgs)
    
    fig, axs = plt.subplots(1, 2)
    
    plot_stacked_imgs(axs[0], stacked_real_img, title='real')
    plot_stacked_imgs(axs[1], stacked_gen_img, title='generated')
    
    fig.suptitle('Stacked Image')
    
    
    plt.tight_layout()
    plt.savefig(f'{save_path}\\stacked_{model_name}.png')
    plt.close()
    
    """Losses"""
    try:
        losses = np.load(f'{root_path}\\{log_path}\\losses.npz')
        g_losses = losses['g_losses']
        d_losses = losses['d_losses']
        epochs = losses['epochs']
        
        plt.figure(figsize=(6,4))
        plt.plot(epochs, g_losses, label='generator')
        plt.plot(epochs, d_losses, label='discriminator')
        plt.title('Model Loss')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path}\\losses_{model_name}.png')
        plt.close()
        
        # Zoom
        plt.figure(figsize=(6,4))
        plt.plot(epochs, g_losses, label='generator')
        plt.plot(epochs, d_losses, label='discriminator')
        plt.title('Model Loss')
        
        plt.axhline(y=0, color='r', alpha=0.2, linestyle='dashed')
        
        plt.ylim(-1,1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path}\\losses_zm_{model_name}.png')
        plt.close()
    except FileNotFoundError:
        print('losses not found')