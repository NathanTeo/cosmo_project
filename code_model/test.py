"""
Author: Nathan Teo
"""

import os

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from code_model.data_modules import *
from code_model.GAN_modules import *

def run_testing(training_params, generation_params, fig_num=5):
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
    gen_upsamp = training_params['generator_upsamp_size']
    dis_conv = training_params['discriminator_conv_size']
    dis_lin = training_params['discriminator_linear_size']

    batch_size = training_params['batch_size']
    num_workers = training_params['num_workers']
    max_epochs = training_params['max_epochs']
    avail_gpus = training_params['avail_gpus']

    """Paths"""
    root_path = training_params['root_path']
    data_folder = f'Data\\{blob_num}_blob'
    data_file_name = f'{blob_num}blob_imgsize{image_size}_blobsize{blob_size}_samplenum{sample_num}_seed{generation_seed}.npy'
    chkpt_path = f'checkpoints/{blob_num}_blob'
    chkpt_file_name = '{}-g{}-d{}-bn{}-bs{}-sn1e{}-is{}-ts{}-lr{}-ld{}-gu{}-dc{}-dl{}'.format(
                gan_version,
                gen_version, dis_version,
                blob_num, blob_size, np.log10(sample_num), image_size,
                training_seed, str(lr)[2:],
                latent_dim, gen_upsamp, dis_conv, dis_lin
                )

    training_params['chkpt_file_name'] = chkpt_file_name
    
    save_path = f'{root_path}\\plots\\{chkpt_file_name}\\images'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    """Initialize seed"""
    torch.manual_seed(training_seed)
    
    """Load data"""
    data = np.load(f'{root_path}\\{data_folder}\\{data_file_name}')
    
    """Load model"""
    model = gans[gan_version].load_from_checkpoint(
        f'{root_path}\\{chkpt_path}\\{chkpt_file_name}.ckpt',
        **training_params
        )

    """Testing"""
    img_row_num = 3
    
    # Get images
    real_imgs = data[:img_row_num*img_row_num]
    
    z = torch.randn(img_row_num*img_row_num, latent_dim)
    gen_imgs = model(z).detach().squeeze()
    
    # Plotting images
    fig = plt.figure(figsize=(6,3))
    subfig = fig.subfigures(1, 2, wspace=0.1)
    # Plot real images
    axsL = subfig[0].subplots(img_row_num, img_row_num)
    for i in range(img_row_num):
        for j in range(img_row_num):
            axsL[i, j].imshow(real_imgs[i+j], interpolation='none')
            axsL[i, j].set_xticks([])
            axsL[i, j].set_yticks([])
            axsL[i, j].axis('off')
    subfig[0].subplots_adjust(wspace=.1, hspace=.1)         
    subfig[0].suptitle('Real imgs')
    # Plot generated images
    axsR = subfig[1].subplots(img_row_num, img_row_num)
    for i in range(img_row_num):
        for j in range(img_row_num):
            axsR[i, j].imshow(gen_imgs[i+j], interpolation='none')
            axsR[i, j].set_xticks([])
            axsR[i, j].set_yticks([])
            axsR[i, j].axis('off') 
    subfig[1].subplots_adjust(wspace=.1, hspace=.1) 
    subfig[1].suptitle('Generated imgs')
    # Save
    plt.savefig(f'{save_path}\\{chkpt_file_name}.png')