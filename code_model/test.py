"""
Author: Nathan Teo
"""

import os

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

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
    chkpt_file_name = '{}-g{}-d{}-bn{}-bs{}-sn{}-is{}-ts{}-lr{}-ld{}-gu{}-dc{}-dl{}'.format(
                gan_version,
                gen_version, dis_version,
                blob_num, blob_size, sample_num, image_size,
                training_seed, str(lr)[2:],
                latent_dim, gen_upsamp, dis_conv, dis_lin
                )

    training_params['chkpt_file_name'] = chkpt_file_name

    """Initialize seed"""
    torch.manual_seed(training_seed)
    
    """Load model"""
    model = gans[gan_version].load_from_checkpoint(
        f'{root_path}\\{chkpt_path}\\{chkpt_file_name}.ckpt',
        **training_params
        )

    """Plot"""
    for _ in range(fig_num):
        z = torch.randn(3*3, latent_dim)
        gen_imgs = model(z)

        fig = plt.figure()
        for i in range(gen_imgs.size(0)):
            plt.subplot(3, 3, i+1)
            plt.tight_layout()
            plt.imshow(gen_imgs.detach()[i, 0, :, :], interpolation='none')
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
        plt.show()