"""
Author: Nathan Teo

This script contains runs testing on the model:
Plot real and generated sample images.
Plot marginal sums of real and generated sample images.
Plot stacked image
Plot loss with epoch (if available). 
"""

import os

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from code_model.data_modules import *
from code_model.GAN_modules import *
from code_model.plotting_utils import *

def run_testing(training_params, generation_params, testing_params):
    """Initialize variables"""
    gan_version = training_params['gan_version']
    gen_version = training_params['generator_version']
    dis_version = training_params['discriminator_version']
    training_seed = training_params['random_seed']
    lr = training_params['lr']

    blob_num = generation_params['blob_num']
    generation_seed = generation_params['seed']
    blob_size = generation_params['blob_size']
    real_sample_num = generation_params['sample_num']
    image_size = training_params['image_size']
    gen_noise = generation_params['noise']

    latent_dim = training_params['latent_dim']
    gen_img_w = training_params['generator_img_w']
    gen_upsamp = training_params['generator_upsamp_size']
    dis_conv = training_params['discriminator_conv_size']
    dis_lin = training_params['discriminator_linear_size']
    training_noise = training_params['noise']

    batch_size = training_params['batch_size']
    num_workers = training_params['num_workers']
    max_epochs = training_params['max_epochs']
    avail_gpus = training_params['avail_gpus']
    
    model_name = '{}-g{}-d{}-bn{}-bs{}-sn1e{}-is{}-ts{}-lr{}-ld{}-gw{}-gu{}-dc{}-dl{}-ns{}'.format(
        gan_version,
        gen_version, dis_version,
        blob_num, blob_size, int(np.log10(real_sample_num)), image_size,
        training_seed, str(lr)[2:],
        latent_dim, gen_img_w, gen_upsamp, dis_conv, dis_lin,
        str(training_noise[1])[2:] if training_noise is not None else '_'
    )
    training_params['model_name'] = model_name
    
    which_checkpoint = testing_params['checkpoint']
    grid_row_num = testing_params['grid_row_num']
    plot_num = testing_params['plot_num']
    gen_sample_num = testing_params['generator_sample_num']
    loss_zoom_bounds = testing_params['loss_zoom_bounds']
    min_distance = testing_params['peak_min_distance']
    threshold_abs = testing_params['peak_threshold_abs']
    filter_sd = testing_params['peak_filter_sd']
    
    """Paths"""
    root_path = training_params['root_path']
    data_path = f'data/{blob_num}_blob'
    data_file_name = f'bn{blob_num}-is{image_size}-bs{blob_size}-sn{real_sample_num}-sd{generation_seed}-ns{int(gen_noise)}.npy'
    chkpt_path = f'checkpoints/{blob_num}_blob/{model_name}'
    log_path = f'logs/{model_name}'    
    save_path = f'{root_path}/plots/{model_name}/images'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    """Initialize seed"""
    torch.manual_seed(training_seed)
    
    """Load data"""
    real_imgs = np.load(f'{root_path}/{data_path}/{data_file_name}')
    data = BlobDataModule(
        data_file=f'{root_path}/{data_path}/{data_file_name}',
        batch_size=batch_size, num_workers=num_workers
        )
    
    """Load model"""
    model = gans[gan_version].load_from_checkpoint(
        f'{root_path}/{chkpt_path}/{which_checkpoint}.ckpt',
        **training_params
        )
    
    """Generate images"""
    trainer = pl.Trainer()
    trainer.test(model, data)
    
    gen_imgs = model.outputs

    """Plot generated images"""
    # Get images
    real_imgs_subset = real_imgs[:grid_row_num**2]
    
    for n in tqdm(range(plot_num), desc='Plotting'):        
        gen_imgs_subset = gen_imgs[n*(grid_row_num**2):(n+1)*(grid_row_num**2)]
        
        # Plotting grid of images
        fig = plt.figure(figsize=(6,3))
        subfig = fig.subfigures(1, 2, wspace=0.2)
        
        plot_img_grid(subfig[0], real_imgs_subset, grid_row_num, title='Real Imgs')
        plot_img_grid(subfig[1], gen_imgs_subset, grid_row_num, title='Generated Imgs')
        
        # Save plot
        plt.savefig(f'{save_path}/gen-imgs_{model_name}_{n}.png')
        plt.close()
    
        # Plotting marginal sums
        fig = plt.figure(figsize=(4,6))
        subfig = fig.subfigures(1, 2)
                
        plot_marginal_sums(real_imgs_subset, subfig[0], grid_row_num, title='Real')
        plot_marginal_sums(gen_imgs_subset, subfig[1], grid_row_num, title='Generated')
        
        # Format
        fig.suptitle('Marginal Sums')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'{save_path}/marg-sums_{model_name}_{n}.png')
        plt.close()
        
        # Peak detection
        real_peaks, real_peak_nums = imgs_peak_finder(
            real_imgs_subset, 
            min_distance=min_distance, threshold_abs=threshold_abs,
            filter_sd=filter_sd
            )
        gen_peaks, gen_peak_nums = imgs_peak_finder(
            gen_imgs_subset, 
            min_distance=min_distance, threshold_abs=threshold_abs,
            filter_sd=filter_sd
            )

        fig = plt.figure(figsize=(6,3))
        subfig = fig.subfigures(1, 2, wspace=0.2)
        
        plot_peak_grid(subfig[0], real_imgs_subset, real_peaks, grid_row_num, 
                       title='real imgaes', subplot_titles=real_peak_nums)
        plot_peak_grid(subfig[1], gen_imgs_subset, gen_peaks, grid_row_num, 
                       title='generated imgaes', subplot_titles=gen_peak_nums)
        
        fig.text(.5, .03, 'number of peaks labelled above image', ha='center')
        
        # Save plot
        plt.savefig(f'{save_path}/peak-imgs_{model_name}_{n}.png')
        plt.close()
        
    
    """Stacking"""
    # Stack images
    print('stacking...')
    stacked_real_img = stack_imgs(real_imgs)
    stacked_gen_img = stack_imgs(gen_imgs)
    
    # Plotting
    fig, axs = plt.subplots(1, 2)
    
    plot_stacked_imgs(axs[0], stacked_real_img, title=f"real\n{real_sample_num} samples")
    plot_stacked_imgs(axs[1], stacked_gen_img, title=f"generated\n{gen_sample_num} samples")
    
    # Format
    fig.suptitle('Stacked Image')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'{save_path}/stacked_{model_name}.png')
    plt.close()
    
    """Mean number of peaks"""
    print('counting peaks...')
    real_imgs_subset = real_imgs[:int(real_sample_num/10)]
    gen_imgs_subset = gen_imgs[:int(gen_sample_num/10)]
     
    _, real_peak_nums = imgs_peak_finder(
            real_imgs_subset, 
            min_distance=min_distance, threshold_abs=threshold_abs,
            filter_sd=filter_sd
            )
    _, gen_peak_nums = imgs_peak_finder(
            gen_imgs_subset, 
            min_distance=min_distance, threshold_abs=threshold_abs,
            filter_sd=filter_sd
            )    
    
    print(f'mean number of real peaks: {np.mean(real_peak_nums)}')
    print(f'mean number of generated peaks: {np.mean(gen_peak_nums)}')
    
    fig, axs = plt.subplots(1,2)
    
    plot_min_num_peaks(axs[0], real_imgs_subset, real_peak_nums, title='real')
    plot_min_num_peaks(axs[1], gen_imgs_subset, gen_peak_nums, title='generated')        
    
    fig.suptitle("Minimum number of peaks, {}")
    if real_sample_num==gen_sample_num:
        fig.suptitle(f"Minimum number of peaks, {int(real_sample_num/10)} samples")
    else:
        fig.suptitle(f"number of samples not equal")
    plt.savefig(f'{save_path}/min-peak_{model_name}.png')
    plt.close()
    
    """Histograms"""
    # Single histogram
    histogram_num = 20
    gen_imgs_subset = gen_imgs[:histogram_num]
    real_imgs_subset = real_imgs[:histogram_num]
    # Create figure
    fig, axs = plt.subplots(1,2)
    # Plot
    plot_histogram(axs[0], real_imgs_subset, color='r', bins=20)
    plot_histogram(axs[1], gen_imgs_subset, color='b', bins=20)
    # Format
    fig.suptitle(f"Histogram of pixel values, {histogram_num} samples")
    plt.tight_layout()
    # Save
    plt.savefig(f'{save_path}/histogram_{model_name}.png')
    plt.close()
    
    # Stack histogram
    fig = plt.figure()
    
    plt.hist(stacked_real_img.ravel(), histtype='step', label='real', color='r')
    plt.hist(stacked_gen_img.ravel(), histtype='step', label='generated', color='b')
    if real_sample_num==gen_sample_num:
        plt.suptitle(f"stack of {real_sample_num} samples")
    else:
        plt.suptitle(f"number of samples not equal\n{real_sample_num} real samples | {gen_sample_num} generated samples")
    # Format
    plt.legend()
    plt.tight_layout()
    # Save
    plt.savefig(f'{save_path}/histogram-stack_{model_name}.png')
    plt.close()
    
    
    """Losses"""
    try:
        # Load loss
        losses = np.load(f'{root_path}/{log_path}/losses.npz')
        g_losses = losses['g_losses']
        d_losses = losses['d_losses']
        epochs = losses['epochs']
        
        # Plot
        plt.figure(figsize=(6,4))
        plt.plot(epochs, g_losses, label='generator')
        plt.plot(epochs, d_losses, label='discriminator')
        
        # Format
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Model Loss')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'{save_path}/losses_{model_name}.png')
        plt.close()
        
        # Plot zoom 
        plt.figure(figsize=(6,4))
        plt.plot(epochs, g_losses, label='generator')
        plt.plot(epochs, d_losses, label='discriminator')
        
        # Format
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Model Loss')
        if gan_version=='CWGAN':
            plt.axhline(y=0, color='r', alpha=0.2, linestyle='dashed')
        plt.ylim(*loss_zoom_bounds)
        plt.legend()
        plt.tight_layout()
        
        # Save plots
        plt.savefig(f'{save_path}/losses-zm_{model_name}.png')
        plt.close()
        
    except FileNotFoundError:
        print('losses not found')