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

def run_testing(training_params, generation_params, testing_params, testing_restart=False):
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
    subset_sample_num = testing_params['subset_sample_num']
    loss_zoom_bounds = testing_params['loss_zoom_bounds']
    min_distance = testing_params['peak_min_distance']
    threshold_abs = testing_params['peak_threshold_abs']
    filter_sd = testing_params['peak_filter_sd']
    blob_threshold_rel = testing_params['blob_threshold_rel']
    testing_seed = testing_params['seed']
    
    """Paths"""
    root_path = training_params['root_path']
    data_path = f'data/{blob_num}_blob'
    data_file_name = f'bn{blob_num}-is{image_size}-bs{blob_size}-sn{real_sample_num}-sd{generation_seed}-ns{int(gen_noise)}.npy'
    chkpt_path = f'checkpoints/{blob_num}_blob/{model_name}'
    log_path = f'logs/{model_name}'    
    plot_save_path = f'{root_path}/plots/{model_name}/images'
    output_save_path = f'{root_path}/plots/{model_name}/model_output'
    
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    if not os.path.exists(output_save_path):
        os.makedirs(output_save_path)
        
    """Initialize seed"""
    torch.manual_seed(testing_seed)
    
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
    try:
        if not testing_restart:
            print('Model output found')
            gen_imgs = np.load(f'{output_save_path}/{which_checkpoint}.npy')
            print('Model output loaded')
        else:
            print('Retesting model...')
            trainer = pl.Trainer()
            trainer.test(model, data)
            
            gen_imgs = model.outputs
            
            # Save outputs
            np.save(f'{output_save_path}/{which_checkpoint}.npy', gen_imgs)
        
    except FileNotFoundError:
        trainer = pl.Trainer()
        trainer.test(model, data)
        
        gen_imgs = model.outputs
        
        # Save outputs
        np.save(f'{output_save_path}/{which_checkpoint}.npy', gen_imgs)

    """Plot generated images"""
    for n in tqdm(range(plot_num), desc='Plotting'):
        # Get images
        real_imgs_subset = real_imgs[n*(grid_row_num**2):(n+1)*(grid_row_num**2)]
        gen_imgs_subset = gen_imgs[n*(grid_row_num**2):(n+1)*(grid_row_num**2)]
        
        # Plotting grid of images
        fig = plt.figure(figsize=(6,3))
        subfig = fig.subfigures(1, 2, wspace=0.2)
        
        plot_img_grid(subfig[0], real_imgs_subset, grid_row_num, title='Real Imgs')
        plot_img_grid(subfig[1], gen_imgs_subset, grid_row_num, title='Generated Imgs')
        
        # Save plot
        plt.savefig(f'{plot_save_path}/gen-imgs_{model_name}_{n}.png')
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
        plt.savefig(f'{plot_save_path}/marg-sums_{model_name}_{n}.png')
        plt.close()
        
        # Peak detection
        real_peaks_coords, real_peak_nums, real_peak_vals = imgs_peak_finder(
            real_imgs_subset, 
            min_distance=min_distance, threshold_abs=threshold_abs,
            filter_sd=filter_sd,
            progress_bar=False
            )
        gen_peaks_coords, gen_peak_nums, gen_peak_vals = imgs_peak_finder(
            gen_imgs_subset, 
            min_distance=min_distance, threshold_abs=threshold_abs,
            filter_sd=filter_sd,
            progress_bar=False
            )
        
        real_indv_peak_counts, real_img_blob_counts = count_blobs(real_peak_vals, generation_params['blob_num'])
        gen_indv_peak_counts, gen_img_blob_counts = count_blobs(gen_peak_vals, generation_params['blob_num'])

        fig = plt.figure(figsize=(6,3))
        subfig = fig.subfigures(1, 2, wspace=0.2)
        
        plot_peak_grid(subfig[0], real_imgs_subset, real_peaks_coords, real_indv_peak_counts, grid_row_num, 
                       title='real imgaes', subplot_titles=real_peak_nums)
        plot_peak_grid(subfig[1], gen_imgs_subset, gen_peaks_coords, gen_indv_peak_counts, grid_row_num, 
                       title='generated imgaes', subplot_titles=gen_peak_nums)
        
        fig.text(.5, .03, 'number of peaks labelled above image', ha='center')
        
        # Save plot
        plt.savefig(f'{plot_save_path}/peak-imgs_{model_name}_{n}.png')
        plt.close()
        
        # Gaussian elimination and blob counting
        real_blob_coords, real_blob_nums, real_peak_vals = imgs_blob_finder(
            real_imgs_subset, 
            blob_size=blob_size, min_peak_threshold=(1/blob_num)*0.7,
            filter_sd=filter_sd,
            progress_bar=False
            )
        gen_blob_coords, gen_blob_nums, gen_peak_vals = imgs_blob_finder(
            gen_imgs_subset, 
            blob_size=blob_size, min_peak_threshold=(1/blob_num)*0.7,
            filter_sd=filter_sd,
            progress_bar=False
            )
        
        real_indv_peak_counts, real_img_blob_counts = count_blobs(real_peak_vals, generation_params['blob_num'])
        gen_indv_peak_counts, gen_img_blob_counts = count_blobs(gen_peak_vals, generation_params['blob_num'])

        fig = plt.figure(figsize=(6,3))
        subfig = fig.subfigures(1, 2, wspace=0.2)
        
        plot_peak_grid(subfig[0], real_imgs_subset, real_blob_coords, real_indv_peak_counts, grid_row_num, 
                       title='real imgaes', subplot_titles=real_img_blob_counts)
        plot_peak_grid(subfig[1], gen_imgs_subset, gen_blob_coords, gen_indv_peak_counts, grid_row_num, 
                       title='generated imgaes', subplot_titles=gen_img_blob_counts)
        
        fig.text(.5, .03, 'number of blobs labelled above image', ha='center')
        
        # Save plot
        plt.savefig(f'{plot_save_path}/counts-imgs_{model_name}_{n}.png')
        plt.close()
    
    """Testing sample subset"""
    real_imgs_subset = real_imgs[:subset_sample_num]
    gen_imgs_subset = gen_imgs[:subset_sample_num]
    
    """Stacking"""
    # Stack images
    print('stacking...')
    stacked_real_img = stack_imgs(real_imgs_subset)
    stacked_gen_img = stack_imgs(gen_imgs_subset)
    
    # Plotting
    fig, axs = plt.subplots(1, 2)
    
    plot_stacked_imgs(axs[0], stacked_real_img, title=f"real\n{subset_sample_num} samples")
    plot_stacked_imgs(axs[1], stacked_gen_img, title=f"generated\n{subset_sample_num} samples")
    
    # Format
    fig.suptitle('Stacked Image')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'{plot_save_path}/stacked_{model_name}.png')
    plt.close()
    
    """Number of peaks statistics"""
    # Count blobs
    print('counting blobs...')
    
    real_blob_coords, real_blob_nums, real_peak_vals = imgs_blob_finder(
        real_imgs_subset, 
        blob_size=blob_size, min_peak_threshold=(1/blob_num)*blob_threshold_rel,
        filter_sd=filter_sd,
        progress_bar=True
        )
    gen_blob_coords, gen_blob_nums, gen_peak_vals = imgs_blob_finder(
        gen_imgs_subset, 
        blob_size=blob_size, min_peak_threshold=(1/blob_num)*blob_threshold_rel,
        filter_sd=filter_sd,
        progress_bar=True
        )
    
    real_indv_peak_counts, real_img_blob_counts = count_blobs(real_peak_vals, generation_params['blob_num'])
    gen_indv_peak_counts, gen_img_blob_counts = count_blobs(gen_peak_vals, generation_params['blob_num'])
    
    'Mean'
    real_blob_num_mean = np.mean(real_img_blob_counts)
    gen_blob_num_mean = np.mean(gen_img_blob_counts)
    print(f'mean number of real peaks: {real_blob_num_mean}')
    print(f'mean number of generated peaks: {gen_blob_num_mean}')
    
    'Histogram'
    # Create figure
    fig = plt.figure()
    
    # Plot
    plt.hist(real_img_blob_counts, bins=np.arange(4.5,14.5,1), 
             histtype='step', label='real', color=(1,0,0,0.8))
    plt.hist(gen_img_blob_counts, bins=np.arange(4.5,14.5,1), 
             histtype='step', label='generated', color=(0,0,1,0.8))
    
    plt.axvline(real_blob_num_mean, color=(1,0,0,0.5), linestyle='dashed', linewidth=1)
    plt.axvline(gen_blob_num_mean, color=(0,0,1,0.5), linestyle='dashed', linewidth=1)
    
    # Format
    min_ylim, max_ylim = plt.ylim()
    plt.text(real_blob_num_mean*1.05, max_ylim*0.9, 'Mean: {:.2f}'.format(real_blob_num_mean), color=(1,0,0,1))
    plt.text(gen_blob_num_mean*1.05, max_ylim*0.8, 'Mean: {:.2f}'.format(gen_blob_num_mean), color=(0,0,1,1))    
    plt.ylabel('counts')
    plt.xlabel('number of blobs')
    plt.suptitle(f"{subset_sample_num} samples")
    plt.legend()
    plt.tight_layout()
    
    # Save
    plt.savefig(f'{plot_save_path}/histogram-number-blobs_{model_name}.png')
    plt.close()    
    
    'Extreme number of blobs'
    # Create figure
    fig = plt.figure(figsize=(3,5))
    subfig = fig.subfigures(1, 2, wspace=0.2)
    
    # Plot
    plot_extremum_num_blobs(subfig[0], real_imgs_subset, real_img_blob_counts, extremum='min', title='real')
    plot_extremum_num_blobs(subfig[1], gen_imgs_subset, gen_img_blob_counts, extremum='min', title='generated')  
    
    # Format
    fig.suptitle(f"Min blob count, {subset_sample_num} samples")
    plt.tight_layout()
    
    # Save
    plt.savefig(f'{plot_save_path}/min-peak_{model_name}.png')
    plt.close()

    # Create figure
    fig = plt.figure(figsize=(3,5))
    subfig = fig.subfigures(1, 2, wspace=0.2)
    
    # Plot
    plot_extremum_num_blobs(subfig[0], real_imgs_subset, real_img_blob_counts, extremum='max', title='real')
    plot_extremum_num_blobs(subfig[1], gen_imgs_subset, gen_img_blob_counts, extremum='max', title='generated')  
    
    # Format
    fig.suptitle(f"Max blobs count, {subset_sample_num} samples")
    plt.tight_layout()
    
    # Save
    plt.savefig(f'{plot_save_path}/max-peak_{model_name}.png')
    plt.close()
    
    """2-point correlation"""
    print('calculating 2 point correlation...')
    real_corrs, edges = stack_two_point_correlation(real_blob_coords, image_size, bins=20, progress_bar=True)
    gen_corrs, _ = stack_two_point_correlation(gen_blob_coords, image_size, bins=20, progress_bar=True)
    
    # Create figure
    fig, ax = plt.subplots()
    
    # Plot
    plot_histogram_stack(ax, real_corrs, edges, color=(1,0,0,0.5), label='real')
    plot_histogram_stack(ax, gen_corrs, edges, color=(0,0,1,0.5), label='generated',
                         xlabel='2-point correlation', ylabel='pair distance')
    
    # Format
    fig.suptitle(f"2 point correlation, {subset_sample_num} samples")
    plt.tight_layout()
    plt.legend(loc='lower right')
    
    # Save
    plt.savefig(f'{plot_save_path}/2-pt-corr_{model_name}.png')
    plt.close() 
    
    """Pixel image histograms"""
    'Single histogram'
    histogram_num = 20
    gen_imgs_subset = gen_imgs[:histogram_num]
    real_imgs_subset = real_imgs[:histogram_num]
    
    # Create figure
    fig, axs = plt.subplots(1,2)
    
    # Plot
    plot_histogram(axs[0], real_imgs_subset, color=(1,0,0,0.5), bins=20)
    plot_histogram(axs[1], gen_imgs_subset, color=(0,0,1,0.5), bins=20)
    
    # Format
    fig.suptitle(f"Histogram of pixel values, {histogram_num} samples")
    plt.tight_layout()
    
    # Save
    plt.savefig(f'{plot_save_path}/histogram_{model_name}.png')
    plt.close()
    
    'Stack histogram'
    print('stacking histograms...')
    real_hist_stack = stack_histograms(real_imgs)
    gen_hist_stack = stack_histograms(gen_imgs)
    
    # Create figure
    fig, ax = plt.subplots()
    
    # Plot
    plot_histogram_stack(ax, *real_hist_stack, color=(1,0,0,0.8), label='real')
    plot_histogram_stack(ax, *gen_hist_stack, color=(0,0,1,0.8), label='generated')
    
    # Format
    fig.suptitle(f"Stacked histogram of pixel values, {subset_sample_num} samples")
    plt.tight_layout()
    
    # Save
    plt.savefig(f'{plot_save_path}/histogram-stack_{model_name}.png')
    plt.close() 
    
    'Stacked image histogram'
    # Create figure
    fig = plt.figure()
    
    # Plot
    plt.hist(stacked_real_img.ravel(), histtype='step', label='real', color=(1,0,0,0.8))
    plt.hist(stacked_gen_img.ravel(), histtype='step', label='generated', color=(0,0,1,0.8))
    
    # Format
    plt.ylabel('counts')
    plt.xlabel('stacked pixel value')
    plt.suptitle(f"stack of {subset_sample_num} samples")
    plt.legend()
    plt.tight_layout()
    
    # Save
    plt.savefig(f'{plot_save_path}/histogram-stacked-img_{model_name}.png')
    plt.close()
    
    
    """Losses"""
    try:
        # Load loss
        losses = np.load(f'{root_path}/{log_path}/losses.npz')
        g_losses = losses['g_losses']
        d_losses = losses['d_losses']
        epochs = losses['epochs']
        
        # Plot
        fig, ax1 = plt.subplots()
        
        ax1.plot(epochs, g_losses, color='C0', linewidth=0.7)
        ax2 = ax1.twinx() 
        ax2.plot(epochs, d_losses, color='C1', linewidth=0.7)

        
        # Format
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('generator loss', color='C0')
        ax2.set_ylabel('discriminator loss', color='C1')
        ax1.tick_params(axis='y', labelcolor='C0')
        ax2.tick_params(axis='y', labelcolor='C1')

        fig.tight_layout()
        
        # Save plot
        plt.savefig(f'{plot_save_path}/losses_{model_name}.png')
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
        plt.savefig(f'{plot_save_path}/losses-zm_{model_name}.png')
        plt.close()
        
    except FileNotFoundError:
        print('losses not found')