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

def run_plot_logs(training_params, generation_params, testing_params, testing_restart=False):    
    """Initialize variables"""
    gan_version = training_params['gan_version']
    gen_version = training_params['generator_version']
    dis_version = training_params['discriminator_version']
    training_seed = training_params['random_seed']
    lr = training_params['lr']

    blob_num = generation_params['blob_num']
    data_distribution = generation_params['distribution']
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
    
    model_name = '{}-g{}-d{}-bn{}{}-bs{}-sn{}-is{}-ts{}-lr{}-ld{}-gw{}-gu{}-dc{}-dl{}-ns{}'.format(
    gan_version,
    gen_version, dis_version,
    blob_num, data_distribution[0], blob_size, "{:.0g}".format(real_sample_num), image_size,
    training_seed, "{:.0g}".format(lr),
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
    data_file_name = f'bn{blob_num}{data_distribution[0]}-is{image_size}-bs{blob_size}-sn{real_sample_num}-sd{generation_seed}-ns{int(gen_noise)}.npy'
    chkpt_path = f'checkpoints/{blob_num}_blob/{model_name}'
    log_path = f'logs/{model_name}'    
    plot_save_path = f'{root_path}/plots/{model_name}/images'
    output_save_path = f'{root_path}/plots/{model_name}/model_output'
    
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    if not os.path.exists(output_save_path):
        os.makedirs(output_save_path)
        
    if not testing_restart and os.path.isfile(f'{plot_save_path}/losses-zm_{model_name}.png'):
        print('losses already plotted, skipping step')
        return
    
    """Logged losses"""
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
    
    """Losses against last model"""
    # Initialize seed
    torch.manual_seed(testing_seed)
    
    # Load data
    ratio = 1000/real_sample_num # temporary, change this!
    
    real_imgs = np.load(f'{root_path}/{data_path}/{data_file_name}')
    data = BlobDataModule(
        data_file=f'{root_path}/{data_path}/{data_file_name}',
        batch_size=batch_size, num_workers=num_workers, 
        truncate_ratio=ratio
        )
    
    real_imgs_subset = real_imgs[:int(len(real_imgs)*ratio)]
    
    # Load models
    filenames = os.listdir(chkpt_path)
    filenames.remove('last.ckpt')
    
    epochs = [int(file[6:-5]) for file in filenames]
    
    models = [gans[gan_version].load_from_checkpoint(
        f'{root_path}/{chkpt_path}/{file}',
        **training_params
        ) for file in filenames]
    
    last_model = gans[gan_version].load_from_checkpoint(
        f'{root_path}/{chkpt_path}/last.ckpt',
        **training_params
        )
    
    # Generate images
    trainer = pl.Trainer()
    
    print('generating images, last model...')
    trainer.test(last_model, data)
    last_gen_imgs = last_model.outputs
    
    print('generating images, earlier models...')
    models_gen_imgs = []
    for epoch, model in zip(epochs, models):
        print(f'epoch {epoch}')
        trainer.test(model, data)
        models_gen_imgs.append(model.outputs)
    
    models_gen_imgs = np.array(models_gen_imgs)
    
    'Calculate model scores for last generator with epoch, check how discriminator evolves'
    # Score
    print('scoring last generator with earlier models...')
    d_evo_models_gen_scores = []
    d_evo_models_real_scores = []
    for epoch, model in zip(epochs, models):
        print(f'epoch {epoch}')
        d_evo_models_gen_scores.append(model.score_samples(last_gen_imgs, progress_bar=True))
        d_evo_models_real_scores.append(model.score_samples(real_imgs_subset, progress_bar=True))
    
    # Loss
    d_evo_loss = [
        -(np.mean(real_score) - np.mean(gen_score))
        for real_score, gen_score in zip(d_evo_models_real_scores, d_evo_models_gen_scores)
        ]
    
    'Calculate model scores for last discriminator with epoch, check how generator evolves'
    # Score
    print('scoring ealier generators with last model...')
    g_evo_models_gen_scores = []
    for epoch, gen_imgs in zip(epochs, models_gen_imgs):
        print(f'epoch {epoch}')
        g_evo_models_gen_scores.append(last_model.score_samples(gen_imgs, progress_bar=True))
        
    # Loss
    g_evo_loss = [-np.mean(gen_scores) for gen_scores in g_evo_models_gen_scores]
    
    'Plot'
    fig, ax1 = plt.subplots()

    ax1.plot(epochs, g_evo_loss, color='C0', linewidth=0.7)
    ax2 = ax1.twinx() 
    ax2.plot(epochs, d_evo_loss, color='C1', linewidth=0.7)


    # Format
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('generator loss', color='C0')
    ax2.set_ylabel('discriminator loss', color='C1')
    ax1.tick_params(axis='y', labelcolor='C0')
    ax2.tick_params(axis='y', labelcolor='C1')

    fig.tight_layout()

    # Save plot
    plt.savefig(f'{plot_save_path}/losses-wrt-last_{model_name}.png')
    plt.close()