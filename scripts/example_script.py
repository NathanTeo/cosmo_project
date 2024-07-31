"""
Author: Nathan Teo

This script is executed to train or test the model.
All tunable parameters are listed in the dictionaries.
"""

import os 
import torch

"""Params"""
state = 'train'
training_restart = True
testing_restart = True

generation_params = {
        'blob_num': 1,
        'distribution': 'uniform',
        'blob_size': 5,
        'image_size': 28,
        'sample_num': 100,
        'seed': 70,
        'noise': False
        }

training_params = {
        'gan_version': 'CGAN',
        'generator_version': 2,
        'discriminator_version': 1,
        'random_seed': 40,
        'avail_gpus': torch.cuda.device_count(),
        'num_workers': int(os.cpu_count() / 2), # cpu count is the number of threads, num workers is the number of cores?
        'image_size': generation_params['image_size'], 
        'batch_size': 32,
        'lr': 0.0005,
        'betas': (0.9,0.999),
        'gp_lambda': 10, # Only for Wasserstein GAN
        'scheduler_params': ((2, 0.1), (0.1, 0.1), 0.95, 50),
        'noise': (0, 0.05),
        'max_epochs': 50,
        'epoch_start_g_train': 0,
        'discriminator_train_freq': 1,
        'latent_dim': 10,
        'generator_img_w': 16,
        'generator_upsamp_size': 8,
        'input_channels': 1,
        'discriminator_conv_size': 4,
        'discriminator_linear_size': 16,
        'linear_dropout': 0.2,
        'conv_dropout': 0.2,
        'root_path': "Users/..."
}

testing_params = {
    'checkpoint': 'last',
    'seed': 40,
    'grid_row_num': 2,
    'plot_num': 5, 
    'subset_sample_num': generation_params['sample_num'],
    'loss_zoom_bounds': (-1,1), # Implement Auto?
    'peak_min_distance': 1, 
    'peak_threshold_abs': 0.02,
    'peak_filter_sd': 1,
    'blob_threshold_rel': 0.7
}

"""Run"""
import sys
sys.path.append(f"{training_params['root_path']}")
from code_model.train import *
from code_model.test import *
from code_model.plot_metric_logs import *

if __name__ == "__main__":
    if state == 'train':
        run_training(training_params, generation_params, training_restart)
    elif state == 'test':
        run_plot_logs(training_params, generation_params, testing_params, testing_restart)
        run_testing(training_params, generation_params, testing_params, testing_restart)