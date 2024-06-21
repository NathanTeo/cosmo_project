import os 
import torch

"""Params"""
state = 'train'
training_restart = False

generation_params = {
        'blob_num': 1,
        'blob_size':5,
        'image_size': 28,
        'sample_num': 10000,
        'seed':70
        }

training_params = {
        'gan_version': 'CGAN',
        'generator_version': 3,
        'discriminator_version': 2,
        'random_seed': 40,
        'avail_gpus': min(1, torch.cuda.device_count()),
        'num_workers': int(os.cpu_count() / 2),
        'image_size': generation_params['image_size'], # update this to directly take from imported
        'batch_size': 32,
        'lr': 0.0005,
        'betas': (0.9,0.999),
        'gp_lambda': 100,
        'max_epochs': 100,
        'epoch_start_g_train': 0,
        'discriminator_train_freq': 1,
        'latent_dim': 15,
        'gen_img_w': 16,
        'generator_upsamp_size': 32,
        'input_channels': 1,
        'discriminator_conv_size': 8,
        'discriminator_linear_size': 32,
        'linear_dropout': 0.2,
        'conv_dropout': 0.2,
        'root_path': "C:/Users/Idiot/Desktop/Research/OFYP/cosmo_project"
}

"""Run"""
import sys
sys.path.append(f"{training_params['root_path']}")
from code_model.train import *
from code_model.test import *

if __name__ == "__main__":
    if state == 'train':
        run_training(training_params, generation_params, training_restart)
    elif state == 'test':
        run_testing(training_params, generation_params)