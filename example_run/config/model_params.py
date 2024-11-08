"""
Author: Nathan Teo

This script is executed to train or test the model.
All tunable parameters are listed in the dictionaries.
"""

import os 

"""Params"""
training_restart = False
testing_restart = False

generation_params = {
        'blob_num': 10,
        'num_distribution': 'delta',
        'clustering': None,
        'blob_size': 5,
        'blob_amplitude': 1,
        'amplitude_distribution': 'delta',
        'image_size': 32,
        'sample_num': 10000,
        'seed': 70,
        'noise': False
        }

training_params = {
        'model_version': 'Diffusion',
        'unet_version': 2,
        'random_seed': 40,
        'num_workers': int(os.cpu_count() / 2),
        'image_size': generation_params['image_size'],
        'batch_size': 512,
        'lr': 3e-4,
        'scheduler_params': (1, 0.99),
        'max_epochs': 100,
        'noise': None,
        'network_params': {
            'image_channels': 1,
            'noise_steps': 1000,
            'model_dim': 32,
            'dim_mult': [1,2,4,8],
            'time_dim': 128
        }
}

testing_params = {
    'checkpoint': 'last',
    'seed': 40,
    'grid_row_num': 2,
    'plot_num': 5, 
    'subset_sample_num': 5000,
    'loss_zoom_bounds': (-1,1), # Implement Auto?
    'counting_params': None
}

"""Run"""
project_path = "C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_project" 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("state")
    args = parser.parse_args()
    
    import torch
    
    training_params['avail_gpus'] = torch.cuda.device_count()
    training_params['root_path'] = f"C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_runs/{args.run}"
    
    import sys
    sys.path.append(project_path)
    from code_model.train import run_training
    from code_model.test import run_testing
    from code_model.generate import run_generation
    
    if args.state == 'train':
        run_training(training_params, generation_params, testing_params, training_restart)
    elif args.state == 'test':
        run_testing(training_params, generation_params, testing_params, testing_restart)
    elif args.state == 'generate':
        run_generation(training_params, generation_params, testing_params)