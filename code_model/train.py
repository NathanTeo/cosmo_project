"""
Author: Nathan Teo

This script runs model training
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from code_model.models import *
from code_model.data_modules import *
from code_model.testers.plotting_utils import *
from code_model.testers.eval_utils import *
from code_model.testers.modules import *

def run_training(training_params, generation_params, testing_params, training_restart=False, generate_samples=False):
        """Initialize Params"""
        model_version = training_params['model_version']

        training_seed = training_params['random_seed']
        lr = training_params['lr']

        blob_num = generation_params['blob_num']
        data_distribution = generation_params['distribution']
        generation_seed = generation_params['seed']
        blob_size = generation_params['blob_size']
        sample_num = generation_params['sample_num']
        image_size = training_params['image_size']
        gen_noise = generation_params['noise']

        network_params = training_params['network_params']
        training_noise = training_params['noise']
        batch_size = training_params['batch_size']
        num_workers = training_params['num_workers']
        max_epochs = training_params['max_epochs']
        avail_gpus = training_params['avail_gpus']


        if 'GAN' in model_version:
                gen_version = training_params['generator_version']
                dis_version = training_params['discriminator_version']
                
                latent_dim = network_params['latent_dim']
                gen_img_w = network_params['generator_img_w']
                gen_upsamp = network_params['generator_upsamp_size']
                dis_conv = network_params['discriminator_conv_size']
                dis_lin = network_params['discriminator_linear_size']
        
                model_name = '{}-g{}-d{}-bn{}{}-bs{}-sn{}-is{}-ts{}-lr{}-ld{}-gw{}-gu{}-dc{}-dl{}-ns{}'.format(
                model_version,
                gen_version, dis_version,
                blob_num, data_distribution[0], blob_size, "{:.0e}".format(sample_num), image_size,
                training_seed, "{:.0e}".format(lr),
                latent_dim, gen_img_w, gen_upsamp, dis_conv, dis_lin,
                str(training_noise[1])[2:] if training_noise is not None else '_'
                )
        elif 'Diffusion' in model_version:
                unet_version = training_params['unet_version']
                
                noise_steps = ['noise_steps']
                time_dim = ['time_dim']
                initial_size = ['ininial_size']
        
                model_name = '{}-n{}-bn{}{}-bs{}-sn{}-is{}-ts{}-lr{}-st{}-td{}-sz{}-ns{}'.format(
                model_version,
                unet_version,
                blob_num, data_distribution[0], blob_size, "{:.0e}".format(sample_num), image_size,
                training_seed, "{:.0e}".format(lr),
                noise_steps, time_dim, initial_size, 
                str(training_noise[1])[2:] if training_noise is not None else '_'
                )
                
                
        """Paths"""
        root_path = training_params['root_path']
        data_path = 'data'
        data_file_name = f'bn{blob_num}{data_distribution[0]}-is{image_size}-bs{blob_size}-sn{sample_num}-sd{generation_seed}-ns{int(gen_noise)}.npy'
        chkpt_path = 'checkpoints'
        training_params['model_name'] = model_name
        
        # Folder for backup
        if training_restart:
                os.system(f'rm -r /{root_path}/logs')
                os.system(f'rm -r /{root_path}/checkpoints')
        if not os.path.exists(f'{root_path}/logs'):
                os.makedirs(f'{root_path}/logs/images')
        if not os.path.exists(f'{root_path}/checkpoints'):
                os.makedirs(f'{root_path}/checkpoints')
        
        # Folder for backup
        if not os.path.exists(f'{root_path}/backup'):
                os.makedirs(f'{root_path}/backup/checkpoints')
                os.makedirs(f'{root_path}/backup/logs')
        
        """Initialize callbacks"""
        # Logger
        wandb.login()
        wandb_logger = WandbLogger(
                project='cosmo_project',
                name=model_name
                )
        wandb_logger.experiment.config.update(training_params)

        # Checkpoint
        checkpoint_callback = ModelCheckpoint(
                monitor=None,
                dirpath=f'{root_path}/{chkpt_path}',
                filename='{epoch:04d}',
                every_n_epochs=20,
                save_top_k=-1,
                save_last=True,
                enable_version_counter=False
        )

        """Initialize seed"""
        torch.manual_seed(training_seed)

        """Training"""
        if __name__ == 'code_model.train':
                # Load data
                data = BlobDataModule(
                        data_file=f'{root_path}/{data_path}/{data_file_name}',
                        batch_size=batch_size, num_workers=num_workers
                        )
                
                # Load model
                model = model_dict[model_version](**training_params)
                
                # Initialize trainer
                if avail_gpus<2:
                        trainer = pl.Trainer(
                                max_epochs=max_epochs,
                                logger=wandb_logger,
                                callbacks=[checkpoint_callback],
                                log_every_n_steps=data.num_training_batches(),
                        )
                elif avail_gpus>=2: # DDP for multi gpu 
                        trainer = pl.Trainer(
                                max_epochs=max_epochs,
                                logger=wandb_logger,
                                callbacks=[checkpoint_callback],
                                log_every_n_steps=data.num_training_batches(),
                                devices=avail_gpus, strategy='ddp'
                        )
                
                # Train
                if training_restart:
                        print('--------------------')
                        print('training restarted')
                        print('--------------------')
                        trainer.fit(model, data)
                else:
                        try:
                                # Continue training from existing checkpoint
                                print('--------------------')
                                print('training continued')
                                print('--------------------')
                                trainer.fit(
                                        model, data,
                                        ckpt_path=f'{root_path}/{chkpt_path}/last.ckpt'
                                )                                
                        except(FileNotFoundError):
                                print('--------------------------------------------')
                                print('no trained model found, training new model')
                                print('--------------------------------------------')
                                trainer.fit(model, data)
        
                # Generate dataset for testing
                dataset = testDataset(generation_params, training_params, testing_params)
                
                if generate_samples:
                        dataset.prep_data(BlobDataModule, model_dict, testing_restart=True)
                
                # plot losses
                if 'GAN' in training_params['model_version']:
                        logs_plotter = ganLogsPlotter(dataset)
                        logs_plotter.plot_logs(BlobDataModule, model_dict, testing_restart=True)
                elif 'Diffusion' in training_params['model_version']:
                        logs_plotter = diffLogsPlotter(dataset)
                        logs_plotter.plot_logs(BlobDataModule, testing_restart=True)


                        