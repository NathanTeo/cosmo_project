"""
Author: Nathan Teo

This script runs GAN training
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from code_model.GAN_modules import *
from code_model.data_modules import *

def run_training(training_params, generation_params, training_restart=False):
        """Initialize Params"""
        gan_version = training_params['gan_version']
        gen_version = training_params['generator_version']
        dis_version = training_params['discriminator_version']
        training_seed = training_params['random_seed']
        lr = training_params['lr']

        blob_num = generation_params['blob_num']
        data_distribution = generation_params['distribution']
        generation_seed = generation_params['seed']
        blob_size = generation_params['blob_size']
        sample_num = generation_params['sample_num']
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
        blob_num, data_distribution[0], blob_size, "{:.0g}".format(sample_num), image_size,
        training_seed, "{:.0g}".format(lr),
        latent_dim, gen_img_w, gen_upsamp, dis_conv, dis_lin,
        str(training_noise[1])[2:] if training_noise is not None else '_'
        )
        
        """Paths"""
        root_path = training_params['root_path']
        data_path = f'data/{blob_num}_blob'
        data_file_name = f'bn{blob_num}{data_distribution[0]}-is{image_size}-bs{blob_size}-sn{sample_num}-sd{generation_seed}-ns{int(gen_noise)}.npy'
        chkpt_path = f'checkpoints/{blob_num}_blob/{model_name}'
        training_params['model_name'] = model_name
        
        # folder for logging imgs
        if not os.path.exists(f'{root_path}/logs/{model_name}'):
                os.makedirs(f'{root_path}/logs/{model_name}/images')
        
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
                monitor = 'g_loss',
                dirpath = f'{root_path}/{chkpt_path}',
                filename = '{}',
                every_n_epochs=20,
                save_top_k = -1,
                save_last = True,
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
                model = gans[gan_version](**training_params)
                
                # Initialize trainer
                trainer = pl.Trainer(
                        max_epochs=max_epochs,
                        logger=wandb_logger,
                        callbacks=[checkpoint_callback],
                        log_every_n_steps=data.num_training_batches()
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
                        