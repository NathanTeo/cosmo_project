"""
Author: Nathan Teo

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
        generation_seed = generation_params['seed']
        blob_size = generation_params['blob_size']
        sample_num = generation_params['sample_num']
        image_size = training_params['image_size']

        latent_dim = training_params['latent_dim']
        gen_img_w = training_params['generator_img_w']
        gen_upsamp = training_params['generator_upsamp_size']
        dis_conv = training_params['discriminator_conv_size']
        dis_lin = training_params['discriminator_linear_size']

        batch_size = training_params['batch_size']
        num_workers = training_params['num_workers']
        max_epochs = training_params['max_epochs']
        avail_gpus = training_params['avail_gpus']

        model_name = '{}-g{}-d{}-bn{}-bs{}-sn1e{}-is{}-ts{}-lr{}-ld{}-gw{}-gu{}-dc{}-dl{}'.format(
        gan_version,
        gen_version, dis_version,
        blob_num, blob_size, int(np.log10(sample_num)), image_size,
        training_seed, str(lr)[2:],
        latent_dim, gen_img_w, gen_upsamp, dis_conv, dis_lin
        )
        
        """Paths"""
        root_path = training_params['root_path']
        data_folder = f'data/{blob_num}_blob'
        data_file_name = f'{blob_num}blob_imgsize{image_size}_blobsize{blob_size}_samplenum{sample_num}_seed{generation_seed}.npy'
        chkpt_path = f'checkpoints/{blob_num}_blob/{model_name}'
        training_params['model_name'] = model_name
        
        # folder for logging imgs
        if not os.path.exists(f'{root_path}/logs/{model_name}'):
                os.makedirs(f'{root_path}/logs/{model_name}/images')
        
        """Initialize callbacks"""
        # os.environ['WANDB_API_KEY']='3c2e56d7951699612266059b4061b9f87f462ec4'
        # os.environ['WANDB_ENTITY']='nathanteo'
        wandb.login()
        wandb_logger = WandbLogger(
                project='cosmo_project',
                name=model_name
                )
        wandb_logger.experiment.config.update(training_params)

        checkpoint_callback = ModelCheckpoint(
                monitor = 'g_loss',
                dirpath = f'{root_path}/{chkpt_path}',
                filename = 'min',
                save_top_k = 3,  # Save only the top 1 checkpoint
                save_last = True,
                enable_version_counter=False
        )

        """Initialize seed"""
        torch.manual_seed(training_seed)

        """Training"""
        if __name__ == 'code_model.train':
                # Load data
                data = BlobDataModule(
                        data_file=f'{root_path}/{data_folder}/{data_file_name}',
                        batch_size=batch_size, num_workers=num_workers
                        )
                'data = MNISTDataModule(batch_size=batch_size, num_workers=num_workers)'
                
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
                                        ckpt_path=f'{root_path}/{chkpt_path}/{model_name}.ckpt'
                                )                                
                        except(FileNotFoundError):
                                print('--------------------------------------------')
                                print('no trained model found, training new model')
                                print('--------------------------------------------')
                                trainer.fit(model, data)
                        
                os.system("rm -r wandb")