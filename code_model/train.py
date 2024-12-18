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

def run_training(training_params, generation_params, testing_params, training_restart=False, generate_samples=True):
	"""Initialize Params"""
	model_version = training_params['model_version']

	training_seed = training_params['random_seed']

	batch_size = training_params['batch_size']
	num_workers = training_params['num_workers']
	max_epochs = training_params['max_epochs']
	avail_gpus = training_params['avail_gpus']
	
	data_transforms = init_param(training_params, 'data_transforms')
			
	"""Paths"""
	root_path = training_params['root_path']
	data_path = f'{root_path}/data'
	log_path = f'{root_path}/logs'
	
	# Get data file name
	filenames = os.listdir(data_path)
	if len(filenames)>1:
		raise Exception('More than 1 data file found')
	else:
		data_file_name = filenames[0]
	
	chkpt_path = f'{root_path}/checkpoints'
        
	# Folder for backup
	if training_restart:
		os.system(f'rm -r /{log_path}')
		os.system(f'rm -r /{chkpt_path}')
	if not os.path.exists(f'{log_path}'):
		os.makedirs(f'{log_path}/images')
	if not os.path.exists(f'{chkpt_path}'):
		os.makedirs(f'{chkpt_path}')
        
	# Folder for backup
	if not os.path.exists(f'{root_path}/backup'):
		os.makedirs(f'{root_path}/backup/checkpoints')
		os.makedirs(f'{root_path}/backup/logs')
        
	"""Initialize callbacks"""
	# Logger
	use_wandb = False
	if use_wandb:
		wandb.login()
		wandb_logger = WandbLogger(
				project='cosmo_project',
				name=f'{root_path.split("/")[-1]}'
				)
		wandb_logger.experiment.config.update(training_params)

	# Checkpoint
	checkpoint_callback = ModelCheckpoint(
		monitor=None,
		dirpath=f'{chkpt_path}',
		filename='{epoch:04d}',
		every_n_epochs=20,
		save_top_k=-1,
		save_last=True,
		enable_version_counter=False
	)
        
	"""Initialize seed"""
	pl.seed_everything(training_seed)

	"""Training"""
	if __name__ == 'code_model.train':
		# Load data
		data = BlobDataModule(
			data_file=f'{data_path}/{data_file_name}',
			batch_size=batch_size, num_workers=num_workers,
			transforms=data_transforms
		)
			
		# Load model
		model = model_dict[model_version](**training_params)
		print('done initiating model')
		
		# Transfer scaling factor from data module to model
		model.scaling_factor = data.scaling_factor
		
		# Save logged params
		np.savez(f'{log_path}/logged_params.npz', scaling_factor=data.scaling_factor)
		
		# Initialize trainer
		if avail_gpus<2:
			trainer = pl.Trainer(
				max_epochs=max_epochs,
				logger=wandb_logger if use_wandb else None,
				callbacks=[checkpoint_callback],
				log_every_n_steps=data.num_training_batches(),
			)
		elif avail_gpus>=2: # DDP for multi gpu 
			trainer = pl.Trainer(
				max_epochs=max_epochs,
				logger=wandb_logger if use_wandb else None,
				callbacks=[checkpoint_callback],
				log_every_n_steps=data.num_training_batches(),
				devices=avail_gpus, strategy='ddp',
			)
					
		print('trainer initiated')
			
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
						ckpt_path=f'{chkpt_path}/last.ckpt'
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
			
		# Plot losses
		if 'GAN' in training_params['model_version']:
			logs_plotter = ganLogsPlotter(dataset)
			logs_plotter.plot_logs(BlobDataModule, model_dict, testing_restart=True)
		elif 'Diffusion' in training_params['model_version']:
			logs_plotter = diffLogsPlotter(dataset)
			logs_plotter.plot_logs(testing_restart=True)