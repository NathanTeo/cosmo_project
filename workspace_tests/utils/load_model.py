import os
import sys
import numpy as np
import torch

project_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_project'
sys.path.append(project_path)
from code_model.testers.eval_utils import init_param

class modelLoader():
    def __init__(self, model_run, epoch=None,
                 real_color='black', model_color='tab:blue', file_image_format='png'):
        """Initialize"""
        self.model_run = model_run
        self.epoch = epoch

        self.root_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_runs'
        self.real_data_path = f'{self.root_path}/{model_run}/data'
        self.model_chkpt_path = f'{self.root_path}/{model_run}/checkpoints'
        self.model_output_path = f'{self.root_path}/{model_run}/plots/model_output'
        self.plot_save_path = f'/Users/Idiot/Desktop/Research/OFYP/cosmo/misc_plots/speed/{model_run}'

        if not os.path.exists(self.plot_save_path):
            os.makedirs(self.plot_save_path)
        
        sys.path.append(self.root_path)
    
        model_config = __import__(f'{model_run}.config.model_params')
        self.model_training_params = model_config.config.model_params.training_params
        
        self.generation_params = model_config.config.model_params.generation_params
        self.image_size = self.generation_params['image_size']
        self.blob_size = self.generation_params['blob_size']
        self.blob_amplitude = self.generation_params['blob_amplitude']
        self.minimum_distance = init_param(self.generation_params, 'minimum_distance')
        self.generation_params['pad'] = 0
        
        self.model_training_params['avail_gpus'] = torch.cuda.device_count()
        self.model_training_params['root_path'] = f"C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_runs/{model_run}"
        self.project_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_project'
        
        sys.path.append(self.project_path)
        from code_model.models import model_dict
        self.model_dict = model_dict
        
        'Plotting'
        self.real_color = real_color
        self.model_color = model_color
        self.image_file_format = file_image_format
        
        
    def load_model(self, checkpoint_file):
        """Load a single model, this model will be stored for a child object"""
        self.model = self.model_dict[self.model_training_params['model_version']].load_from_checkpoint(
            f'{self.model_chkpt_path}/{checkpoint_file}',
            **self.model_training_params
        )
        
        self.model_size = self.model.count_learnable_params()

        return self.model
        
    def load_models(self, checkpoint_files):
        """Loads models from a list of filenames"""
        return [self.load_model(file) for file in checkpoint_files]
    
    def load_generated_samples(self):
        """Load in generated samples"""
        # Load model samples
        filenames = os.listdir(f'{self.model_output_path}')
        if self.epoch is not None:    
            sample_file = [file for file in filenames if f'{self.epoch:04d}' in file][0]
        else:
            sample_file = filenames[-1]
        self.model_samples = np.load(f'{self.model_output_path}/{sample_file}', allow_pickle=True)
        print(sample_file)
        
    def load_real_samples(self):
        # Load real samples
        file = os.listdir(f'{self.real_data_path}')[0]
        self.real_samples = np.load(f'{self.real_data_path}/{file}')[:len(self.model_samples)]
        self.subset_sample_num = len(self.real_samples)
        
    def load_samples(self):
        self.load_generated_samples()
        self.load_real_samples()
        
    def load_counts(self):
        """Load counts"""
        file = np.load('{}/counts.npz'.format(self.model_output_path), allow_pickle=True)
        
        self.model_blob_counts = file['gen_counts'][-1]
        self.model_blob_coords = file['gen_coords'].tolist()[-1]
        self.real_blob_counts = file['real_counts']
        self.real_blob_coords = file['real_coords'].tolist()
        
        self.real_blob_num_mean = np.mean(self.real_blob_counts)
        self.model_blob_num_mean = np.mean(self.model_blob_counts)
        
    def generate(self, model, num_of_samples):
        """Generate samples"""
        if 'gan' in self.model_run:
            z = torch.randn(num_of_samples, self.model_training_params['network_params']['latent_dim'])
            samples = model(z)
        elif 'diffusion' in self.model_run:
            samples = model.sample(self.model.ema_network, n=num_of_samples)
        
        return samples.detach().squeeze().numpy()