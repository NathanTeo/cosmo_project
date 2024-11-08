"""


"""

import time as t
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

project_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_project'
sys.path.append(project_path)
from code_model.models import model_dict
from code_model.testers.eval_utils import *
from code_model.testers.plotting_utils import *
from code_data.utils import *

"""RUNS"""
model_run = 'diffusion_3c'

##########################################################
class Utils():
    def __init__(self, model_run):
        """Initialize"""
        self.model_run = model_run

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
        self.generation_params['pad'] = 0
        
        self.model_training_params['avail_gpus'] = torch.cuda.device_count()
        self.model_training_params['root_path'] = f"C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_runs/{model_run}"
        
        'Plotting'
        self.real_color = 'black'
        self.model_color = 'tab:blue'
        self.image_file_format = 'png'
        
        
    def load_models(self):
        """Load models"""
        self.model = model_dict[self.model_training_params['model_version']].load_from_checkpoint(
            f'{self.model_chkpt_path}/last.ckpt',
            **self.model_training_params
        )
        
        self.model_size = self.model.count_learnable_params()
    
    def load_generated_samples(self):
        """Load in generated samples"""
        # Load modelusion samples
        filenames = os.listdir(f'{self.model_output_path}')
        sample_file = [file for file in filenames if 'last' in file][0]
        self.model_samples = np.load(f'{self.model_output_path}/{sample_file}', allow_pickle=True)
        
        # Load real samples
        file = os.listdir(f'{self.real_data_path}')[0]
        self.real_samples = np.load(f'{self.real_data_path}/{file}')[:len(self.model_samples)]
        self.subset_sample_num = len(self.real_samples)
    
    def stand_speed(self, num_of_samples):
        """Speed of standard method of generating dataset --> method used to create real dataset"""
        generator = blobDataset(**self.generation_params)
        start = t.time()
        for _ in range(num_of_samples):    
            generator.realize_sample(generator.center_generator)
        end = t.time()
        return end - start
                
    def gan_speed(self, num_of_samples):
        """Record gan generation speed"""
        start = t.time()
        z = torch.randn(num_of_samples, self.model_training_params['network_params']['latent_dim'])
        self.model(z)
        end = t.time()
        return end - start
    
    def diff_speed(self, num_of_samples):
        """Record diffusion generation speed"""
        start = t.time()
        self.model.sample(self.model.ema_network, n=num_of_samples)
        end = t.time()
        return end - start

class speedModule(Utils):
    """Class used to test the generation speed of standard methods and models"""
    def __init__(self, model_run):
        super().__init__(model_run)
        
        print(f'run {model_run}')
        
        print('loading models and data...')
        self.load_models()
        self.load_generated_samples()
        print('loading complete')
        
        print('')
        print('model sizes, in number of learnable parameters')
        print(f'diffusion -  {millify(self.model_size)}')
        print('')
        print(f'loaded sample size \nreal: {self.real_samples.shape} | model: {self.model_samples.shape}')
        print('')
        
    def test_speed(self, num_of_samples):
        """Test the generation speed"""
        print(f"cpu | {num_of_samples} samples")
        print('--------------------')
        
        print('generating samples...')
        stand_time = self.stand_speed(num_of_samples)
        if 'gan' in self.model_run:
            model_time = self.gan_speed(num_of_samples)
        elif 'diff' in self.model_run:
            model_time = self.diff_speed(num_of_samples)
        
        print(f'standard: {stand_time:.4f}s | model: {model_time:.4f}s')
        print(f'Standard method is {model_time/stand_time:.0f} times faster')
    
    def run_tests(self):
        print('running all tests...')
        print()
        
        print('generation speed')
        self.test_speed(10)
        print()
        
if __name__=="__main__":
    tester = speedModule(model_run)
    tester.run_tests()