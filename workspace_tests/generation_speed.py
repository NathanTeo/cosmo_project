"""
Author: Nathan Teo

This script compares the generation speed of 
- the algorithm used to create the real dataset
- the neural network  
"""

import time as t
import sys
import torch
from utils.load_model import modelLoader

project_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_project'
sys.path.append(project_path)

from code_model.testers.eval_utils import *
from code_model.testers.plotting_utils import *
from code_data.utils import *

"""RUNS"""
model_run = 'diffusion_7e'

##########################################################

class speedModule(modelLoader):
    """Class used to test the generation speed of standard methods and models"""
    def __init__(self, model_run):
        super().__init__(model_run)
        
        print(f'run {model_run}')
        
        print('loading models and data...')
        self.load_model('last.ckpt')
        self.load_samples()
        print('loading complete')
        
        print('')
        print('model sizes, in number of learnable parameters')
        print(f'diffusion -  {millify(self.model_size)}')
        print('')
        print(f'loaded sample size \nreal: {self.real_samples.shape} | model: {self.model_samples.shape}')
        print('')
        
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
    
    def stand_speed(self, num_of_samples):
        """Speed of standard method of generating dataset --> method used to create real dataset"""
        generator = blobDataset(**self.generation_params)
        start = t.time()
        for _ in range(num_of_samples):    
            generator.realize_sample(seed=None)
        end = t.time()
        return end - start
                
    def test_speed(self, num_of_samples):
        """Test the generation speed"""
        print(f"cpu | {num_of_samples} samples")
        print('--------------------')
        
        print('generating samples...')
        stand_time = self.stand_speed(num_of_samples)
        print(f'complete, {stand_time:.4f}s')
        
        if 'gan' in self.model_run:
            model_time = self.gan_speed(num_of_samples)
        elif 'diff' in self.model_run:
            model_time = self.diff_speed(num_of_samples)
        print(f'complete, {model_time:.4f}s')
        
        print(f'standard: {stand_time:.4f}s | model: {model_time:.4f}s')
        print(f'Standard method is {model_time/stand_time:.0f} times faster')
    
    def run_tests(self):
        print('running all tests...')
        print()
        
        print('generation speed')
        self.test_speed(3)
        print()
        
if __name__=="__main__":
    tester = speedModule(model_run)
    tester.run_tests()