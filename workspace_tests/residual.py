import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys

project_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_project'
sys.path.append(project_path)
from code_model.models import model_dict
from code_model.testers.eval_utils import *
from code_model.testers.plotting_utils import *

class dataLoader():
    def __init__(self, run):
        """Initialize"""
        self.run = run

        self.root_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_runs'
        self.real_data_path = f'{self.root_path}/{run}/data'
        self.output_path = f'{self.root_path}/{run}/plots/model_output'
        
    def load_generated_samples(self):
        """Load in generated samples and counts"""
        # Load gan samples
        filenames = os.listdir(f'{self.output_path}')
        sample_file = [file for file in filenames if 'last' in file][0]
        self.gen_samples = np.load(f'{self.output_path}/{sample_file}', allow_pickle=True)
        
        # Load real samples
        file = os.listdir(f'{self.real_data_path}')[0]
        self.real_samples = np.load(f'{self.real_data_path}/{file}')[:len(self.gen_samples)]
        self.subset_sample_num = len(self.real_samples)
        
        # Load model counts and coordinates
        counts = np.load(f'{self.output_path}/counts.npz', allow_pickle=True)
        
        self.real_blob_counts = counts['real_counts']
        self.real_blob_coords = counts['real_coords'].tolist()         
        self.gen_blob_counts = counts['gen_counts'][-1]
        self.gen_blob_coords = counts['gen_coords'].tolist()[-1]
        
        # Calculate means
        self.real_blob_count_mean = np.mean(self.real_blob_counts)
        self.gen_blob_count_mean = np.mean(self.gen_blob_counts)
        
class residualTester(dataLoader):
    def __init__(self, run):
        super().__init__(run)
        
        """Initialize"""
        self.run = run

        self.root_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_runs'
        self.plot_save_path = f'/Users/Idiot/Desktop/Research/OFYP/cosmo/misc_plots/residual'

        if not os.path.exists(self.plot_save_path):
            os.makedirs(self.plot_save_path)
        
        sys.path.append(self.root_path)
        
        config = __import__(f'{run}.config.model_params')
        self.training_params = config.config.model_params.training_params
        self.generation_params = config.config.model_params.generation_params
        
        self.blob_size = self.generation_params['blob_size']
        self.image_size = self.generation_params['image_size']
        self.blob_count = self.generation_params['blob_num']
        self.blob_amplitude = 1/self.blob_count
        
        self.training_params['avail_gpus'] = torch.cuda.device_count()
        self.training_params['root_path'] = f"C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_runs/{run}"
        
        'Plotting'
        self.real_color = 'black'
        self.model_color = 'tab:orange'
        self.image_file_format = 'png'
        
        """Load"""
        self.load_generated_samples()
        
    def _make_gaussian(self, center, var, image_size):
        """
        Make a square gaussian kernel
        """
        x = np.arange(0, image_size, 1, float)
        y = x[:,np.newaxis]

        x0 = center[1]
        y0 = center[0]

        return np.exp(-0.5 * ((x-x0)**2 + (y-y0)**2) / var)

    def _create_blobs(self, centers):
        """
        Create a sample of gaussian blobs
        """
        return np.array([normalize_2d(self._make_gaussian(coord, self.blob_size, self.image_size))*self.blob_amplitude
                            for coord in centers]).sum(axis=0)
        
    # def realize_gaussian_from_points(self, samples_centers):
        # return np.array([self._create_blobs(centers) for centers in samples_centers])
        
    def get_residuals(self):
        """Calculate residuals"""
        # Realize gaussian sample from generated blob center coordinates
        self.gen_to_gaussian_samples = np.array([self._create_blobs(centers) for centers in self.gen_blob_coords])
        
        # Get residuals
        self.gen_residuals = self.gen_to_gaussian_samples - self.gen_samples
        
    def plot_residual_samples(self):
        pass 