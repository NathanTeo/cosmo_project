import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys

"""RUN"""
run = 'cwgan_6b'


#################################################################################
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
        print(run)
        print('loading data...')
        super().__init__(run)
        
        """Initialize"""
        self.run = run

        self.root_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_runs'
        self.plot_save_path = f'/Users/Idiot/Desktop/Research/OFYP/cosmo/misc_plots/residual/{run}'

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
        self.gen_color = 'tab:orange'
        self.image_file_format = 'png'
        
        """Load"""
        self.load_generated_samples()
        
        """Get residuals"""
        print("getting residuals...")
        self.get_residuals()
        
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
        if len(centers)==0:
            return np.zeros((self.image_size, self.image_size))
        else:
            return np.array([normalize_2d(self._make_gaussian(coord, self.blob_size, self.image_size))*self.blob_amplitude
                               for coord in centers]).sum(axis=0)
        
    # def realize_gaussian_from_points(self, samples_centers):
        # return np.array([self._create_blobs(centers) for centers in samples_centers])
        
    def get_residuals(self):
        """Calculate residuals"""
        # Realize gaussian sample from generated blob center coordinates
        self.gen_to_gaussian_samples = np.array([self._create_blobs(centers) for centers in tqdm(self.gen_blob_coords)])
        
        # Get residuals
        self.gen_residuals = self.gen_to_gaussian_samples - self.gen_samples
        
    def plot_residual_samples(self, num_plots=5, grid_row_num=2):
        for n in range(num_plots):    
            # Get images
            gen_sample_subset = self.gen_samples[n*(grid_row_num**2):(n+1)*(grid_row_num**2)]
            gen_res_subset = self.gen_residuals[n*(grid_row_num**2):(n+1)*(grid_row_num**2)]
                
            'Images'
            # Plotting grid of images
            fig = plt.figure(figsize=(6,3))
            subfig = fig.subfigures(1, 2, wspace=0.2)
            
            subplots_samp = plot_img_grid(subfig[0], gen_sample_subset, grid_row_num, title='Generated Samples',
                                          vmin=-0.05, vmax=np.max(gen_sample_subset))
            subplots_res = plot_img_grid(subfig[1], gen_res_subset, grid_row_num, title='Residuals', 
                          cmap='RdBu', vmin=-0.07, vmax=0.07)
            
            fig.subplots_adjust(bottom=0.2)
            cbar_samp_ax = subfig[0].add_axes([0.1, 0.1, 0.8, 0.05])
            cbar_res_ax = subfig[1].add_axes([0.1, 0.1, 0.8, 0.05])
            subfig[0].colorbar(subplots_samp[0][0], cax=cbar_samp_ax, orientation='horizontal')
            subfig[1].colorbar(subplots_res[0][0], cax=cbar_res_ax, orientation='horizontal')
            
            # Save plot
            # plt.savefig(f'{self.plot_save_path}/residual_{n}_{self.model_name}.{self.image_file_format}')
            plt.savefig(f'{self.plot_save_path}/residual_{n}.{self.image_file_format}')
            plt.close()
        
    def plot_residual_pixel_histogram(self):
        'Single image histogram'
        histogram_num = 20
        gen_residuals_subset_sh = self.gen_residuals[:histogram_num]

        # Create figure
        fig, ax = plt.subplots(1)

        # Plot
        plot_pixel_histogram(ax, gen_residuals_subset_sh, color=(self.gen_color,0.5), bins=20, logscale=False)

        # Format
        plt.ylabel('pixel count')
        plt.xlabel('pixel value')
        fig.suptitle(f"Histogram of residual pixel values, {histogram_num} samples")
        plt.tight_layout()

        # Save
        # plt.savefig(f'{self.plot_save_path}/residual-pixel-histogram_{self.model_name}.{self.image_file_format}')
        plt.savefig(f'{self.plot_save_path}/residual-pixel-histogram.{self.image_file_format}')
        plt.close()

        'Stacked histogram'
        print('stacking histograms...')
        gen_residuals_hist_stack = stack_histograms(self.gen_residuals, bins=np.linspace(-0.1, 0.1, 40))

        # Create figure
        fig, ax = plt.subplots()

        # Plot
        plot_histogram_stack(ax, *gen_residuals_hist_stack, color=(self.real_color,0.8), logscale=False)

        # Format
        plt.ylabel('pixel count')
        plt.xlabel('pixel value')
        fig.suptitle(f"Stacked histogram of residual pixel values, {self.subset_sample_num} samples")
        plt.legend()
        plt.tight_layout()

        # Save
        # plt.savefig(f'{self.plot_save_path}/residual-pixel-stack-histogram_{self.model_name}.{self.image_file_format}')
        plt.savefig(f'{self.plot_save_path}/residual-pixel-stack-histogram.{self.image_file_format}')
        plt.close()
    
    def plot_residual_stack(self):
        'Stack'
        print('stacking...')
        stacked_gen_residuals = stack_imgs(self.gen_residuals)

        'Plot image'
        # Create figure
        fig, ax = plt.subplots(1)

        # Plot
        plot_stacked_imgs(ax, stacked_gen_residuals)

        # Format
        
        fig.suptitle('Stacked residual image')
        plt.tight_layout()

        # Save plot
        # plt.savefig(f'{self.plot_save_path}/residual-stack_{self.model_name}.{self.image_file_format}')
        plt.savefig(f'{self.plot_save_path}/residual-stack.{self.image_file_format}')
        plt.close()
    
    def test(self):
        print("plotting...")
        self.plot_residual_samples()
        self.plot_residual_pixel_histogram()
        self.plot_residual_stack()
        print("complete")


if __name__=="__main__":
    tester = residualTester(run)
    tester.test()
    