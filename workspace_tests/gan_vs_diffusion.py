"""
Author: Nathan Teo

This script compares the result of two models
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

"""RUNS"""
gan_run = 'cwgan_6b'
diffusion_run = 'diffusion_3d'

##########################################################
class compareUtils():
    def __init__(self, gan_run, diff_run):
        """Initialize"""
        self.gan_run = gan_run
        self.diff_run = diff_run

        self.root_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_runs'
        self.real_data_path = f'{self.root_path}/{gan_run}/data'
        self.gan_chkpt_path = f'{self.root_path}/{gan_run}/checkpoints'
        self.diff_chkpt_path = f'{self.root_path}/{diff_run}/checkpoints'
        self.gan_output_path = f'{self.root_path}/{gan_run}/plots/model_output'
        self.diff_output_path = f'{self.root_path}/{diff_run}/plots/model_output'
        self.plot_save_path = f'/Users/Idiot/Desktop/Research/OFYP/cosmo/misc_plots/gan-v-diff/{gan_run}-v-{diff_run}'

        if not os.path.exists(self.plot_save_path):
            os.makedirs(self.plot_save_path)
        
        sys.path.append(self.root_path)
        
        gan_config = __import__(f'{gan_run}.config.model_params')
        self.gan_training_params = gan_config.config.model_params.training_params
        
        diff_config = __import__(f'{diff_run}.config.model_params')
        self.diff_training_params = diff_config.config.model_params.training_params
        
        generation_params = diff_config.config.model_params.generation_params
        self.image_size = generation_params['image_size']
        self.blob_size = generation_params['blob_size']
        self.blob_amplitude = 1/generation_params['blob_num'] 
        
        self.gan_training_params['avail_gpus'] = torch.cuda.device_count()
        self.gan_training_params['root_path'] = f"C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_runs/{gan_run}"
        
        self.diff_training_params['avail_gpus'] = torch.cuda.device_count()
        self.diff_training_params['root_path'] = f"C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_runs/{diff_run}"
        
        'Plotting'
        self.real_color = 'black'
        self.gan_color = 'tab:orange'
        self.diff_color = 'tab:blue'
        self.image_file_format = 'png'
        
        
    def load_models(self):
        """Load models"""
        self.gan = model_dict[self.gan_training_params['model_version']].load_from_checkpoint(
            f'{self.gan_chkpt_path}/last.ckpt',
            **self.gan_training_params
        )

        self.diff = model_dict[self.diff_training_params['model_version']].load_from_checkpoint(
            f'{self.diff_chkpt_path}/last.ckpt',
            **self.diff_training_params
        )
        
        self.gan_size = self.gan.count_learnable_params()
        self.diff_size = self.diff.count_learnable_params()
    
    def load_generated_samples(self):
        """Load in generated samples and counts"""
        # Load gan samples
        filenames = os.listdir(f'{self.gan_output_path}')
        sample_file = [file for file in filenames if 'last' in file][0]
        self.gan_samples = np.load(f'{self.gan_output_path}/{sample_file}', allow_pickle=True)
        
        # Load diffusion samples
        filenames = os.listdir(f'{self.diff_output_path}')
        sample_file = [file for file in filenames if 'last' in file][0]
        self.diff_samples = np.load(f'{self.diff_output_path}/{sample_file}', allow_pickle=True)
        
        # Load real samples
        file = os.listdir(f'{self.real_data_path}')[0]
        self.real_samples = np.load(f'{self.real_data_path}/{file}')[:len(self.gan_samples)]
        self.subset_sample_num = len(self.real_samples)
        
        # Load gan counts and coordinates
        gan_counts = np.load(f'{self.gan_output_path}/counts.npz', allow_pickle=True)
        
        self.real_blob_counts = gan_counts['real_counts']
        self.real_blob_coords = gan_counts['real_coords'].tolist()         
        self.gan_blob_counts = gan_counts['gen_counts'][-1]
        self.gan_blob_coords = gan_counts['gen_coords'].tolist()[-1]

        # Load diffusion counts and coordinates, real dataset should be the same --- loading is not repeated
        diff_counts = np.load(f'{self.diff_output_path}/counts.npz', allow_pickle=True)

        self.diff_blob_counts = diff_counts['gen_counts'][-1]
        self.diff_blob_coords = diff_counts['gen_coords'].tolist()[-1]
        
        # Calculate means
        self.real_blob_count_mean = np.mean(self.real_blob_counts)
        self.gan_blob_count_mean = np.mean(self.gan_blob_counts)
        self.diff_blob_count_mean = np.mean(self.diff_blob_counts)
        
        # Get residuals
        print('finding residuals...')
        self.gan_residuals = get_residuals(self.gan_samples, self.gan_blob_coords, 
                                           self.image_size, self.blob_size, self.blob_amplitude)
        self.diff_residuals = get_residuals(self.diff_samples, self.diff_blob_coords,
                                            self.image_size, self.blob_size, self.blob_amplitude)
                
    def gan_speed(self, num_of_samples):
        """Record gan generation speed"""
        start = t.time()
        z = torch.randn(num_of_samples, self.gan_training_params['network_params']['latent_dim'])
        self.gan(z)
        end = t.time()
        return end - start
    
    def diff_speed(self, num_of_samples):
        """Record diffusion generation speed"""
        start = t.time()
        self.diff.sample(self.diff.ema_network, n=num_of_samples)
        end = t.time()
        return end - start
    
    def plot_samples(self, grid_row_num=2):
        'Images'
        real_samples_subset = self.real_samples[:grid_row_num**2]
        gan_samples_subset = self.gan_samples[:grid_row_num**2]
        diff_samples_subset = self.diff_samples[:grid_row_num**2]
        
        # Plotting grid of images
        fig = plt.figure(figsize=(7,3))
        subfig = fig.subfigures(1, 3, wspace=0.3)
        
        plot_img_grid(subfig[0], real_samples_subset, grid_row_num, title='Real', wspace=0.1)
        plot_img_grid(subfig[1], gan_samples_subset, grid_row_num, title='GAN', wspace=0.1) 
        plot_img_grid(subfig[2], diff_samples_subset, grid_row_num, title='Diffusion', wspace=0.1)
        
        fig.tight_layout()
        
        # Save plot
        plt.savefig(f'{self.plot_save_path}/samples.{self.image_file_format}')
        plt.close()

        'Marginal sums'
        # Plotting marginal sums
        real_marginal_sums = [marginal_sums(real_samples_subset[i]) for i in range(grid_row_num**2)]
        gan_marginal_sums = [marginal_sums(gan_samples_subset[i]) for i in range(grid_row_num**2)]
        diff_marginal_sums = [marginal_sums(diff_samples_subset[i]) for i in range(grid_row_num**2)]

        fig = plt.figure(figsize=(8,6))
        subfig = fig.subfigures(1, 3)
        
        plot_marginal_sums(real_marginal_sums, subfig[0], grid_row_num, title='Real')
        plot_marginal_sums(gan_marginal_sums, subfig[1], grid_row_num, title='GAN')
        plot_marginal_sums(diff_marginal_sums, subfig[2], grid_row_num, title='Diffusion')
        
        # Format
        fig.suptitle('Marginal Sums')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'{self.plot_save_path}/marg-sums.{self.image_file_format}')
        plt.close()
        
        'FFT'
        # Fourier transform images
        real_ffts = fourier_transform_samples(real_samples_subset)
        gan_ffts = fourier_transform_samples(gan_samples_subset)
        diff_ffts = fourier_transform_samples(diff_samples_subset)

        # Plotting fft
        fig = plt.figure(figsize=(8,3))
        subfig = fig.subfigures(1, 3, wspace=0.2)
        
        plot_img_grid(subfig[0], real_ffts, grid_row_num, title='Real FFT')
        plot_img_grid(subfig[1], gan_ffts, grid_row_num, title='GAN FFT')
        plot_img_grid(subfig[2], diff_ffts, grid_row_num, title='Diffusion FFT')
        
        # Save plot
        plt.savefig(f'{self.plot_save_path}/fft.{self.image_file_format}')
        plt.close()
            
    def plot_count_histogram(self, ignore_outliers=True):
        # Create figure
        fig = plt.figure(figsize=(4,3))

        # Bins for histogram
        concat_fluxes = np.concatenate([self.real_blob_counts, self.gan_blob_counts, self.diff_blob_counts])
        if ignore_outliers:
            bin_min = np.floor(np.percentile(concat_fluxes, 1))
            bin_max = np.ceil(np.percentile(concat_fluxes, 99))
        else:        
            bin_min = np.min(concat_fluxes)
            bin_max = np.min(concat_fluxes)

        bins = np.arange(bin_min-1.5, bin_max+1.5,1)
        
        # Plot histogram
        plt.hist(self.real_blob_counts, bins=bins,
                histtype='step', label=f'Real',
                color=(self.real_color, 0.8)
                )
        plt.axvline(self.real_blob_count_mean, color=(self.real_color,0.5), linestyle='dashed', linewidth=1) # Only label mean for last model
        
        plt.hist(self.gan_blob_counts, bins=bins, 
                    histtype='step', label='GAN', color=(self.gan_color,0.8))
        plt.axvline(self.gan_blob_count_mean, color=(self.gan_color,0.5), linestyle='dashed', linewidth=1)
 
        plt.hist(self.diff_blob_counts, bins=bins, 
                    histtype='step', label='Diffusion', color=(self.diff_color,0.8))
        plt.axvline(self.diff_blob_count_mean, color=(self.diff_color,0.5), linestyle='dashed', linewidth=1)


        _, max_ylim = plt.ylim()
        plt.text(self.real_blob_count_mean*1.05, max_ylim*0.3,
                'Mean: {:.2f}'.format(self.real_blob_count_mean), color=(self.real_color,1))
        plt.text(self.gan_blob_count_mean*1.05, max_ylim*0.2,
                'Mean: {:.2f}'.format(self.gan_blob_count_mean), color=(self.gan_color,1))
        plt.text(self.diff_blob_count_mean*1.05, max_ylim*0.1,
                'Mean: {:.2f}'.format(self.diff_blob_count_mean), color=(self.diff_color,1))
        
        # Format
        plt.ylabel('image count')
        plt.xlabel('blob count')
        plt.suptitle(f"Histogram of blob count, {len(self.real_blob_counts)} samples")
        plt.legend()
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/count-histogram.{self.image_file_format}')
        plt.close()
    
    def plot_stack_images(self):
        """Stack images"""
        'Stack'
        stacked_real_img = stack_imgs(self.real_samples)
        stacked_gan_img = stack_imgs(self.gan_samples)
        stacked_diff_img = stack_imgs(self.diff_samples)

        'Plot image'
        # Create figure
        fig, axs = plt.subplots(1, 3, figsize=(4,2.5))

        # Plot
        plot_stacked_imgs(axs[0], stacked_real_img, title=f"Real\n{self.subset_sample_num} samples")
        plot_stacked_imgs(axs[1], stacked_gan_img, title=f"GAN\n{self.subset_sample_num} samples")
        plot_stacked_imgs(axs[2], stacked_diff_img, title=f"Diffusion\n{self.subset_sample_num} samples")
        
        # Format
        fig.suptitle('Stacked image')
        plt.tight_layout()

        # Save plot
        plt.savefig(f'{self.plot_save_path}/stack.{self.image_file_format}')
        plt.close()
        
        'Stacked image histogram'
        # Create figure
        fig = plt.figure(figsize=(4,3))

        # Plot
        plt.hist(stacked_real_img.ravel(), histtype='step', label='Real', color=(self.real_color,0.8))
        plt.hist(stacked_gan_img.ravel(), histtype='step', label='GAN', color=(self.gan_color,0.8))
        plt.hist(stacked_diff_img.ravel(), histtype='step', label='Diffusion', color=(self.diff_color,0.8))
        
        # Format
        plt.ylabel('pixel count')
        plt.xlabel('stacked pixel value')
        plt.suptitle(f"Stack of {self.subset_sample_num} samples")
        plt.legend()
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/stack-pixel-histogram.{self.image_file_format}')
        plt.close()
        
    def plot_pixel_histogram(self, max_sample_size=500):
        real_samples_subset = self.real_samples[:max_sample_size]
        gan_samples_subset = self.gan_samples[:max_sample_size]
        diff_samples_subset = self.diff_samples[:max_sample_size]
        
        'Stacked histogram'
        real_hist_stack = stack_histograms(real_samples_subset)
        gan_hist_stack = stack_histograms(gan_samples_subset)
        diff_hist_stack = stack_histograms(diff_samples_subset)

        # Create figure
        fig, ax = plt.subplots()

        # Plot
        plot_histogram_stack(ax, *real_hist_stack, color=(self.real_color,0.8), label='Real')
        plot_histogram_stack(ax, *gan_hist_stack, color=(self.gan_color,0.8), label='GAN')
        plot_histogram_stack(ax, *diff_hist_stack, color=(self.diff_color,0.8), label='Diff')

        # Format
        plt.ylabel('image count')
        plt.xlabel('pixel value')
        fig.suptitle(f"Stacked histogram of pixel values, {len(real_samples_subset)} samples")
        plt.legend()
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/pixel-histogram.{self.image_file_format}')
        plt.close()
    
    def plot_samples_with_extreme_count(self):
        'Extreme number of blobs'
        k=1
        # Create figure
        fig = plt.figure(figsize=(4,2*k+0.2))
        subfig = fig.subfigures(1, 3, wspace=0.2)

        # Plot
        plot_extremum_num_blobs(
            subfig[0], self.real_samples,
            self.real_blob_coords, self.real_blob_counts,
            extremum='min', title='Real', title_y=0.87, k=k
            )
        plot_extremum_num_blobs(
            subfig[1], self.gan_samples,
            self.gan_blob_coords, self.gan_blob_counts,
            extremum='min', title='GAN', title_y=0.87, k=k
            )  
        plot_extremum_num_blobs(
            subfig[2], self.diff_samples,
            self.diff_blob_coords, self.diff_blob_counts,
            extremum='min', title='Diffusion', title_y=0.87, k=k
            )  
        # Format
        fig.suptitle(f"Min blob count, {self.subset_sample_num} samples")
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/min.{self.image_file_format}')
        plt.close()

        # Create figure
        fig = plt.figure(figsize=(4,2*k+0.2))
        subfig = fig.subfigures(1, 3, wspace=0.2)

        # Plot
        plot_extremum_num_blobs(
            subfig[0], self.real_samples,
            self.real_blob_coords, self.real_blob_counts,
            extremum='max', title='Real', title_y=0.87, k=k
            )
        plot_extremum_num_blobs(
            subfig[1], self.gan_samples,
            self.gan_blob_coords, self.gan_blob_counts,
            extremum='max', title='GAN', title_y=0.87, k=k
            )  
        plot_extremum_num_blobs(
            subfig[2], self.diff_samples,
            self.diff_blob_coords, self.diff_blob_counts,
            extremum='max', title='Diffusion', title_y=0.87, k=k
            ) 
        # Format
        fig.suptitle(f"Max blobs count, {self.subset_sample_num} samples")
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/max.{self.image_file_format}')
        plt.close()

    def plot_total_flux(self, ignore_outliers=True):
        'Total flux histogram'
        # Find flux
        real_img_fluxes = find_total_fluxes(self.real_samples)
        gan_img_fluxes = find_total_fluxes(self.gan_samples)
        diff_img_fluxes = find_total_fluxes(self.diff_samples)

        # Create figure
        fig = plt.figure()
        
        # Bins for histogram
        concat_fluxes = np.concatenate([real_img_fluxes, gan_img_fluxes, diff_img_fluxes])
        if ignore_outliers:
            bin_min = np.floor(np.percentile(concat_fluxes, 1))
            bin_max = np.ceil(np.percentile(concat_fluxes, 99))
        else:        
            bin_min = np.min(concat_fluxes)
            bin_max = np.min(concat_fluxes)

        bins = np.arange(bin_min-3, bin_max+3,bin_max/50)

        # Plot
        plt.hist(real_img_fluxes, bins=bins, histtype='step', label='Real', color=(self.real_color,0.8))
        plt.hist(gan_img_fluxes, bins=bins, histtype='step', label='GAN', color=(self.gan_color,0.8))
        plt.hist(diff_img_fluxes, bins=bins, histtype='step', label='Diffusion', color=(self.diff_color,0.8))
 
        # Format   
        plt.ylabel('image count')
        plt.xlabel('total flux')
        plt.suptitle(f"Histogram of total flux, {self.subset_sample_num} samples")
        plt.legend()
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/total-flux-histogram.{self.image_file_format}')
        plt.close()
        
    def plot_residual_samples(self, num_plots=5, grid_row_num=2):
        for n in range(num_plots):    
            # Get images
            gan_sample_subset = self.gan_samples[n*(grid_row_num**2):(n+1)*(grid_row_num**2)]
            gan_res_subset = self.gan_residuals[n*(grid_row_num**2):(n+1)*(grid_row_num**2)]
            
            diff_sample_subset = self.diff_samples[n*(grid_row_num**2):(n+1)*(grid_row_num**2)]
            diff_res_subset = self.diff_residuals[n*(grid_row_num**2):(n+1)*(grid_row_num**2)]
                
            'Images'
            # For image max plot
            concat_subset = np.concatenate([gan_sample_subset, diff_sample_subset])
            
            # Plotting grid of images
            fig = plt.figure(figsize=(4.5, 6))
            subfig = fig.subfigures(4, 2, wspace=0.1, hspace=0.3, height_ratios=(0.5,5,5,1))
            
            subplots_samp = plot_img_grid(subfig[1,0], gan_sample_subset, grid_row_num,
                                        title='GAN samples', title_y=1.06,
                                        vmin=-0.05, vmax=np.max(concat_subset), wspace=0.05, hspace=0.01)
            subplots_res = plot_img_grid(subfig[1,1], gan_res_subset, grid_row_num, 
                                        title='GAN residuals', title_y=1.06, 
                                        cmap='RdBu', vmin=-0.07, vmax=0.07, wspace=0.05, hspace=0.01)
            
            subplots_samp = plot_img_grid(subfig[2,0], diff_sample_subset, grid_row_num,
                                        title='Diffusion samples', title_y=1.06,
                                        vmin=-0.05, vmax=np.max(concat_subset), wspace=0.05, hspace=0.01)
            subplots_res = plot_img_grid(subfig[2,1], diff_res_subset, grid_row_num,
                                        title='Diffusion residuals', title_y=1.06,
                                        cmap='RdBu', vmin=-0.07, vmax=0.07, wspace=0.05, hspace=0.01)
            
            cax0, empty0 = subfig[3,0].subplots(2)
            cax1, empty1 = subfig[3,1].subplots(2)
            blank_plot(empty0)
            blank_plot(empty1)
            fig.colorbar(subplots_samp[0][0], cax=cax0, orientation='horizontal')
            fig.colorbar(subplots_res[0][0], cax=cax1, orientation='horizontal')

            fig.tight_layout()
            
            # Save plot
            plt.savefig(f'{self.plot_save_path}/residual_{n}.{self.image_file_format}')
            plt.close()

class compareModule(compareUtils):
    def __init__(self, gan_run, diff_run):
        super().__init__(gan_run, diff_run)
        
        print(f'run {gan_run} vs run {diffusion_run}')
        
        print('loading models and data...')
        self.load_models()
        self.load_generated_samples()
        print('loading complete')
        
        print('')
        print('model sizes, in number of learnable parameters')
        print(f'gan - \n {millify(np.sum(self.gan_size))} (generator: {millify(self.gan_size[0])} | discrinimator: {millify(self.gan_size[1])})')
        print(f'diffusion -  {millify(self.diff_size)}')
        print('')
        print(f'loaded sample size \nreal: {self.real_samples.shape} | gan: {self.gan_samples.shape} | diff: {self.diff_samples.shape}')
        print('')
        
    def test_speed(self, num_of_samples):
        """Test the generation speed"""
        print(f"cpu | {num_of_samples} samples")
        print('--------------------')
        
        print('generating samples...')
        gan_time = self.gan_speed(num_of_samples)
        diff_time = self.diff_speed(num_of_samples)
        
        print(f'GAN: {gan_time:.4f}s | Diff: {diff_time:.4f}s')
        print(f'GAN is {diff_time/gan_time:.0f} times faster')
    
    def run_tests(self):
        print('running all tests...')
        print()
        
        print('generation speed')
        self.test_speed(3)
        print()
        
        print('plotting...')
        self.plot() 
        print()
    
    def plot(self):
        n = 7
        print(f'task 1/{n} | samples')
        self.plot_samples()
        print(f'task 2/{n} | count histogram')
        self.plot_count_histogram()
        print(f'task 3/{n} | stacking')
        self.plot_stack_images()
        print(f'task 4/{n} | pixel histogram')
        self.plot_pixel_histogram()
        print(f'task 5/{n} | extreme counts')
        self.plot_samples_with_extreme_count()
        print(f'task 6/{n} | total flux')
        self.plot_total_flux()
        print(f'task 7/{n} | residual samples')
        self.plot_residual_samples()
        
if __name__=="__main__":
    tester = compareModule(gan_run, diffusion_run)
    tester.plot()