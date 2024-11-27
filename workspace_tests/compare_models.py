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
model1_run = 'cwgan_8a'
model2_run = 'diffusion_3e'
labels = ['GAN', 'diffusion']
model_epochs = None # Uses largest epoch if None

###############################################################################################
class compareUtils():
    def __init__(self, model1_run, model2_run, labels, model_epochs=None):
        """Initialize"""
        self.model1_run = model1_run
        self.model2_run = model2_run

        self.root_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_runs'
        self.real_data_path = f'{self.root_path}/{model1_run}/data'
        self.model1_chkpt_path = f'{self.root_path}/{model1_run}/checkpoints'
        self.model2_chkpt_path = f'{self.root_path}/{model2_run}/checkpoints'
        self.model1_output_path = f'{self.root_path}/{model1_run}/plots/model_output'
        self.model2_output_path = f'{self.root_path}/{model2_run}/plots/model_output'
        self.plot_save_path = f'/Users/Idiot/Desktop/Research/OFYP/cosmo/misc_plots/model_comparison/{model1_run}-v-{model2_run}'

        if not os.path.exists(self.plot_save_path):
            os.makedirs(self.plot_save_path)
        
        sys.path.append(self.root_path)
        
        model1_config = __import__(f'{model1_run}.config.model_params')
        self.model1_training_params = model1_config.config.model_params.training_params
        
        model2_config = __import__(f'{model2_run}.config.model_params')
        self.model2_training_params = model2_config.config.model_params.training_params
        
        generation_params = model2_config.config.model_params.generation_params
        self.image_size = generation_params['image_size']
        self.blob_size = generation_params['blob_size']
        self.blob_amplitude = generation_params['blob_amplitude'] 
        
        self.model1_training_params['avail_gpus'] = torch.cuda.device_count()
        self.model1_training_params['root_path'] = f"C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_runs/{model1_run}"
        
        self.model2_training_params['avail_gpus'] = torch.cuda.device_count()
        self.model2_training_params['root_path'] = f"C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_runs/{model2_run}"
        
        'Plotting'
        self.real_color = 'black'
        self.model1_color = 'tab:orange'
        self.model2_color = 'tab:blue'
        self.image_file_format = 'png'
        self.labels = labels
    
        # Dictionary for logging metrics
        self.log_dict = {
            'model': [],
            'stat': [],
            'metric': [],
            'result': []
        }
        
        '''Get epochs'''
        # Get last epochs of each model
        if model_epochs is None:
            filenames1 = os.listdir(f'{self.model1_chkpt_path}')
            filenames1.remove('last.ckpt')
            filenames1.sort()
            filenames2 = os.listdir(f'{self.model2_chkpt_path}')
            filenames2.remove('last.ckpt')
            filenames2.sort()
            self.model_epochs = (filenames1[-1][-9,-5], filenames2[-1][-9,-5])
        # Use given model epochs
        else:
            self.model_epochs = model_epochs
            
    def log_in_dict(self, entry):
        """Log desired metrics in the initialized dictionary"""
        self.log_dict['model'].append(entry[0])
        self.log_dict['stat'].append(entry[1])
        self.log_dict['metric'].append(entry[2])
        self.log_dict['result'].append(entry[3])
    
    def get_log_dict(self):
        return self.log_dict
        
    def load_models(self):
        """Load models"""
        filenames = os.listdir(f'{self.model1_chkpt_path}')
        checkpoint_file = [file for file in filenames if str(self.model1_epoch) in file][0]
        self.model1 = model_dict[self.model1_training_params['model_version']].load_from_checkpoint(
            f'{self.model1_chkpt_path}/{checkpoint_file}',
            **self.model1_training_params
        )
        
        filenames = os.listdir(f'{self.model2_chkpt_path}')
        checkpoint_file = [file for file in filenames if str(self.model2_epoch) in file][0]
        self.model2 = model_dict[self.model2_training_params['model_version']].load_from_checkpoint(
            f'{self.model2_chkpt_path}/{checkpoint_file}',
            **self.model2_training_params
        )
        
        self.model1_size = self.model1.count_learnable_params()
        self.model2_size = self.model2.count_learnable_params()
    
    def load_generated_samples(self):
        """Load in generated samples and counts"""
        # Load model1 samples
        filenames = os.listdir(f'{self.model1_output_path}')
        sample_file = [file for file in filenames if str(self.model1_epoch) in file][0]
        self.model1_samples = np.load(f'{self.model1_output_path}/{sample_file}', allow_pickle=True)
        
        # Load model2 samples
        filenames = os.listdir(f'{self.model2_output_path}')
        sample_file = [file for file in filenames if str(self.model2_epoch) in file][0]
        self.model2_samples = np.load(f'{self.model2_output_path}/{sample_file}', allow_pickle=True)
        
        # Load real samples
        file = os.listdir(f'{self.real_data_path}')[0]
        self.real_samples = np.load(f'{self.real_data_path}/{file}')[:len(self.model1_samples)]
        self.subset_sample_num = len(self.real_samples)
        
        # Load model1 counts and coordinates
        model1_counts = np.load(f'{self.model1_output_path}/counts.npz', allow_pickle=True)
        
        self.real_blob_counts = model1_counts['real_counts']
        self.real_blob_coords = model1_counts['real_coords'].tolist()         
        self.model1_blob_counts = model1_counts['gen_counts'][-1]
        self.model1_blob_coords = model1_counts['gen_coords'].tolist()[-1]

        # Load model2 counts and coordinates, real dataset should be the same --- loading is not repeated
        model2_counts = np.load(f'{self.model2_output_path}/counts.npz', allow_pickle=True)

        self.model2_blob_counts = model2_counts['gen_counts'][-1]
        self.model2_blob_coords = model2_counts['gen_coords'].tolist()[-1]
        
        # Calculate means
        self.real_blob_count_mean = np.mean(self.real_blob_counts)
        self.model1_blob_count_mean = np.mean(self.model1_blob_counts)
        self.model2_blob_count_mean = np.mean(self.model2_blob_counts)
        
        # Get residuals
        print('finding residuals...')
        self.model1_residuals = get_residuals(self.model1_samples, self.model1_blob_coords, 
                                           self.image_size, self.blob_size, self.blob_amplitude)
        self.model2_residuals = get_residuals(self.model2_samples, self.model2_blob_coords,
                                            self.image_size, self.blob_size, self.blob_amplitude)
                
    def gan_speed(self, model, num_of_samples, latent_dim):
        """Record gan generation speed"""
        start = t.time()
        z = torch.randn(num_of_samples, latent_dim)
        model(z)
        end = t.time()
        return end - start
    
    def diff_speed(self, model, num_of_samples):
        """Record diffusion generation speed"""
        start = t.time()
        model.sample(model.ema_network, n=num_of_samples)
        end = t.time()
        return end - start
    
    def plot_samples(self, grid_row_num=2):
        'Images'
        real_samples_subset = self.real_samples[:grid_row_num**2]
        model1_samples_subset = self.model1_samples[:grid_row_num**2]
        model2_samples_subset = self.model2_samples[:grid_row_num**2]
        
        # Plotting grid of images
        fig = plt.figure(figsize=(7,3))
        subfig = fig.subfigures(1, 3, wspace=0.3)
        
        plot_img_grid(subfig[0], real_samples_subset, grid_row_num, title='Target', wspace=0.1)
        plot_img_grid(subfig[1], model1_samples_subset, grid_row_num, title=capword(self.labels[0]), wspace=0.1) 
        plot_img_grid(subfig[2], model2_samples_subset, grid_row_num, title=capword(self.labels[1]), wspace=0.1)
        
        fig.tight_layout()
        
        # Save plot
        plt.savefig(f'{self.plot_save_path}/samples.{self.image_file_format}')
        plt.close()

        'Marginal sums'
        # Plotting marginal sums
        real_marginal_sums = [marginal_sums(real_samples_subset[i]) for i in range(grid_row_num**2)]
        model1_marginal_sums = [marginal_sums(model1_samples_subset[i]) for i in range(grid_row_num**2)]
        model2_marginal_sums = [marginal_sums(model2_samples_subset[i]) for i in range(grid_row_num**2)]

        fig = plt.figure(figsize=(8,6))
        subfig = fig.subfigures(1, 3)
        
        plot_marginal_sums(real_marginal_sums, subfig[0], grid_row_num, title='Target')
        plot_marginal_sums(model1_marginal_sums, subfig[1], grid_row_num, title=capword(self.labels[0]))
        plot_marginal_sums(model2_marginal_sums, subfig[2], grid_row_num, title=capword(self.labels[1]))
        
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
        model1_ffts = fourier_transform_samples(model1_samples_subset)
        model2_ffts = fourier_transform_samples(model2_samples_subset)

        # Plotting fft
        fig = plt.figure(figsize=(8,3))
        subfig = fig.subfigures(1, 3, wspace=0.2)
        
        plot_img_grid(subfig[0], real_ffts, grid_row_num, title='Target FFT')
        plot_img_grid(subfig[1], model1_ffts, grid_row_num, title=f'{capword(self.labels[0])} FFT')
        plot_img_grid(subfig[2], model2_ffts, grid_row_num, title=f'{capword(self.labels[1])} FFT')
        
        # Save plot
        plt.savefig(f'{self.plot_save_path}/fft.{self.image_file_format}')
        plt.close()
            
    def plot_count_histogram(self, ignore_outliers=True):
        # Create figure
        fig = plt.figure(figsize=(4,3))

        # Bins for histogram
        bins = find_good_bins([self.real_blob_counts, self.model1_blob_counts, self.model2_blob_counts],
                              method='arange', ignore_outliers=True, percentile_range=(0,99))
        
        # Plot histogram
        real_hist, _, _ = plt.hist(self.real_blob_counts, bins=bins,
                histtype='step', label='target',
                color=(self.real_color, 0.8),
                facecolor=(self.real_color, 0.1), fill=True
                )
        plt.axvline(self.real_blob_count_mean, color=(self.real_color,0.5), linestyle='dashed', linewidth=1) # Only label mean for last model
        
        model1_hist, _, _ = plt.hist(self.model1_blob_counts, bins=bins, 
                    histtype='step', label=self.labels[0], color=(self.model1_color,0.8), linewidth=1.2)
        plt.axvline(self.model1_blob_count_mean, color=(self.model1_color,0.5), linestyle='dashed', linewidth=1)
 
        model2_hist, _, _ = plt.hist(self.model2_blob_counts, bins=bins, 
                    histtype='step', label=self.labels[1], color=(self.model2_color,0.8), linewidth=1)
        plt.axvline(self.model2_blob_count_mean, color=(self.model2_color,0.5), linestyle='dashed', linewidth=1)

        _, max_ylim = plt.ylim()
        plt.text(self.real_blob_count_mean*1.1, max_ylim*0.3,
                'Mean: {:.2f}'.format(self.real_blob_count_mean), color=(self.real_color,1))
        plt.text(self.model1_blob_count_mean*1.1, max_ylim*0.2,
                'Mean: {:.2f}'.format(self.model1_blob_count_mean), color=(self.model1_color,1))
        plt.text(self.model2_blob_count_mean*1.1, max_ylim*0.1,
                'Mean: {:.2f}'.format(self.model2_blob_count_mean), color=(self.model2_color,1))
        
        # Format
        plt.ylabel('sample count')
        plt.xlabel('blob count')
        plt.suptitle(f"Histogram of blob count, {len(self.real_blob_counts)} samples")
        plt.legend()
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/count-histogram.{self.image_file_format}')
        plt.close()
                
        # Log js    
        js_1 = JSD(real_hist, model1_hist)
        js_2 = JSD(real_hist, model2_hist)
        self.log_in_dict([self.labels[0], 'blob count', 'JS div', js_1])
        self.log_in_dict([self.labels[1], 'blob count', 'JS div', js_2])
            
    def plot_stack_images(self):
        """Stack images"""
        'Stack'
        stacked_real_img = stack_imgs(self.real_samples)
        stacked_model1_img = stack_imgs(self.model1_samples)
        stacked_model2_img = stack_imgs(self.model2_samples)

        'Plot image'
        # Create figure
        fig, axs = plt.subplots(1, 3, figsize=(4,2.5))

        # Plot
        plot_stacked_imgs(axs[0], stacked_real_img, title=f"Target\n{self.subset_sample_num} samples")
        plot_stacked_imgs(axs[1], stacked_model1_img, title=f"model1\n{self.subset_sample_num} samples")
        plot_stacked_imgs(axs[2], stacked_model2_img, title=f"model2usion\n{self.subset_sample_num} samples")
        
        # Format
        fig.suptitle('Stacked image')
        plt.tight_layout()

        # Save plot
        plt.savefig(f'{self.plot_save_path}/stack.{self.image_file_format}')
        plt.close()
        
        'Stacked image histogram'
        # Plot
        real_pxl_lst = stacked_real_img.ravel()
        model1_pxl_lst = stacked_model1_img.ravel()
        model2_pxl_lst = stacked_model2_img.ravel()
        
        bins = find_good_bins([real_pxl_lst, model1_pxl_lst, model2_pxl_lst],
                              method='linspace', num_bins=15, ignore_outliers=False)
        # Create figure
        fig = plt.figure(figsize=(4,3))

        # Plot
        real_hist, _, _ = plt.hist(
            real_pxl_lst, bins=bins,
            histtype='step', label='target', color=(self.real_color,0.8), 
            facecolor=(self.real_color, 0.1), fill=True
            )
        model1_hist, _, _ = plt.hist(
            model1_pxl_lst, bins=bins, 
            histtype='step', label=self.labels[0], color=(self.model1_color,0.8), linewidth=1.2
            )
        model2_hist, _, _ = plt.hist(
            model2_pxl_lst, bins=bins, 
            histtype='step', label=self.labels[1], color=(self.model2_color,0.8)
            )
        
        # Format
        plt.ylabel('pixel count')
        plt.xlabel('stacked pixel value')
        plt.suptitle(f"Histogram of stacked image")
        plt.legend()
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/stack-pixel-histogram.{self.image_file_format}')
        plt.close()
        
        # Log js    
        js_1 = JSD(real_hist, model1_hist)
        js_2 = JSD(real_hist, model2_hist)
        self.log_in_dict([self.labels[0], 'stack img pixels', 'JS div', js_1])
        self.log_in_dict([self.labels[1], 'stack img pixels', 'JS div', js_2])
        
    def plot_pixel_histogram(self, max_sample_size=500):
        real_samples_subset = self.real_samples[:max_sample_size]
        model1_samples_subset = self.model1_samples[:max_sample_size]
        model2_samples_subset = self.model2_samples[:max_sample_size]
        
        'Stacked histogram'
        real_hist_stack = stack_histograms(real_samples_subset)
        model1_hist_stack = stack_histograms(model1_samples_subset)
        model2_hist_stack = stack_histograms(model2_samples_subset)

        # Create figure
        fig, ax = plt.subplots()

        # Plot
        plot_histogram_stack(ax, *real_hist_stack, color=(self.real_color,0.8), label='target')
        plot_histogram_stack(ax, *model1_hist_stack, color=(self.model1_color,0.8), label=self.labels[0])
        plot_histogram_stack(ax, *model2_hist_stack, color=(self.model2_color,0.8), label=self.labels[1])

        # Format
        plt.ylabel('sample count')
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
            extremum='min', title='Target', title_y=0.87, k=k
            )
        plot_extremum_num_blobs(
            subfig[1], self.model1_samples,
            self.model1_blob_coords, self.model1_blob_counts,
            extremum='min', title=capword(self.labels[0]), title_y=0.87, k=k
            )  
        plot_extremum_num_blobs(
            subfig[2], self.model2_samples,
            self.model2_blob_coords, self.model2_blob_counts,
            extremum='min', title=capword(self.labels[1]), title_y=0.87, k=k
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
            extremum='max', title='Target', title_y=0.87, k=k
            )
        plot_extremum_num_blobs(
            subfig[1], self.model1_samples,
            self.model1_blob_coords, self.model1_blob_counts,
            extremum='max', title=capword(self.labels[0]), title_y=0.87, k=k
            )  
        plot_extremum_num_blobs(
            subfig[2], self.model2_samples,
            self.model2_blob_coords, self.model2_blob_counts,
            extremum='max', title=capword(self.labels[1]), title_y=0.87, k=k
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
        model1_img_fluxes = find_total_fluxes(self.model1_samples)
        model2_img_fluxes = find_total_fluxes(self.model2_samples)

        # Create figure
        fig = plt.figure()
        
        # Bins for histogram
        concat_fluxes = np.concatenate([real_img_fluxes, model1_img_fluxes, model2_img_fluxes])
        if ignore_outliers:
            bin_min = np.floor(np.percentile(concat_fluxes, 1))
            bin_max = np.ceil(np.percentile(concat_fluxes, 99))
        else:        
            bin_min = np.min(concat_fluxes)
            bin_max = np.min(concat_fluxes)

        bins = np.arange(bin_min-3, bin_max+3,bin_max/50)

        # Plot
        real_hist, _, _ = plt.hist(
            real_img_fluxes, bins=bins, 
            histtype='step', label='target', color=(self.real_color,0.8)
            )
        model1_hist, _, _ = plt.hist(
            model1_img_fluxes, bins=bins,
            histtype='step', label=self.labels[0], color=(self.model1_color,0.8)
            )
        model2_hist, _, _ = plt.hist(
            model2_img_fluxes, bins=bins,
            histtype='step', label=self.labels[1], color=(self.model2_color,0.8)
            )
 
        # Format   
        plt.ylabel('sample count')
        plt.xlabel('total flux')
        plt.suptitle(f"Histogram of total flux, {self.subset_sample_num} samples")
        plt.legend()
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/total-flux-histogram.{self.image_file_format}')
        plt.close()
        
        # Log js    
        js_1 = JSD(real_hist, model1_hist)
        js_2 = JSD(real_hist, model2_hist)
        self.log_in_dict([self.labels[0], 'blob count', 'JS div', js_1])
        self.log_in_dict([self.labels[1], 'blob count', 'JS div', js_2])
        
    def plot_residual_samples(self, num_plots=5, grid_row_num=2):
        'Inspect residuals (sample-fit)'
        for n in range(num_plots):    
            # Get images
            model1_sample_subset = self.model1_samples[n*(grid_row_num**2):(n+1)*(grid_row_num**2)]
            model1_res_subset = self.model1_residuals[n*(grid_row_num**2):(n+1)*(grid_row_num**2)]
            
            model2_sample_subset = self.model2_samples[n*(grid_row_num**2):(n+1)*(grid_row_num**2)]
            model2_res_subset = self.model2_residuals[n*(grid_row_num**2):(n+1)*(grid_row_num**2)]
                
            'Images'
            # For image max plot
            concat_subset = np.concatenate([model1_sample_subset, model2_sample_subset])
            
            # Plotting grid of images
            fig = plt.figure(figsize=(4.5, 6))
            subfig = fig.subfigures(4, 2, wspace=0.1, hspace=0.3, height_ratios=(0.5,5,5,1))
            
            subplots_samp = plot_img_grid(subfig[1,0], model1_sample_subset, grid_row_num,
                                        title=f'{capword(self.labels[0])} samples', title_y=1.06,
                                        vmin=-0.05, vmax=np.max(concat_subset), wspace=0.05, hspace=0.01)
            subplots_res = plot_img_grid(subfig[1,1], model1_res_subset, grid_row_num, 
                                        title=f'{capword(self.labels[0])} residuals', title_y=1.06, 
                                        cmap='RdBu', vmin=-0.07, vmax=0.07, wspace=0.05, hspace=0.01)
            
            subplots_samp = plot_img_grid(subfig[2,0], model2_sample_subset, grid_row_num,
                                        title=f'{capword(self.labels[1])} samples', title_y=1.06,
                                        vmin=-0.05, vmax=np.max(concat_subset), wspace=0.05, hspace=0.01)
            subplots_res = plot_img_grid(subfig[2,1], model2_res_subset, grid_row_num,
                                        title=f'{capword(self.labels[1])} residuals', title_y=1.06,
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
    
    def residual_mse(self):
        'MSE of the generated samples and fit'
        model1_mse = np.mean(mse_from_residuals(self.model1_residuals))
        model2_mse = np.mean(mse_from_residuals(self.model2_residuals))
        
        self.log_in_dict([self.labels[0], 'mean sample residual', 'mse', model1_mse])
        self.log_in_dict([self.labels[1], 'mean sample residual', 'mse', model2_mse])
        

class compareModule(compareUtils):
    def __init__(self, model1_run, model2_run, labels):
        super().__init__(model1_run, model2_run, labels)
        
        print(f'{model1_run}: epoch {self.model_epochs[0]} vs {model2_run}: epoch {self.model_epochs[1]}')
        print('---------------------------------------')
        print('loading models and data...')
        self.load_models()
        self.load_generated_samples()
        print('loading complete')
        
        print('')
        print('model sizes, in number of learnable parameters')
        if 'gan' in model1_run:
            print(f'{self.model1_run} -  {millify(np.sum(self.model1_size))} (generator: {millify(self.model1_size[0])} | discrinimator: {millify(self.model1_size[1])})')
        elif 'diff' in model1_run:
            print(f'{self.model1_run} -  {millify(self.model1_size)}')
        if 'gan' in model2_run:
            print(f'{self.model2_run} -  {millify(np.sum(self.model2_size))} (generator: {millify(self.model2_size[0])} | discrinimator: {millify(self.model2_size[1])})')
        elif 'diff' in model2_run:
            print(f'{self.model2_run} -  {millify(self.model2_size)}')
        print('')
        print(f'loaded sample size \nreal: {self.real_samples.shape} | model1: {self.model1_samples.shape} | model2: {self.model2_samples.shape}')
        print('')
        
        # Dictionary for logging metrics
        self.log_dict = {
            'model': [],
            'stat': [],
            'metric': [],
            'result': []
        }
        
    def test_speed(self, num_of_samples):
        """Test the generation speed"""
        print(f"cpu | {num_of_samples} samples")
        print('--------------------')
        
        print('generating samples...')
        if 'gan' in self.model1_run:
            model1_time = self.gan_speed(self.model1, num_of_samples,
                self.model1_training_params['network_params']['latent_dim'])
        elif 'diff' in self.model1_run:
            model1_time = self.diff_speed(self.model1, num_of_samples)
        if 'gan' in self.model2_run:
            model2_time = self.gan_speed(self.model2, num_of_samples,
                self.model2_training_params['network_params']['latent_dim'])
        elif 'diff' in self.model2_run:
            model2_time = self.diff_speed(self.model2, num_of_samples)
        
        print(f'{self.model1_run}: {model1_time:.4f}s | {self.model2_run}: {model2_time:.4f}s')
        print(f'{self.model1_run} is {model2_time/model1_time:.0f} times faster')
    
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
        self.residual_mse()
        
        log_dict = self.get_log_dict()
        save_log_dict(f'{self.plot_save_path}/metrics', log_dict)

if __name__=="__main__":
    tester = compareModule(model1_run, model2_run, labels, model_epochs)
    tester.plot()