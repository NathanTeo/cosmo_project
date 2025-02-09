"""
Author: Nathan Teo

This script contains classes that loads, tests, and plots for model evaluation
"""

import os

import numpy as np
from scipy import stats
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm.auto import tqdm

from code_model.testers.eval_utils import *
from code_model.testers.plotting_utils import *

class testDataset():
    """
    Prepares dataset for testing
    1. Initialize params
    2. Load real images
    3. Generate images
    """
    def __init__(self, generation_params, training_params, testing_params):
        """Initialize variables"""
        self.training_params = training_params
        
        self.model_version = training_params['model_version']

        self.blob_num = generation_params['blob_num']
        self.num_distr = generation_params['num_distribution']
        self.blob_amplitude = init_param(generation_params, 'blob_amplitude', default=1)
        self.clustering = init_param(generation_params, 'clustering')
        self.blob_size = init_param(generation_params, 'blob_size', 5)
        self.real_sample_num = generation_params['sample_num']
        self.image_size = training_params['image_size']
        self.min_dist = init_param(generation_params, 'minimum_distance')

        self.batch_size = training_params['batch_size']
        self.num_workers = training_params['num_workers']
        self.data_transforms = init_param(training_params, 'data_transforms')
        
        self.grid_row_size = init_param(testing_params,'grid_row_size', default=2)
        self.num_plots = init_param(testing_params, 'num_plots', default=5)
        self.subset_sample_num = int(init_param(testing_params, 'subset_sample_num', default=5_000))
        self.loss_zoom_bounds = init_param(testing_params, 'loss_zoom_bounds', default=(-1,1))
        self.enable_count = init_param(testing_params, 'count_blobs', default=False)
        self.counting_params = init_param(testing_params, 'counting_params')
        self.testing_seed = init_param(testing_params, 'seed', default=1)
        self.num_models = init_param(testing_params, 'num_models', default=1)
        self.model_epochs = init_param(testing_params, 'model_epochs')
        
        # Use model_epochs for num_models if model_epochs is given
        if self.model_epochs is not None:
            if self.num_models!=len(self.model_epochs):
                print(f'note: number of models changed from {self.num_models} to {len(self.model_epochs)} to follow model epoch input')
            self.num_models = len(self.model_epochs)
        
        """Paths"""
        self.root_path = training_params['root_path']
        self.data_path = f'{self.root_path}/data'
        # Get data file name
        filenames = os.listdir(self.data_path)
        if len(filenames)>1:
            raise Exception('More than 1 data file found')
        else:
            self.data_file_name = filenames[0]

        self.chkpt_path = f'{self.root_path}/checkpoints'
        self.log_path = f'{self.root_path}/logs'
        self.plot_save_path = f'{self.root_path}/plots/images'
        self.output_save_path = f'{self.root_path}/plots/model_output'
        
        if not os.path.exists(self.plot_save_path):
            os.makedirs(self.plot_save_path)

        if not os.path.exists(self.output_save_path):
            os.makedirs(self.output_save_path)
        
        """Load logs"""
        # Load loss
        self.losses = np.load(f'{self.log_path}/losses.npz')
        self.epoch_last = self.losses['epochs'][-1]
        
        logged_params = np.load(f'{self.log_path}/logged_params.npz')
        self.scaling_factor = logged_params['scaling_factor']
        print(f'scaling factor: {self.scaling_factor}')
        
        """Plotting style"""
        self.plot_fullsize = (6,4)
        self.plot_halfsize = (3.5,3)
        self.plot_2thirdsize = (4,3)
        
        self.image_file_format = 'png'
        self.real_color = 'black'
        self.gen_color = 'red'
        
        self.num_distr_range = (0,100)
        
        # Need to change if different number of models are plotted, find a way to make this automatic/input?
        self.fill_alphas = [*[0 for _ in range(self.num_models-1)], 0.2]
        self.line_alphas = [*[min(0.2*(i+1),0.6) for i in range(self.num_models-1)], 0.8] # number of models loaded should be less than 5
        self.line_widths = [*[1 for _ in range(self.num_models-1)], 1.2]
        self.line_styles = [*['-' for _ in range(self.num_models-1)], '--']
        self.select_last_epoch = [*[False for _ in range(self.num_models-1)], True]
        
        print(f'fill alphas: {self.fill_alphas}')
        print(f'line alphas: {self.line_alphas}')
        print(f'line widths: {self.line_widths}')
        
        """Initialize seed"""
        pl.seed_everything(self.testing_seed)
        
    def load_data(self, DataModule):
        """Load real data"""
        self.real_imgs = np.load(f'{self.data_path}/{self.data_file_name}')
        self.data = DataModule(
            data_file=f'{self.data_path}/{self.data_file_name}',
            batch_size=self.batch_size, num_workers=self.num_workers,
            truncate_ratio=self.subset_sample_num/self.real_sample_num,
            transforms=self.data_transforms
            )
    
    def load_models(self, model_dict):
        """Load models"""
        # Get checkpoints of models to be tested
        all_filenames = os.listdir(self.chkpt_path)
        all_filenames.sort()
        if 'last.ckpt' in all_filenames:
            # unable to get epoch of last.ckpt without calling trainer, so only use saved epochs for testing
            all_filenames.remove('last.ckpt')
        
        # If only model number if given,
        # load models, starting with the final epoch and halving the epoch number each time 
        if self.model_epochs is None:
            self.filenames = []
            for x in np.arange(self.num_models,0,-1):
                self.filenames.append(all_filenames[int(len(all_filenames)/(2**(x-1))-1)])
            self.model_epochs = [int(file[-9:-5]) for file in self.filenames]
        # If model epochs are given, load the corresponding epochs
        elif isinstance(self.model_epochs[0], int):
            self.filenames = [filename for filename in all_filenames if any(f'{epoch:04d}' in filename for epoch in self.model_epochs)]
            self.is_checkpoints_found(self.filenames, self.model_epochs)
        
        self.models = [model_dict[self.model_version].load_from_checkpoint(
            f'{self.chkpt_path}/{file}',
            **self.training_params
            ) for file in self.filenames]
        
        # Set scaling factor
        for model in self.models:
            model.scaling_factor = self.scaling_factor
    
    def load_samples(self):
        """Load samples from npy file"""
        self.all_gen_imgs = []
        for filename in self.filenames:
            self.all_gen_imgs.append(np.load(
                '{}/{}_{:.0g}.npy'.format(
                    self.output_save_path,
                    filename[:-5],
                    self.subset_sample_num
            )))
        return self.all_gen_imgs
    
    def generate_samples(self):
        """Generate samples given a list of models"""
        self.all_gen_imgs = []
        trainer = pl.Trainer()
        for model, filename, epoch in zip(self.models, self.filenames, self.model_epochs):
            print(f'epoch {epoch} | file {filename}')
            trainer.test(model, self.data)
            
            # Apply inverse tranform
            if self.data_transforms is not None:
                data_inv_transforms = [inv_transform_dict[transform] for transform in self.data_transforms][::-1]
                for transform in data_inv_transforms:
                    samples = transform(model.outputs)
            else:
                samples = model.outputs
            
            # Add outputs to list
            self.all_gen_imgs.append(samples)
                
            # Save outputs
            np.save('{}/{}_{:.0g}.npy'.format(
                self.output_save_path, filename[:-5], self.subset_sample_num
                ), samples)
            
        return self.all_gen_imgs
    
    def is_checkpoints_found(self, checkpoints, epochs):
        """Check if all checkpoints from the epochs requested are found"""
        if len(checkpoints)!=len(epochs):
            print()
            print(f'checkpoints: {checkpoints}')
            print(f'epochs: {epochs}')
            epochs_found = [int(file[-9:-5]) for file in checkpoints]
            epochs_not_found = [epoch for epoch in epochs if epoch not in epochs_found]
            raise Exception(f'could not find checkpoints {epochs_not_found}')
        else:
            pass
    
    def truncate(self):
        """Make subset used for testing"""
        self.real_imgs_subset = self.real_imgs[:self.subset_sample_num]
        self.all_gen_imgs_subset = [gen_imgs[:self.subset_sample_num] for gen_imgs in self.all_gen_imgs]

    def prep_data(self, dataModule, model_dict, testing_restart=False):
        """Run all steps to prepare data"""
        if testing_restart:
            print('restart testing')
        print('loading data and models...', end='\t')
        self.load_data(dataModule)
        self.load_models(model_dict)
        print('complete')
        print(f'models epochs: {self.model_epochs}')
        try:
            if not testing_restart:
                # Load saved outputs if available
                print('model output found')
                print('loading model output...', end='\t')
                self.load_samples()
                print('complete')
            else:
                # Retest model
                print('generating samples...')
                self.generate_samples()
            
        except FileNotFoundError:
            # Test model if no saved outputs are found
            print('generating samples')
            self.generate_samples()
        
        self.truncate()
    
    def prep_data_no_gen(self, dataModule, model_dict):
        """Run all steps to prepare data"""
        print('loading all models and samples...', end='\t')
        self.load_data(dataModule)
        self.load_models(model_dict)
        try:
            self.load_samples()
            print('complete')
        except FileNotFoundError:
            raise Exception("generated samples not found, please generate samples first")
        print(f'models epochs: {self.model_epochs}')
        self.truncate()
        
class blobTester(testDataset):
    """
    Run tests for blob data
    """
    def __init__(self, *args):
        if type(args[0]) is testDataset:
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(*args)
        
        # Dictionary for logging metrics
        self.log_dict = {
            'epoch': [],
            'stat': [],
            'metric': [],
            'result': []
        }
        
    def log_in_dict(self, entry):
        """Log desired metrics in the initialized dictionary"""
        self.log_dict['epoch'].append(entry[0])
        self.log_dict['stat'].append(entry[1])
        self.log_dict['metric'].append(entry[2])
        self.log_dict['result'].append(entry[3])
    
    def images(self):
        """Plot generated images - grid of images, marginal sums, blob coordinates"""
        for gen_imgs, epoch in zip(self.all_gen_imgs, self.model_epochs):
            print(f'epoch: {epoch}')
            # Make folder
            save_path = f'{self.plot_save_path}/epoch-{epoch}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                    
            for n in tqdm(range(self.num_plots), desc='plotting'):
                # Get images
                real_imgs_subset = self.real_imgs_subset[n*(self.grid_row_size**2):(n+1)*(self.grid_row_size**2)]
                gen_imgs_subset = gen_imgs[n*(self.grid_row_size**2):(n+1)*(self.grid_row_size**2)] # Only use last model for this section
                
                'Images'
                # Plotting grid of images
                fig = plt.figure(figsize=(5,2.5))
                subfig = fig.subfigures(1, 2, wspace=0.1)
                
                vmin = np.min(np.concatenate([real_imgs_subset, gen_imgs_subset]))
                vmax = np.max(np.concatenate([real_imgs_subset, gen_imgs_subset]))
                
                plot_img_grid(subfig[0], real_imgs_subset, self.grid_row_size, title='Target Imgs', vmin=vmin, vmax=vmax)
                plot_img_grid(subfig[1], gen_imgs_subset, self.grid_row_size, title='Generated Imgs', vmin=vmin, vmax=vmax)
                
                # Save plot
                plt.savefig(f'{save_path}/gen-imgs_{n}.{self.image_file_format}')
                plt.close()
                
                # Plotting without consistent cmap range
                fig = plt.figure(figsize=(5,2.5))
                subfig = fig.subfigures(1, 2, wspace=0.1)
                
                plot_img_grid(subfig[0], real_imgs_subset, self.grid_row_size, title='Target Imgs')
                plot_img_grid(subfig[1], gen_imgs_subset, self.grid_row_size, title='Generated Imgs')
                
                # Save plot
                plt.savefig(f'{save_path}/gen-imgs_autoscale_{n}.{self.image_file_format}')
                plt.close()

                'Marginal sums'
                # Plotting marginal sums
                real_marginal_sums = [marginal_sums(real_imgs_subset[i]) for i in range(self.grid_row_size**2)]
                gen_marginal_sums = [marginal_sums(gen_imgs_subset[i]) for i in range(self.grid_row_size**2)]
                
                fig = plt.figure(figsize=(4,6))
                subfig = fig.subfigures(1, 2)
                
                plot_marginal_sums(real_marginal_sums, subfig[0], self.grid_row_size, title='Target')
                plot_marginal_sums(gen_marginal_sums, subfig[1], self.grid_row_size, title='Generated')
                
                # Format
                fig.suptitle('Marginal Sums')
                plt.legend()
                plt.tight_layout()
                
                # Save plot
                plt.savefig(f'{save_path}/marg-sums_{n}.{self.image_file_format}')
                plt.close()
                
                'FFT'
                # Fourier transform images
                real_ffts = fourier_transform_samples(real_imgs_subset)
                gen_ffts = fourier_transform_samples(gen_imgs_subset)

                # Plotting fft
                fig = plt.figure(figsize=(6,3))
                subfig = fig.subfigures(1, 2, wspace=0.2)
                
                plot_img_grid(subfig[0], real_ffts, self.grid_row_size, title='Target FFT')
                plot_img_grid(subfig[1], gen_ffts, self.grid_row_size, title='Generated FFT')
                
                # Save plot
                plt.savefig(f'{save_path}/fft_{n}.{self.image_file_format}')
                plt.close()
            
    def stack(self):
        """Stack images"""
        'Stack'
        print('stacking...', end='   ')
        stacked_real_img = stack_imgs(self.real_imgs_subset)
        stacked_gen_img = stack_imgs(self.all_gen_imgs_subset[-1]) # Only stack for last model
        print('complete')
        
        'Plot image'
        # Create figure
        fig, axs = plt.subplots(1, 2, figsize=self.plot_halfsize)

        # Plot
        plot_stacked_imgs(axs[0], stacked_real_img, title=f"target")
        plot_stacked_imgs(axs[1], stacked_gen_img, title=f"generated")

        # Format
        fig.suptitle('Stacked Image')
        plt.tight_layout()

        # Save plot
        plt.savefig(f'{self.plot_save_path}/stack.{self.image_file_format}')
        plt.close()
        
        'Stacked image histogram'
        # Create figure
        fig = plt.figure(figsize=self.plot_halfsize)

        # Plot
        real_pxl_lst = stacked_real_img.ravel()
        gen_pxl_lst = stacked_gen_img.ravel()
        
        bins = find_good_bins([real_pxl_lst, gen_pxl_lst], method='linspace', num_bins=15, ignore_outliers=False)
        
        real_hist, _, _ = plt.hist(real_pxl_lst, bins=bins, histtype='step', label='target',
                                   color=(self.real_color,0.8), facecolor=(self.real_color,0.1), fill=True)
        gen_hist, _, _ = plt.hist(gen_pxl_lst, bins=bins, histtype='step', label='generated',
                                  color=(self.gen_color,0.8))

        # Format
        plt.ylabel('pixel count')
        plt.xlabel('stacked pixel value')
        plt.suptitle(f"Histogram of stacked image")
        plt.legend()
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/stack-histogram-img.{self.image_file_format}')
        plt.close()
        
        # Log js
        js = JSD(real_hist, gen_hist)
        self.log_in_dict([self.model_epochs[-1],'stack img hist', 'JS div', js])

    def count_blobs_fast(self):
        """Count blobs only using gaussian decomp"""
        print('counting blobs...')
        filter_sd = self.counting_params[0]
        blob_threshold_rel = self.counting_params[1]

        # Real
        self.real_blob_coords, self.real_blob_nums, self.real_peak_vals = samples_blob_counter_fast(
            self.real_imgs_subset, 
            blob_size=self.blob_size, min_peak_threshold=self.blob_amplitude*blob_threshold_rel,
            filter_sd=filter_sd,
            progress_bar=True
            )
        self.real_indv_peak_counts, self.real_blob_counts = count_blobs_from_peaks(self.real_peak_vals, self.blob_num)

        # Generated
        self.all_gen_blob_coords, self.all_gen_blob_nums, self.all_gen_peak_vals = map(
            list, zip(*[samples_blob_counter_fast(
                subset, 
                blob_size=self.blob_size, min_peak_threshold=self.blob_amplitude*blob_threshold_rel,
                filter_sd=filter_sd,
                progress_bar=True
                ) for subset in self.all_gen_imgs_subset])
            )
        
        # Count
        self.all_gen_indv_peak_counts, self.all_gen_blob_counts = map(
            list, zip(*[count_blobs_from_peaks(peak_vals, self.blob_num) for peak_vals in self.all_gen_peak_vals]))

        # Find mean number of blobs
        self.real_blob_num_mean = np.mean(self.real_blob_counts)
        self.all_gen_blob_num_mean = [np.mean(blob_counts) for blob_counts in self.all_gen_blob_counts]
        print(f'mean number of target peaks: {self.real_blob_num_mean}')
        print(f'mean number of generated peaks: {self.all_gen_blob_num_mean[-1]}')
        
    def count_non_overlapping_blobs(self):
        """Count blobs using gaussian decomp"""
        print('counting blobs...')
        blob_threshold_rel = self.counting_params[1]

        # Real
        self.real_blob_coords, self.real_blob_counts, self.real_blob_amplitudes = samples_blob_counter_fast(
            self.real_imgs_subset, 
            blob_size=self.blob_size, min_peak_threshold=self.blob_amplitude*blob_threshold_rel,
            method='zero', progress_bar=True
            )

        # Generated
        self.all_gen_blob_coords, self.all_gen_blob_counts, self.all_gen_blob_amplitudes = map(
            list, zip(*[samples_blob_counter_fast(
                subset, 
                blob_size=self.blob_size, min_peak_threshold=self.blob_amplitude*blob_threshold_rel,
                method='zero', progress_bar=True
                ) for subset in self.all_gen_imgs_subset])
            )

        # Find mean number of blobs
        self.real_blob_num_mean = np.mean(self.real_blob_counts)
        self.all_gen_blob_num_mean = [np.mean(blob_counts) for blob_counts in self.all_gen_blob_counts]
        print(f'mean number of target peaks: {self.real_blob_num_mean}')
        print(f'mean number of generated peaks: {self.all_gen_blob_num_mean[-1]}')

    def count_blobs(self):
        """Count blobs by fitting"""
        print('counting blobs...')
        
        # Initialize counter
        counter = blobFitter(blob_size=self.blob_size, blob_amplitude=self.blob_amplitude)
        
        # Count blobs for real images
        counter.load_samples(self.real_imgs_subset)
        self.real_blob_coords, self.real_blob_counts = counter.count()

        # Count blobs for generated images
        self.all_gen_blob_coords, self.all_gen_blob_counts = [], []
        for subset in self.all_gen_imgs_subset:
            counter.load_samples(subset)
            coords, counts = counter.count()       
            self.all_gen_blob_coords.append(coords)
            self.all_gen_blob_counts.append(counts)
        
        # Find mean
        self.real_blob_num_mean = np.mean(self.real_blob_counts)
        self.all_gen_blob_num_mean = [np.mean(counts) for counts in self.all_gen_blob_counts]
    
    def save_counts(self):
        """Save counts"""
        np.savez('{}/counts.npz'.format(self.output_save_path), 
                 gen_counts = self.all_gen_blob_counts,
                 gen_coords = np.array(self.all_gen_blob_coords, dtype=object),
                 real_counts = self.real_blob_counts,
                 real_coords = np.array(self.real_blob_coords, dtype=object)
        )
        
    def load_counts(self):
        """Load counts"""
        file = np.load('{}/counts.npz'.format(self.output_save_path), allow_pickle=True)
        
        self.all_gen_blob_counts = file['gen_counts']
        self.all_gen_blob_coords = file['gen_coords'].tolist()
        self.real_blob_counts = file['real_counts']
        self.real_blob_coords = file['real_coords'].tolist()
        
        self.real_blob_num_mean = np.mean(self.real_blob_counts)
        self.all_gen_blob_num_mean = [np.mean(counts) for counts in self.all_gen_blob_counts]
        
        # TEMP FIX, CHANGE THIS 
        if self.num_models==1:
            self.all_gen_blob_counts = [self.all_gen_blob_counts[-1]]
            self.all_gen_blob_coords = [self.all_gen_blob_coords[-1]]
            self.all_gen_blob_num_mean = [self.all_gen_blob_num_mean[-1]]
        
    def blob_amp_stats(self):
        """Blob amplitude analysis"""
        'Histogram'
        # Create figure
        fig = plt.figure(figsize=self.plot_halfsize)
        
        real_amplitudes_concat = np.concatenate(self.real_blob_amplitudes)
        all_gen_amplitudes_concat = [np.concatenate(amps) for amps in self.all_gen_blob_amplitudes]
        
        # Bins for histogram
        bins = np.arange(0, 10+1, 0.5)
        
        # Plot histogram
        for i, blob_amplitudes in enumerate(all_gen_amplitudes_concat):
            gen_hist, _, _ = plt.hist(
                blob_amplitudes, bins=bins,
                histtype='step', label=f'epoch {self.model_epochs[i]}',
                color=(self.gen_color,self.line_alphas[i]), linewidth=self.line_widths[i],
                facecolor=(self.gen_color,self.fill_alphas[i]), fill=self.select_last_epoch[i]
                )
        
        real_hist, _, _ = plt.hist(real_amplitudes_concat, bins=bins, 
                histtype='step', label='target', color=(self.real_color,0.8))

        # Format
        plt.ylabel('blob count')
        plt.xlabel('blob amplitude')
        plt.suptitle(f"Histogram of blob amplitude")
        plt.legend()
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/amplitude-blobs-histogram.{self.image_file_format}')
        plt.close()

        'CDF'
        all_amps = np.concatenate([real_amplitudes_concat, *all_gen_amplitudes_concat])
        lower = np.min(all_amps)
        upper = np.max(all_amps) 
        x = np.arange(lower, upper, 1)

        plt.ecdf(real_amplitudes_concat, label='target')
        for i, amplitudes in enumerate(all_gen_amplitudes_concat):
            plt.ecdf(amplitudes, color=('r', self.line_alphas[i]), 
                     linewidth=self.line_widths[i], linestyle=self.line_styles[i],
                     label=f'epoch {self.model_epochs[i]}')
        
        # Format
        plt.legend()
        plt.title('CDF of the blob amplitudes')
        plt.xlabel('blob amplitude')
        plt.ylabel('probability')
        plt.tight_layout()
        plt.savefig(f'{self.plot_save_path}/amplitude-blobs-cdf.{self.image_file_format}')
        plt.close()
        
        'Log stats'
        # JS div
        js = JSD(real_hist, gen_hist)
        self.log_in_dict([self.model_epochs[-1], 'blob amplitude', 'JS div', js])
        # KS test
        for gen_amplitude, epoch in zip(all_gen_amplitudes_concat, self.model_epochs):
            ks = stats.kstest(gen_amplitude, real_amplitudes_concat)
            self.log_in_dict([epoch, 'blob amplitude', 'KS test', ks])
          
     
    def blob_num_stats(self):
        """Number of blob analysis"""
        'Histogram'
        # Create figure
        fig = plt.figure(figsize=self.plot_halfsize)

        # Bins for histogram
        bins = find_good_bins([self.real_blob_counts, *self.all_gen_blob_counts], method='arange',
                              ignore_outliers=True, percentile_range=self.num_distr_range,
                              spacing=(1.5,1.5))
                
        # Plot histogram
        for i, blob_counts in enumerate(self.all_gen_blob_counts):
            gen_hist, _, _ = plt.hist(
                blob_counts, bins=bins,
                histtype='step', label=f'epoch {self.model_epochs[i]}',
                color=(self.gen_color,self.line_alphas[i]), linewidth=self.line_widths[i],
                facecolor=(self.gen_color,self.fill_alphas[i]), fill=self.select_last_epoch[i]
                )
        plt.axvline(self.all_gen_blob_num_mean[-1], color=(self.gen_color,0.5), linestyle='dashed', linewidth=1) # Only label mean for last model
        
        real_hist, _, _ = plt.hist(self.real_blob_counts, bins=bins, 
                histtype='step', label='target', color=(self.real_color,0.8))
        plt.axvline(self.real_blob_num_mean, color=(self.real_color,0.5), linestyle='dashed', linewidth=1)

        _, max_ylim = plt.ylim()
        plt.text(self.real_blob_num_mean*1.1, max_ylim*0.9,
                'Mean: {:.2f}'.format(self.real_blob_num_mean), color=(self.real_color,1))
        plt.text(self.all_gen_blob_num_mean[-1]*1.1, max_ylim*0.8,
                    'Mean: {:.2f}'.format(self.all_gen_blob_num_mean[-1]), color=(self.gen_color,1))

        # Format
        ax = plt.gca()
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        plt.ylabel('sample count')
        plt.xlabel('blob count')
        plt.suptitle(f"Histogram of blob count")
        plt.legend()
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/number-blobs-histogram.{self.image_file_format}')
        plt.close()
        
        'CDF'
        lower = np.min(np.concatenate(self.all_gen_blob_counts))
        upper = np.max(np.concatenate(self.all_gen_blob_counts)) 
        x = np.arange(lower, upper, 1)

        plt.step(x, stats.poisson.cdf(x, mu=self.blob_num), where='post', color=('black', 0.5), linestyle='--', label='theory')
        for i, counts in enumerate(self.all_gen_blob_counts):
            plt.ecdf(counts, color=('r', self.line_alphas[i]), 
                     linewidth=self.line_widths[i], linestyle=self.line_styles[i],
                     label=f'epoch {self.model_epochs[i]}')
        
        # Format
        plt.legend()
        plt.title('CDF of the blob count')
        plt.xlabel('blob count')
        plt.ylabel('probability')
        plt.tight_layout()
        plt.savefig(f'{self.plot_save_path}/number-blobs-cdf.{self.image_file_format}')
        plt.close()
        
        'Log stats'
        # JS div
        js = JSD(real_hist, gen_hist)
        self.log_in_dict([self.model_epochs[-1], 'blob count', 'JS div', js])
        for gen_counts, epoch in zip(self.all_gen_blob_counts, self.model_epochs):
            # KS test
            if self.num_distr=='poisson':
                ks_stat = ks_poisson(gen_counts, self.blob_num)
                self.log_in_dict([epoch, 'blob count', 'KS test', ks_stat])
            # Accuracy
            elif self.num_distr=='delta':
                gen_r = np.where(gen_counts==self.blob_num, 1, 0).sum()/len(gen_counts)
                real_r = np.where(self.real_blob_counts==self.blob_num, 1, 0).sum()/len(self.real_blob_counts)
                self.log_in_dict([epoch, 'blob count', 'accuracy-generated', gen_r])
                self.log_in_dict([None, 'blob count', 'accuracy-target', real_r])
        
        'Extreme number of blobs'
        # Create figure
        fig = plt.figure(figsize=(3.5,2))
        subfig = fig.subfigures(1, 2, wspace=0.2)

        # Plot
        plot_extremum_num_blobs(
            subfig[0], self.real_imgs_subset,
            self.real_blob_coords, self.real_blob_counts,
            k=1, extremum='min', title='target'
            )
        plot_extremum_num_blobs(
            subfig[1], self.all_gen_imgs_subset[-1],
            self.all_gen_blob_coords[-1], self.all_gen_blob_counts[-1],
            k=1, extremum='min', title='generated'
            )  # Only perform for last model

        # Format
        fig.suptitle(f"Min blob count")
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/min-peak.{self.image_file_format}')
        plt.close()
        
        'Extreme number of blobs'
        # Create figure
        fig = plt.figure(figsize=(2.1,self.plot_2thirdsize[1]))

        # Plot
        plot_extremum_num_blobs(
            fig, self.all_gen_imgs_subset[-1],
            self.all_gen_blob_coords[-1], self.all_gen_blob_counts[-1],
            k=1, extremum='min', title=None
            )  # Only perform for last model

        # Format
        fig.suptitle(f"Minimum blob count", fontsize='large')
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/min-peak-gen.{self.image_file_format}')
        plt.close()
        
        # Create figure
        fig = plt.figure(figsize=(3.5,2))
        subfig = fig.subfigures(1, 2, wspace=0.2)

        # Plot
        plot_extremum_num_blobs(
            subfig[0], self.real_imgs_subset,
            self.real_blob_coords, self.real_blob_counts,
            k=1, extremum='max', title='target'
            )
        plot_extremum_num_blobs(
            subfig[1], self.all_gen_imgs_subset[-1],
            self.all_gen_blob_coords[-1], self.all_gen_blob_counts[-1],
            k=1, extremum='max', title='generated'
            )  

        # Format
        fig.suptitle(f"Max blobs count")
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/max-peak.{self.image_file_format}')
        plt.close()
        
        for n in range(self.num_plots):
            fig = plt.figure(figsize=(6,3))
            subfig = fig.subfigures(1, 2, wspace=0.2)
                
            plot_peak_grid(subfig[0], self.real_imgs_subset[n*4:(n+1)*4], self.real_blob_coords[n*4:(n+1)*4], self.grid_row_size, 
                            title='target imgaes', subplot_titles=self.real_blob_counts[n*4:(n+1)*4])
            plot_peak_grid(subfig[1], self.all_gen_imgs_subset[-1][n*4:(n+1)*4], self.all_gen_blob_coords[-1][n*4:(n+1)*4], self.grid_row_size, 
                            title='generated imgaes', subplot_titles=self.all_gen_blob_counts[-1][n*4:(n+1)*4])
            
            fig.text(.5, .03, 'number of blobs labelled above image', ha='center')
            
            # Save plot
            plt.savefig(f'{self.plot_save_path}/counts-imgs_{n}.{self.image_file_format}')
            plt.close()
        
    def power_spec(self):
        'Power spectrum'
        print('Calculating power spectrum...')
        
        # Get Cls
        image_size_angular = 1
        delta_ell = 500
        max_ell = 12000
        ell2d = ell_coordinates(self.image_size, image_size_angular/self.image_size)
        
        real_cl, real_err, bins = power_spectrum_stack(self.real_imgs_subset, 
                                                       delta_ell, max_ell, ell2d, image_size_angular,
                                                       progress_bar=True)
        all_gen_cl, all_gen_err, _ = map(
            list, zip(*[power_spectrum_stack(
                samples, 
                delta_ell, max_ell, ell2d, image_size_angular,
                progress_bar=True
                ) for samples in self.all_gen_imgs_subset])
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.plot_halfsize)
        
        # Plot
        plot_errorbars(
            ax, real_cl, bins, real_err, 
            color=((self.real_color,1), (self.real_color,0.5)),
            linewidth=1.2, elinewidth=2, capsize=4, fmt='o',
            label='target', scale='semilog_x', errorbars=True, line=False
        )
         
        for i, (cl, err) in enumerate(zip(all_gen_cl, all_gen_err)):
            plot_errorbars(
                ax, cl, bins, err, 
                color=((self.gen_color,0.5), (self.gen_color,0.5)),
                linewidth=self.line_widths[i],
                label=f'epoch {self.model_epochs[i]}', errorbars=self.select_last_epoch[i], line=False,
                scale='semilog_x'
            )
        
        # Format
        fig.suptitle(f"Power spectrum")
        plt.xlabel(r'$l$')
        plt.ylabel(r'$C_l$')
        plt.xlim(np.min(bins)*.9,np.max(bins)*1.1)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.tight_layout()
        
        plt.legend()

        # Save
        plt.savefig(f'{self.plot_save_path}/power-spec.{self.image_file_format}')
        plt.close() 

    def flux_stats(self):
        'Calculate total fluxes'
        print("calculating total flux...", end="   ")
        real_img_fluxes = find_total_fluxes(self.real_imgs_subset)
        all_gen_img_fluxes = [find_total_fluxes(subset) for subset in self.all_gen_imgs_subset]
        print('complete')
        
        'Total flux histogram'
        # Create figure
        fig = plt.figure(figsize=self.plot_halfsize)

        # Bins for histogram
        space = 0.1*np.min(all_gen_img_fluxes[-1])
        bins = find_good_bins([real_img_fluxes, *all_gen_img_fluxes], method='linspace', num_bins=20, ignore_outliers=True, 
                              percentile_range=(0,99.5), spacing=(space, space))
        
        # Plot
        for i, fluxes in enumerate(all_gen_img_fluxes):
            gen_hist, _, _ = plt.hist(fluxes, bins=bins,
                    histtype='step', label=f'epoch {self.model_epochs[i]}', 
                    color=(self.gen_color,self.line_alphas[i]),
                    facecolor=(self.gen_color,self.fill_alphas[i]), fill=self.select_last_epoch[i], 
                    linewidth=self.line_widths[i]
                    )
        real_hist, _, _ = plt.hist(real_img_fluxes, bins=bins,
                    histtype='step', label='target', color=(self.real_color,0.8))
 
        # Format   
        plt.ylabel('sample count')
        plt.xlabel('total flux')
        plt.suptitle(f"Histogram of total flux")
        plt.legend()
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/total-flux-histogram.{self.image_file_format}')
        plt.close()
        
        'Total flux cdf'
        # Create figure
        fig = plt.figure(figsize=self.plot_halfsize)

        # Plot
        plt.ecdf(real_img_fluxes,
                label='target', color=(self.real_color,0.8))
        for i, fluxes in enumerate(all_gen_img_fluxes):
            plt.ecdf(fluxes,
                    label=f'epoch {self.model_epochs[i]}', 
                    color=(self.gen_color,self.line_alphas[i]), 
                    linewidth=self.line_widths[i], linestyle=self.line_styles[i]
                    )
        
 
        # Format   
        plt.ylabel('sample count')
        plt.xlabel('probability')
        plt.suptitle(f"CDF of total flux")
        plt.legend()
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/total-flux-cdf.{self.image_file_format}')
        plt.close()
        
        'Log stats'
        # JS div
        js = JSD(real_hist, gen_hist)
        self.log_in_dict([self.model_epochs[-1], 'total flux', 'JS div', js])
        # KS test
        for gen_fluxes, epoch in zip(all_gen_img_fluxes, self.model_epochs):
            ks = stats.kstest(gen_fluxes, find_total_fluxes(self.real_imgs))
            self.log_in_dict([epoch, 'total flux', 'KS test', ks])
        
        'Extreme flux'
        # Create figure
        fig = plt.figure(figsize=(3.5,2))
        subfig = fig.subfigures(1, 2, wspace=0.2)

        # Plot
        plot_extremum_flux(
            subfig[0], self.real_imgs_subset,
            real_img_fluxes,
            k=1, extremum='min', title='target'
            )
        plot_extremum_flux(
            subfig[1], self.all_gen_imgs_subset[-1],
            all_gen_img_fluxes[-1],
            k=1, extremum='min', title='generated'
            )  # Only perform for last model

        # Format
        fig.suptitle(f"Min flux")
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/min-flux.{self.image_file_format}')
        plt.close()

        # Create figure
        fig = plt.figure(figsize=(3.5,2))
        subfig = fig.subfigures(1, 2, wspace=0.2)

        # Plot
        plot_extremum_flux(
            subfig[0], self.real_imgs_subset,
            real_img_fluxes,
            k=1, extremum='max', title='target'
            )
        plot_extremum_flux(
            subfig[1], self.all_gen_imgs_subset[-1],
            all_gen_img_fluxes[-1],
            k=1, extremum='max', title='generated'
            )  

        # Format
        fig.suptitle(f"Max flux")
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/max-flux.{self.image_file_format}')
        plt.close()

    def two_point_correlation(self):
        """2-point correlation analysis"""
        print('calculating 2 point correlation...')
        max_dist = int(self.image_size *2/3)
        
        real_corrs, real_errs, edges = two_point_stack(
            self.real_blob_coords, max_dist, bins=15, progress_bar=True,
            logscale=True if self.clustering is not None else False
            )
        all_gen_corrs, all_gen_errs, _ = map(
            list, zip(*[two_point_stack(
                blob_coords, max_dist, bins=15, progress_bar=True,
                logscale=True if self.clustering is not None else False
                ) for blob_coords in self.all_gen_blob_coords])
        )

        # Create figure
        fig, ax = plt.subplots(figsize=self.plot_halfsize)
        
        # Plot
        if self.clustering is None:
            n = 10
            lin = np.linspace(-10, edges[-1]+10, n)
            plt.plot(lin, [0]*n, color=('black', 0.3), linestyle='dashed')
        
        plot_errorbars(
            ax, real_corrs, midpoints_of_bins(edges), real_errs, 
            color=((self.real_color,1), (self.real_color,0.5)),
            linewidth=1.2, elinewidth=1.5, capsize=4, fmt='o',
            label='target',
            scale='log' if self.clustering is not None else 'linear',
            line=False
        )
         
        for i, (corrs, errs) in enumerate(zip(all_gen_corrs, all_gen_errs)):
            plot_errorbars(
                ax, corrs, midpoints_of_bins(edges), errs, 
                color=((self.gen_color,self.line_alphas[i]), (self.gen_color,0.5)), 
                linewidth=self.line_widths[i],
                label=f'epoch {self.model_epochs[i]}', errorbars=self.select_last_epoch[i],
                scale='log' if self.clustering is not None else 'linear',
                line=False
            )
        
        # Format
        fig.suptitle(f"2-point correlation")
        plt.xlabel('pair distance')
        plt.ylabel('2-point correlation')
        plt.xlim(0, edges[-1]+1) # custom value 
        plt.tight_layout()
        
        plt.legend()

        # Save
        plt.savefig(f'{self.plot_save_path}/2-pt-corr.{self.image_file_format}')
        plt.close() 

    def pixel_stats(self):
        """Pixel value analysis"""
        'Single image histogram'
        histogram_num = 20
        real_imgs_subset_sh = self.real_imgs_subset[:histogram_num]
        gen_imgs_subset_sh = self.all_gen_imgs_subset[-1][:histogram_num]

        # Create figure
        fig, axs = plt.subplots(1,2, figsize=self.plot_halfsize)

        # Plot
        plot_pixel_histogram(axs[0], real_imgs_subset_sh, color=(self.real_color,0.5), bins=20)
        plot_pixel_histogram(axs[1], gen_imgs_subset_sh, color=(self.gen_color,0.5), bins=20)

        # Format
        plt.ylabel('pixel count')
        plt.xlabel('pixel value')
        fig.suptitle(f"Histogram of pixel values, {histogram_num} samples")
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/pixel-histogram.{self.image_file_format}')
        plt.close()

        'Stacked histogram'
        print('stacking histograms...')
        real_hist_stack = stack_histograms(self.real_imgs_subset)
        all_gen_hist_stack = [stack_histograms(subset) for subset in self.all_gen_imgs_subset]

        # Create figure
        fig, ax = plt.subplots(figsize=self.plot_halfsize)

        # Plot
        fill = [None, None, (self.gen_color, self.fill_alphas[-1])]
        for i, hist_stack in enumerate(all_gen_hist_stack):
            plot_histogram_stack(
                ax, *hist_stack,
                color=(self.gen_color,self.line_alphas[i]), linewidth=self.line_widths[i],
                fill_color=fill[i],
                label=f'epoch {self.model_epochs[i]}'
                )

        plot_histogram_stack(ax, *real_hist_stack, color=(self.real_color,0.8), label='target')

        # Format
        plt.ylabel('pixel count')
        plt.xlabel('pixel value')
        fig.suptitle(f"Stacked histogram of pixel values")
        plt.legend()
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/pixel-stack-histogram.{self.image_file_format}')
        plt.close()
    
    def test(self, testing_restart=False):
        """Run all testing methods"""
        self.images()
        self.stack()
        if self.enable_count:
            if os.path.exists(f'{self.output_save_path}/counts.npz') and not testing_restart:
                print('previous counts found, loading counts...', end='\t')
                self.load_counts()
                print('complete')
            elif self.min_dist is None:
                self.count_blobs()
                self.save_counts()
            elif self.min_dist is not None:
                self.count_non_overlapping_blobs()
                self.blob_amp_stats()
            self.blob_num_stats()
            self.two_point_correlation()
        self.power_spec()
        self.flux_stats()
        self.pixel_stats()
        save_log_dict(f'{self.plot_save_path}/metrics', self.log_dict)
        
class ganLogsPlotter(testDataset):
    def __init__(self, *args):        
        if type(args[0]) is testDataset:
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(*args)
    
    def load_last_model(self, model_dict):
        self.last_model = model_dict[self.model_version].load_from_checkpoint(
            f'{self.chkpt_path}/last.ckpt',
            **self.training_params
            )

    def loss(self):
        """Logged losses"""
        # Load loss
        g_losses = self.losses['g_losses']
        d_losses = self.losses['d_losses']
        epochs = self.losses['epochs']

        # Plot
        fig, ax1 = plt.subplots()

        ax1.plot(epochs, g_losses, color='C0', linewidth=0.7)
        ax2 = ax1.twinx() 
        ax2.plot(epochs, d_losses, color='C1', linewidth=0.7)

        # Format
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('generator loss', color='C0')
        ax2.set_ylabel('discriminator loss', color='C1')
        ax1.tick_params(axis='y', labelcolor='C0')
        ax2.tick_params(axis='y', labelcolor='C1')

        fig.tight_layout()

        # Save plot
        plt.savefig(f'{self.plot_save_path}/losses.{self.image_file_format}')
        plt.close()

        # Plot zoom 
        plt.figure(figsize=(6,4))
        plt.plot(epochs, g_losses, label='generator')
        plt.plot(epochs, d_losses, label='discriminator')

        # Format
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Model Loss')
        if self.model_version=='CWGAN':
            plt.axhline(y=0, color='r', alpha=0.2, linestyle='dashed')
        plt.ylim(*self.loss_zoom_bounds)
        plt.legend()
        plt.tight_layout()

        # Save plots
        plt.savefig(f'{self.plot_save_path}/losses-zm.{self.image_file_format}')
        plt.close()

    def score_models(self, model_dict):
        # Load models
        filenames = os.listdir(f'{self.chkpt_path}')
        filenames.sort()
        filenames.remove('last.ckpt')
        
        self.epochs = [int(file[6:-5]) for file in filenames]
        
        trainer = pl.Trainer()
        
        # Score
        print('scoring...')
        
        print('generating imgs, last model...')
        trainer.test(self.last_model, self.data)
        last_gen_imgs = self.last_model.outputs
        
        self.d_evo_models_gen_scores = []
        self.d_evo_models_real_scores = []
        self.g_evo_models_gen_scores = []
        
        for epoch, file in zip(self.epochs, filenames):
            print('--------------')
            print(f'epoch {epoch}')
            print('--------------')

            model = model_dict[self.model_version].load_from_checkpoint(
                f'{self.chkpt_path}/{file}',
                **self.training_params
            )
            
            print('scoring imgs, last model...')
            self.d_evo_models_gen_scores.append(model.score_samples(last_gen_imgs, progress_bar=True))
            self.d_evo_models_real_scores.append(model.score_samples(self.real_imgs_subset, progress_bar=True))

            print('generating imgs, earlier models...')
            trainer.test(model, self.data)
            
            print('scoring imgs, earlier models...')
            self.g_evo_models_gen_scores.append(
                self.last_model.score_samples(np.array(model.outputs),
                                              progress_bar=True)
                )
            model.outputs = None # Clear vram

    def loss_wrt_last_model(self, model_dict):
        self.load_last_model(model_dict)
        self.score_models(model_dict)
        
        'Loss as discriminator evolves'
        # Loss
        d_evo_loss = [
            -(np.mean(real_score) - np.mean(gen_score))
            for real_score, gen_score in zip(self.d_evo_models_real_scores, self.d_evo_models_gen_scores)
            ]
        
        'Loss as generator evolves'
        # Loss
        g_evo_loss = [-np.mean(gen_scores) for gen_scores in self.g_evo_models_gen_scores]
        
        'Plot'
        # Create figure
        fig, ax1 = plt.subplots()

        # Plot
        ax1.plot(self.epochs, g_evo_loss, color='C0', linewidth=0.7)
        ax2 = ax1.twinx() 
        ax2.plot(self.epochs, d_evo_loss, color='C1', linewidth=0.7)

        # Format
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('generator loss', color='C0')
        ax2.set_ylabel('discriminator loss', color='C1')
        ax1.tick_params(axis='y', labelcolor='C0')
        ax2.tick_params(axis='y', labelcolor='C1')

        fig.tight_layout()

        # Save plot
        plt.savefig(f'{self.plot_save_path}/losses-wrt-last.{self.image_file_format}')
        plt.close()
        
    def plot_logs(self, dataModule, model_dict, testing_restart=False):
        if not testing_restart and os.path.isfile(f'{self.plot_save_path}/losses.{self.image_file_format}'):
            print('losses already plotted, skipping step')
            return
        else:
            self.load_data(dataModule)
            self.loss()
            self.loss_wrt_last_model(model_dict)

class diffLogsPlotter(testDataset):
    def __init__(self, *args):        
        if type(args[0]) is testDataset:
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(*args)   
    
    def loss(self):
        """Logged losses"""
        # Load loss
        losses = self.losses['losses']
        epochs = self.losses['epochs']

        # Plot
        fig, ax = plt.subplots()

        ax.plot(epochs, losses, color='C0', linewidth=0.7)

        # Format
        ax.set_yscale('log')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')

        fig.tight_layout()

        # Save plot
        plt.savefig(f'{self.plot_save_path}/losses.{self.image_file_format}')
        plt.close()
    
    def plot_logs(self, testing_restart=False):
        if not testing_restart and os.path.isfile(f'{self.plot_save_path}/losses.{self.image_file_format}'):
            print('losses already plotted, skipping step')
            return
        else:
            self.loss()

###############################################################################################

"""Depreciated"""
class pointTester(testDataset):
    """
    Run tests for blob data
    """
    def __init__(self, *args):
        if type(args[0]) is testDataset:
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(*args)
            
    def images(self):
        """Plot generated images - grid of images, marginal sums, blob coordinates"""
        for n in tqdm(range(self.num_plots), desc='Plotting'):
            # Get images
            real_imgs_subset = self.real_imgs_subset[n*(self.grid_row_size**2):(n+1)*(self.grid_row_size**2)]
            gen_imgs_subset = self.all_gen_imgs[-1][n*(self.grid_row_size**2):(n+1)*(self.grid_row_size**2)] # Only use last model for this section
            
            # Plotting grid of images
            fig = plt.figure(figsize=(6,3))
            subfig = fig.subfigures(1, 2, wspace=0.2)
            
            plot_img_grid(subfig[0], real_imgs_subset, self.grid_row_size, title='Target Imgs')
            plot_img_grid(subfig[1], gen_imgs_subset, self.grid_row_size, title='Generated Imgs')
            
            # Save plot
            plt.savefig(f'{self.plot_save_path}/gen-imgs_{n}.{self.image_file_format}')
            plt.close()

