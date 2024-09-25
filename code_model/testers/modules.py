import os

import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
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

        self.training_seed = training_params['random_seed']
        self.lr = training_params['lr']

        self.blob_num = generation_params['blob_num']
        self.data_distribution = generation_params['distribution']
        self.generation_seed = generation_params['seed']
        self.blob_size = generation_params['blob_size']
        self.real_sample_num = generation_params['sample_num']
        self.image_size = training_params['image_size']
        self.gen_noise = generation_params['noise']

        network_params = training_params['network_params']
        self.training_noise = training_params['noise']
        self.batch_size = training_params['batch_size']
        self.num_workers = training_params['num_workers']
        self.max_epochs = training_params['max_epochs']
        self.avail_gpus = training_params['avail_gpus']
        
        if 'GAN' in self.model_version:
            gen_version = training_params['generator_version']
            dis_version = training_params['discriminator_version']
            
            latent_dim = network_params['latent_dim']
            gen_img_w = network_params['generator_img_w']
            gen_upsamp = network_params['generator_upsamp_size']
            dis_conv = network_params['discriminator_conv_size']
            dis_lin = network_params['discriminator_linear_size']
            
            self.model_name = '{}-g{}-d{}-bn{}{}-bs{}-sn{}-is{}-ts{}-lr{}-ld{}-gw{}-gu{}-dc{}-dl{}-ns{}'.format(
                self.model_version,
                gen_version, dis_version,
                self.blob_num, self.data_distribution[0], self.blob_size, "{:.0e}".format(self.real_sample_num), self.image_size,
                self.training_seed, "{:.0e}".format(self.lr),
                latent_dim, gen_img_w, gen_upsamp, dis_conv, dis_lin,
                str(self.training_noise[1])[2:] if self.training_noise is not None else '_'
            )
        elif 'Diffusion' in self.model_version:
            unet_version = training_params['unet_version']
            
            noise_steps = ['noise_steps']
            time_dim = ['time_dim']
            initial_size = ['ininial_size']
    
            self.model_name = '{}-n{}-bn{}{}-bs{}-sn{}-is{}-ts{}-lr{}-st{}-td{}-sz{}-ns{}'.format(
                self.model_version,
                unet_version,
                self.blob_num, self.data_distribution[0], self.blob_size, "{:.0g}".format(self.real_sample_num), self.image_size,
                self.training_seed, "{:.0g}".format(self.lr),
                noise_steps, time_dim, initial_size, 
                str(self.training_noise[1])[2:] if self.training_noise is not None else '_'
            )
        self.training_params['model_name'] = self.model_name
        
        self.grid_row_num = testing_params['grid_row_num']
        self.plot_num = testing_params['plot_num']
        self.subset_sample_num = int(testing_params['subset_sample_num'])
        self.loss_zoom_bounds = testing_params['loss_zoom_bounds']
        self.counting_params = testing_params['counting_params']
        self.testing_seed = testing_params['seed']
        
        """Paths"""
        self.root_path = training_params['root_path']
        self.data_path = 'data'
        self.data_file_name = f'bn{self.blob_num}{self.data_distribution[0]}-is{self.image_size}-bs{self.blob_size}-sn{self.real_sample_num}-sd{self.generation_seed}-ns{int(self.gen_noise)}.npy'
        self.chkpt_path = 'checkpoints'
        self.log_path = 'logs'
        self.plot_save_path = f'{self.root_path}/plots/images'
        self.output_save_path = f'{self.root_path}/plots/model_output'
        
        if not os.path.exists(self.plot_save_path):
            os.makedirs(self.plot_save_path)

        if not os.path.exists(self.output_save_path):
            os.makedirs(self.output_save_path)
        
        """Epochs"""
        # Load loss
        self.losses = np.load(f'{self.root_path}/{self.log_path}/losses.npz')
        self.epoch_last = self.losses['epochs'][-1]
        
        """Plotting style"""
        self.real_color = 'black'
        self.gen_color = 'red'
        # Need to change if different number of models are plotted, find a way to make this automatic/input?
        self.hist_alphas = [0.3, 0.7, 0.5]
        self.line_alphas = [0.3, 0.7, 0.9]
        self.select_last_epoch = [False, False, True]
        
        """Initialize seed"""
        torch.manual_seed(self.testing_seed)
        
    def load_data(self, DataModule):
        """Load real data"""
        self.real_imgs = np.load(f'{self.root_path}/{self.data_path}/{self.data_file_name}')[:self.subset_sample_num]
        self.data = DataModule(
            data_file=f'{self.root_path}/{self.data_path}/{self.data_file_name}',
            batch_size=self.batch_size, num_workers=self.num_workers,
            truncate_ratio=self.subset_sample_num/self.real_sample_num
            )
    
    def load_models(self, model_dict):
        """Load models"""
        # Get checkpoints of models to be tested
        filenames = os.listdir(f'{self.root_path}/{self.chkpt_path}')
        filenames.sort()
        self.filenames = [filenames[int(len(filenames)/4)], filenames[int(len(filenames)/2)], 'last.ckpt']
        
        self.model_epochs = [int(file[6:-5]) for file in self.filenames[:-1]]
        self.model_epochs.append(self.epoch_last)
        
        self.models = [model_dict[self.model_version].load_from_checkpoint(
            f'{self.root_path}/{self.chkpt_path}/{file}',
            **self.training_params
            ) for file in filenames]
    
    def generate_images(self, testing_restart):
        """Generate images given a list of models"""
        self.all_gen_imgs = []
        try:
            if not testing_restart:
                # Load saved outputs if available
                print('Model output found')
                for filename in self.filenames:
                    self.all_gen_imgs.append(np.load(
                        '{}/{}_{:.0g}.npy'.format(
                            self.output_save_path,
                            filename[:-5],
                            self.subset_sample_num
                    )))
                print('Model output loaded')
                return self.all_gen_imgs
            else:
                # Retest model
                print('Retesting model...')
                trainer = pl.Trainer()
                for model, filename, epoch in zip(self.models, self.filenames, self.model_epochs):
                    print(f'generating images, epoch {epoch}')
                    trainer.test(model, self.data)
                    
                    # Add outputs to list
                    self.all_gen_imgs.append(model.outputs)
                
                    # Save outputs
                    np.save('{}/{}_{:.0g}.npy'.format(
                        self.output_save_path, filename[:-5], self.subset_sample_num
                        ), model.outputs)
                
                return self.all_gen_imgs
            
        except FileNotFoundError:
            # Test model if no saved outputs are found
            trainer = pl.Trainer()
            for model, filename, epoch in zip(self.models, self.filenames, self.model_epochs):
                print(f'generating images, epoch {epoch}')
                trainer.test(model, self.data)
                
                # Add outputs to list
                self.all_gen_imgs.append(model.outputs)
            
                # Save outputs
                np.save('{}/{}_{:.0g}.npy'.format(
                    self.output_save_path, filename[:-5], self.subset_sample_num
                    ), model.outputs)
        
            return self.all_gen_imgs
    
    def truncate(self):
        """Make subset used for testing"""
        self.real_imgs_subset = self.real_imgs[:self.subset_sample_num]
        self.all_gen_imgs_subset = [gen_imgs[:self.subset_sample_num] for gen_imgs in self.all_gen_imgs]
    
    def prep_data(self, dataModule, model_dict, testing_restart=False):
        """Run all steps to prepare data"""
        print('loading data and models...')
        self.load_data(dataModule)
        self.load_models(model_dict)
        print('loading complete')
        self.generate_images(testing_restart)
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
        
        self.blob_amplitude = 1/self.blob_num
    
    def images(self):
        """Plot generated images - grid of images, marginal sums, blob coordinates"""
        for n in tqdm(range(self.plot_num), desc='Plotting'):
            # Get images
            real_imgs_subset = self.real_imgs[n*(self.grid_row_num**2):(n+1)*(self.grid_row_num**2)]
            gen_imgs_subset = self.all_gen_imgs[-1][n*(self.grid_row_num**2):(n+1)*(self.grid_row_num**2)] # Only use last model for this section
            
            # Plotting grid of images
            fig = plt.figure(figsize=(6,3))
            subfig = fig.subfigures(1, 2, wspace=0.2)
            
            plot_img_grid(subfig[0], real_imgs_subset, self.grid_row_num, title='Real Imgs')
            plot_img_grid(subfig[1], gen_imgs_subset, self.grid_row_num, title='Generated Imgs')
            
            # Save plot
            plt.savefig(f'{self.plot_save_path}/gen-imgs_{n}_{self.model_name}.png')
            plt.close()

            # Plotting marginal sums
            real_marginal_sums = [marginal_sums(real_imgs_subset[i]) for i in range(self.grid_row_num**2)]
            gen_marginal_sums = [marginal_sums(gen_imgs_subset[i]) for i in range(self.grid_row_num**2)]
            
            fig = plt.figure(figsize=(4,6))
            subfig = fig.subfigures(1, 2)
            
            plot_marginal_sums(real_marginal_sums, subfig[0], self.grid_row_num, title='Real')
            plot_marginal_sums(gen_marginal_sums, subfig[1], self.grid_row_num, title='Generated')
            
            # Format
            fig.suptitle('Marginal Sums')
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plt.savefig(f'{self.plot_save_path}/marg-sums_{n}_{self.model_name}.png')
            plt.close()
           
            # Fitting blobs
            counter = blobCounter(blob_size=self.blob_size, blob_amplitude=self.blob_amplitude)
            
            counter.load_samples(real_imgs_subset)
            real_blob_coords, real_blob_counts = counter.count(progress_bar=False)
            
            counter.load_samples(gen_imgs_subset)
            gen_blob_coords, gen_img_blob_counts = counter.count(progress_bar=False)

            fig = plt.figure(figsize=(6,3))
            subfig = fig.subfigures(1, 2, wspace=0.2)
            
            plot_peak_grid(subfig[0], real_imgs_subset, real_blob_coords, self.grid_row_num, 
                            title='real imgaes', subplot_titles=real_blob_counts)
            plot_peak_grid(subfig[1], gen_imgs_subset, gen_blob_coords, self.grid_row_num, 
                            title='generated imgaes', subplot_titles=gen_img_blob_counts)
            
            fig.text(.5, .03, 'number of blobs labelled above image', ha='center')
            
            # Save plot
            plt.savefig(f'{self.plot_save_path}/counts-imgs_{n}_{self.model_name}.png')
            plt.close()

    def stack(self):
        """Stack images"""
        'Stack'
        print('stacking...')
        stacked_real_img = stack_imgs(self.real_imgs_subset)
        stacked_gen_img = stack_imgs(self.all_gen_imgs_subset[-1]) # Only stack for last model

        'Plot image'
        # Create figure
        fig, axs = plt.subplots(1, 2)

        # Plot
        plot_stacked_imgs(axs[0], stacked_real_img, title=f"real\n{self.subset_sample_num} samples")
        plot_stacked_imgs(axs[1], stacked_gen_img, title=f"generated\n{self.subset_sample_num} samples")

        # Format
        fig.suptitle('Stacked Image')
        plt.tight_layout()

        # Save plot
        plt.savefig(f'{self.plot_save_path}/stack_{self.model_name}.png')
        plt.close()
        
        'Stacked image histogram'
        # Create figure
        fig = plt.figure()

        # Plot
        plt.hist(stacked_real_img.ravel(), histtype='step', label='real', color=(self.real_color,0.8))
        plt.hist(stacked_gen_img.ravel(), histtype='step', label='generated', color=(self.gen_color,0.8))

        # Format
        plt.ylabel('image count')
        plt.xlabel('stacked pixel value')
        plt.suptitle(f"stack of {self.subset_sample_num} samples")
        plt.legend()
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/stack-histogram-img_{self.model_name}.png')
        plt.close()

    def count_blobs_fast(self):
        """Count blobs only using gaussian decomp"""
        print('counting blobs...')
        filter_sd = self.counting_params[0]
        blob_threshold_rel = self.counting_params[1]

        # Real
        self.real_blob_coords, self.real_blob_nums, self.real_peak_vals = imgs_blob_finder(
            self.real_imgs_subset, 
            blob_size=self.blob_size, min_peak_threshold=(1/self.blob_num)*blob_threshold_rel,
            filter_sd=filter_sd,
            progress_bar=True
            )
        self.real_indv_peak_counts, self.real_blob_counts = count_blobs_from_peaks(self.real_peak_vals, self.blob_num)

        # Generated
        self.all_gen_blob_coords, self.all_gen_blob_nums, self.all_gen_peak_vals = map(
            list, zip(*[imgs_blob_finder(
                subset, 
                blob_size=self.blob_size, min_peak_threshold=(1/self.blob_num)*blob_threshold_rel,
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
        print(f'mean number of real peaks: {self.real_blob_num_mean}')
        print(f'mean number of generated peaks: {self.all_gen_blob_num_mean[-1]}')

    def count_blobs(self):
        """Count blobs by fitting"""
        print('counting blobs...')
        
        # Initialize counter
        counter = blobCounter(blob_size=self.blob_size, blob_amplitude=self.blob_amplitude)
        
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
     
    def blob_num_stats(self):
        """Number of blob analysis"""
        'Histogram'
        # Create figure
        fig = plt.figure()

        # Bins for histogram
        concat_blob_counts = self.real_blob_counts
        for blob_counts in self.all_gen_blob_counts: 
            concat_blob_counts = np.concatenate([concat_blob_counts, blob_counts])
        
        min = np.min(concat_blob_counts)
        max = np.max(concat_blob_counts)
        
        bins = np.arange(min-1.5, max+1.5,1)
        
        # Plot histogram
        for i, blob_counts in enumerate(self.all_gen_blob_counts):
            plt.hist(blob_counts, bins=bins,
                    histtype='step', label=f'epoch {self.model_epochs[i]}',
                    color=(self.gen_color,self.hist_alphas[i]), linewidth=set_linewidth(i, len(self.models)),
                    fill=self.select_last_epoch[i]
                    )
        plt.axvline(self.all_gen_blob_num_mean[-1], color=(self.gen_color,0.5), linestyle='dashed', linewidth=1) # Only label mean for last model
        
        plt.hist(self.real_blob_counts, bins=bins, 
                    histtype='step', label='real', color=(self.real_color,0.8))
        plt.axvline(self.real_blob_num_mean, color=(self.real_color,0.5), linestyle='dashed', linewidth=1)

        _, max_ylim = plt.ylim()
        plt.text(self.real_blob_num_mean*1.05, max_ylim*0.9,
                'Mean: {:.2f}'.format(self.real_blob_num_mean), color=(self.real_color,1))
        plt.text(self.all_gen_blob_num_mean[-1]*1.05, max_ylim*0.8,
                    'Mean: {:.2f}'.format(self.all_gen_blob_num_mean[-1]), color=(self.gen_color,1))

        # Format
        plt.ylabel('image count')
        plt.xlabel('blob count')
        plt.suptitle(f"Histogram of number of blobs, {self.subset_sample_num} samples")
        plt.legend()
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/number-blobs-histogram_{self.model_name}.png')
        plt.close()    

        'Extreme number of blobs'
        # Create figure
        fig = plt.figure(figsize=(3,5))
        subfig = fig.subfigures(1, 2, wspace=0.2)

        # Plot
        plot_extremum_num_blobs(
            subfig[0], self.real_imgs_subset,
            self.real_blob_coords, self.real_blob_counts,
            extremum='min', title='real'
            )
        plot_extremum_num_blobs(
            subfig[1], self.all_gen_imgs_subset[-1],
            self.all_gen_blob_coords[-1], self.all_gen_blob_counts[-1],
            extremum='min', title='generated'
            )  # Only perform for last model

        # Format
        fig.suptitle(f"Min blob count, {self.subset_sample_num} samples")
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/min-peak_{self.model_name}.png')
        plt.close()

        # Create figure
        fig = plt.figure(figsize=(3,5))
        subfig = fig.subfigures(1, 2, wspace=0.2)

        # Plot
        plot_extremum_num_blobs(
            subfig[0], self.real_imgs_subset,
            self.real_blob_coords, self.real_blob_counts,
            extremum='max', title='real'
            )
        plot_extremum_num_blobs(
            subfig[1], self.all_gen_imgs_subset[-1],
            self.all_gen_blob_coords[-1], self.all_gen_blob_counts[-1],
            extremum='max', title='generated'
            )  

        # Format
        fig.suptitle(f"Max blobs count, {self.subset_sample_num} samples")
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/max-peak_{self.model_name}.png')
        plt.close()

        'Total flux histogram'
        # Find flux
        print("calculating total flux...")
        real_img_fluxes = find_total_fluxes(self.real_imgs_subset)
        all_gen_img_fluxes = [find_total_fluxes(subset) for subset in self.all_gen_imgs_subset]

        # Create figure
        fig = plt.figure()

        # Plot
        for i, fluxes in enumerate(all_gen_img_fluxes):
            plt.hist(fluxes, 
                    histtype='step', label=f'epoch {self.model_epochs[i]}', 
                    color=(self.gen_color,self.hist_alphas[i]), linewidth=set_linewidth(i, len(self.models)),
                    fill=self.select_last_epoch[i]
                    )
        plt.hist(real_img_fluxes, 
                    histtype='step', label='real', color=(self.real_color,0.8))
 
        # Format   
        plt.ylabel('image count')
        plt.xlabel('total flux')
        plt.suptitle(f"Histogram of total flux, {self.subset_sample_num} samples")
        plt.legend()
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/total-flux-histogram_{self.model_name}.png')
        plt.close()

    def two_point_correlation(self):
        """2-point correlation analysis"""
        print('calculating 2 point correlation...')
        real_corrs, real_errs, edges = two_point_stack(self.real_blob_coords, self.image_size, bins=20, progress_bar=True)
        all_gen_corrs, all_gen_errs, _ = map(
            list, zip(*[two_point_stack(
                blob_coords, self.image_size, bins=20, progress_bar=True
                ) for blob_coords in self.all_gen_blob_coords])
        )

        # Create figure
        fig, ax = plt.subplots()

        # Plot
        plot_two_point(
            ax, real_corrs, edges, real_errs, 
            color=((self.real_color,1), (self.real_color,0.5)),
            label='real'
        )
         
        for i, (corrs, errs) in enumerate(zip(all_gen_corrs, all_gen_errs)):
            plot_two_point(
                ax, corrs, edges, errs, 
                color=((self.gen_color,self.line_alphas[i]), (self.gen_color,0.5)), linewidth=set_linewidth(i, len(all_gen_corrs)),
                label=f'epoch {self.model_epochs[i]}', errorbars=self.select_last_epoch[i]
            )
        
        # Format
        fig.suptitle(f"2-point correlation, {self.subset_sample_num} samples")
        plt.xlabel('pair distance')
        plt.ylabel('2-point correlation')
        plt.tight_layout()
        
        plt.legend(loc='lower right')

        # Save
        plt.savefig(f'{self.plot_save_path}/2-pt-corr_{self.model_name}.png')
        plt.close() 

    def pixel_stats(self):
        """Pixel value analysis"""
        'Single image histogram'
        histogram_num = 20
        real_imgs_subset_sh = self.real_imgs_subset[:histogram_num]
        gen_imgs_subset_sh = self.all_gen_imgs_subset[-1][:histogram_num]

        # Create figure
        fig, axs = plt.subplots(1,2)

        # Plot
        plot_pixel_histogram(axs[0], real_imgs_subset_sh, color=(self.real_color,0.5), bins=20)
        plot_pixel_histogram(axs[1], gen_imgs_subset_sh, color=(self.gen_color,0.5), bins=20)

        # Format
        fig.suptitle(f"Histogram of pixel values, {histogram_num} samples")
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/pixel-histogram_{self.model_name}.png')
        plt.close()

        'Stacked histogram'
        print('stacking histograms...')
        real_hist_stack = stack_histograms(self.real_imgs_subset)
        all_gen_hist_stack = [stack_histograms(subset) for subset in self.all_gen_imgs_subset]

        # Create figure
        fig, ax = plt.subplots()

        # Plot
        fill = [None, None, (self.gen_color, self.hist_alphas[-1])]
        for i, hist_stack in enumerate(all_gen_hist_stack):
            plot_histogram_stack(
                ax, *hist_stack,
                color=(self.gen_color,self.hist_alphas[i]), linewidth=set_linewidth(i, len(self.models)),
                fill_color=fill[i],
                label=f'epoch {self.model_epochs[i]}'
                )

        plot_histogram_stack(ax, *real_hist_stack, color=(self.real_color,0.8), label='real')

        # Format
        plt.ylabel('image count')
        plt.xlabel('pixel value')
        fig.suptitle(f"Stacked histogram of pixel values, {self.subset_sample_num} samples")
        plt.legend()
        plt.tight_layout()

        # Save
        plt.savefig(f'{self.plot_save_path}/pixel-stack-histogram_{self.model_name}.png')
        plt.close()
    
    def test(self):
        """Run all testing methods"""
        self.images()
        self.stack()
        if os.path.exists(f'{self.output_save_path}/counts.npz'):
            print('previous counts found, loading counts...')
            self.load_counts()
        else:
            self.count_blobs()
            self.save_counts()
        self.blob_num_stats()
        self.two_point_correlation()
        self.pixel_stats()
        
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
        for n in tqdm(range(self.plot_num), desc='Plotting'):
            # Get images
            real_imgs_subset = self.real_imgs[n*(self.grid_row_num**2):(n+1)*(self.grid_row_num**2)]
            gen_imgs_subset = self.all_gen_imgs[-1][n*(self.grid_row_num**2):(n+1)*(self.grid_row_num**2)] # Only use last model for this section
            
            # Plotting grid of images
            fig = plt.figure(figsize=(6,3))
            subfig = fig.subfigures(1, 2, wspace=0.2)
            
            plot_img_grid(subfig[0], real_imgs_subset, self.grid_row_num, title='Real Imgs')
            plot_img_grid(subfig[1], gen_imgs_subset, self.grid_row_num, title='Generated Imgs')
            
            # Save plot
            plt.savefig(f'{self.plot_save_path}/gen-imgs_{self.model_name}_{n}.png')
            plt.close()

class ganLogsPlotter(testDataset):
    def __init__(self, *args):        
        if type(args[0]) is testDataset:
            self.__dict__ = args[0].__dict__.copy()
        else:
            super().__init__(*args)
    
    def load_last_model(self, model_dict):
        self.last_model = model_dict[self.model_version].load_from_checkpoint(
            f'{self.root_path}/{self.chkpt_path}/last.ckpt',
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
        plt.savefig(f'{self.plot_save_path}/losses_{self.model_name}.png')
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
        plt.savefig(f'{self.plot_save_path}/losses-zm_{self.model_name}.png')
        plt.close()

    def score_models(self, model_dict):
        # Load models
        filenames = os.listdir(f'{self.root_path}/{self.chkpt_path}')
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
                f'{self.root_path}/{self.chkpt_path}/{file}',
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
        plt.savefig(f'{self.plot_save_path}/losses-wrt-last_{self.model_name}.png')
        plt.close()
        
    def plot_logs(self, dataModule, model_dict, testing_restart=False):
        if not testing_restart and os.path.isfile(f'{self.plot_save_path}/losses_{self.model_name}.png'):
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
        plt.savefig(f'{self.plot_save_path}/losses_{self.model_name}.png')
        plt.close()
    
    def plot_logs(self, dataModule, testing_restart=False):
        if not testing_restart and os.path.isfile(f'{self.plot_save_path}/losses_{self.model_name}.png'):
            print('losses already plotted, skipping step')
            return
        else:
            self.load_data(dataModule)
            self.loss()
