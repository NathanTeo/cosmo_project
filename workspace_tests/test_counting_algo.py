"""
Author: Nathan Teo

This scripts compares the counting algorithm output with the true counts and coordinates    
"""

# Params
generation_params = {
        'blob_num': 10,
        'num_distribution': 'poisson',
        'clustering': None,
        'blob_amplitude': 0.1,
        'amplitude_distribtuion': 'delta',
        'minimum_distance': None,
        'blob_size': 5,
        'image_size': 32,
        'sample_num': 50_000,
        'seed': 70,
        'noise': 0
        }

image_file_format = 'png'

should_plot_samples = False

###############################################################################################

# Import
import os
import numpy as np
import sys

project_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_project'
sys.path.append(project_path)
from code_model.testers.eval_utils import *

# Initiate params
blob_num = generation_params['blob_num']
num_distribution = generation_params['num_distribution']
clustering = init_param(generation_params, 'clustering')
blob_amplitude = init_param(generation_params, 'blob_amplitude', 1)
amplitude_distribution = init_param(generation_params, 'amplitude_distribtuion', 'delta')
min_dist = init_param(generation_params, 'minimum_distance')
gen_seed = generation_params['seed']
blob_size = generation_params['blob_size']
sample_num = generation_params['sample_num']
image_size = generation_params['image_size']
gen_noise = init_param(generation_params, 'noise', 0)

project_path = "C:/Users/Idiot/Desktop/Research/OFYP/cosmo"

data_file_name = 'bn{}{}-cl{}-is{}-bs{}-ba{}{}-md{}-sn{}-sd{}-ns{}'.format(
    blob_num, num_distribution[0], 
    '{:.0e}_{:.0e}'.format(*clustering) if clustering is not None else '_',
    image_size, blob_size, 
    '{:.0e}'.format(blob_amplitude), amplitude_distribution[0],
    min_dist, f'{sample_num:.0e}', gen_seed, gen_noise
)
data_folder = f'cosmo_data/{blob_num}_blob'
save_folder = 'misc_plots/counting_algorithm'

# Make save folders
if not os.path.exists(f'{project_path}/{save_folder}'):
    os.makedirs(f'{project_path}/{save_folder}')


if __name__=="__main__":
    # Load data
    true_counts = np.load(f'{project_path}/{data_folder}/{data_file_name}_counts.npy')
    true_centers = np.load(f'{project_path}/{data_folder}/{data_file_name}_coords.npy', allow_pickle=True)
    samples = np.load(f'{project_path}/{data_folder}/{data_file_name}.npy')

    # Truncate
    sample_size = 5000
    true_counts = true_counts[:sample_size]
    true_centers = true_centers[:sample_size]
    samples = samples[:sample_size]

    # Import functions
    import sys
    sys.path.append(project_path)
    from cosmo_project.code_model.testers.eval_utils import *
    
    # Example sample
    if should_plot_samples:
        samples_subset = samples[:4]
        counter = blobFitter(blob_size=blob_size, blob_amplitude=blob_amplitude)
        counter.load_samples(samples_subset)
        algo_centers, algo_counts = counter.count()
        
        fig, ax = plt.subplots(figsize=(3.5,3.5))
        
        # Plot sample images in grid
        # Img and peak coords 
        n = 2
        img = samples_subset[n]
        img_true_count = true_counts[n]
        img_algo_count = algo_counts[n]
        img_true_centers = np.array(true_centers[n])
        img_algo_centers = np.array(algo_centers[n])
        
        # Plot
        ax.imshow(img, interpolation='none', vmin=-0.05, vmax=None)
        
        if len(img_true_centers)>0: # skip if zero blob case    
            coords_x = img_true_centers[:, 1]
            coords_y = img_true_centers[:, 0]
            ax.scatter(coords_x, coords_y, c='black', marker='+', alpha=0.5, label='true')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
        if len(img_algo_centers)>0: # skip if zero blob case    
            coords_x = img_algo_centers[:, 1]
            coords_y = img_algo_centers[:, 0]
            ax.scatter(coords_x, coords_y, c='r', marker='x', alpha=0.5, label='fit')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
        ax.set_title('true count: {} | fit count: {}'.format(img_true_count, img_algo_count))
        
        # fig.text(.5, .96, 'count labelled above image', ha='center')
        plt.legend(bbox_to_anchor=(0.5, -0.07), loc='center', ncol=2)
                
        # Save plot
        plt.savefig(f'{save_folder}/counts-imgs.{image_file_format}')
        plt.close()

    if True:
        # Count with fitting
        file_counts_path = f'{project_path}/{save_folder}/counts-{sample_size}_{data_file_name}.npz'
        if os.path.exists(file_counts_path):
            print('previous counts found')
            
            file = np.load(file_counts_path, allow_pickle=True)
            
            algo_counts_1 = file['algo_counts_1']
            algo_counts_2 = file['algo_counts_2']
        elif min_dist is not None:
            # Real
            blob_threshold_rel = 0.6
            
            _, algo_counts_2, _ = samples_blob_counter_fast(
                samples, 
                blob_size=blob_size, min_peak_threshold=blob_amplitude*blob_threshold_rel,
                method='zero', progress_bar=True
                )
        else:
            # Count with gaussian decomp
            blob_threshold_rel = 0.7
            
            _, _, real_peak_vals = samples_blob_counter_fast(
                samples, 
                blob_size=blob_size, min_peak_threshold=(1/blob_num)*blob_threshold_rel,
                filter_sd=1,
                progress_bar=True
                )
            _, algo_counts_1 = count_blobs_from_peaks(real_peak_vals, blob_num)
            
            counter = blobFitter(blob_size=blob_size, blob_amplitude=(1/blob_num))
            counter.load_samples(samples)
            _, algo_counts_2 = counter.count()
            
            np.savez(
                file_counts_path, 
                algo_counts_1 = algo_counts_1,
                algo_counts_2 = algo_counts_2
            )
            
        # Plot
        all_counts = np.concatenate((true_counts, algo_counts_2))
        bins = np.arange(np.min(all_counts)-1.5, np.max(all_counts)+1.5, 1)

        fig = plt.figure(figsize=(4,3))
        fig.suptitle(f'Histogram of blob counts')
        plt.hist(true_counts, bins=bins, color='black', histtype='step', label='true', fill=True, facecolor=('black',0.1))
        # plt.hist(algo_counts_1, bins=bins, color=('orange', 0.7), histtype='step', label='algorithm-old')
        plt.hist(algo_counts_2, bins=bins, color=('red', 0.9), linestyle='dashed', histtype='step', label='fit')
        plt.xlabel('blob counts')
        plt.ylabel('sample counts')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{project_path}/{save_folder}/counting-algorithm-check-{sample_size}_{data_file_name}')
        plt.close()
        