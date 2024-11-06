import os
import numpy as np

# Params
generation_params = {
        'blob_num': 10,
        'distribution': 'poisson',
        'clustering': None,
        'blob_size': 5,
        'image_size': 32,
        'sample_num': 50000,
        'seed': 70,
        'noise': 0
        }

blob_threshold_rel = 0.7
sample_size = 5000

# Initiate params
blob_num = generation_params['blob_num']
data_distribution = generation_params['distribution']
clustering = generation_params['clustering']
generation_seed = generation_params['seed']
blob_size = generation_params['blob_size']
real_sample_num = generation_params['sample_num']
image_size = generation_params['image_size']
gen_noise = generation_params['noise']

project_path = "C:/Users/Idiot/Desktop/Research/OFYP/cosmo"
data_file_name = 'bn{}{}-cl{}-is{}-bs{}-sn{}-sd{}-ns{}'.format(
    blob_num, data_distribution[0], 
    '{:.0e}_{:.0e}'.format(*clustering) if clustering is not None else '_',
    image_size, blob_size, real_sample_num,
    generation_seed, gen_noise
)
data_folder = f'cosmo_data/{blob_num}_blob'
save_folder = 'misc_plots/counting_algorithm'

# Make save folders
if not os.path.exists(f'{project_path}/{save_folder}'):
    os.makedirs(f'{project_path}/{save_folder}')

if __name__=="__main__":
    # Load data
    true_counts = np.load(f'{project_path}/{data_folder}/{data_file_name}_counts.npy')
    samples = np.load(f'{project_path}/{data_folder}/{data_file_name}.npy')

    # Truncate
    true_counts= true_counts[:sample_size]
    samples = samples[:sample_size]

    # Import functions
    import sys
    sys.path.append(project_path)
    from cosmo_project.code_model.testers.eval_utils import *

    # Count with fitting
    file_counts_path = f'{project_path}/{save_folder}/counts-{sample_size}_{data_file_name}.npz'
    if os.path.exists(file_counts_path):
        print('previous counts found')
        
        file = np.load(file_counts_path, allow_pickle=True)
        
        algo_counts_1 = file['algo_counts_1']
        algo_counts_2 = file['algo_counts_2']
    else:
        # Count with gaussian decomp
        _, _, real_peak_vals = imgs_blob_finder(
            samples, 
            blob_size=blob_size, min_peak_threshold=(1/blob_num)*blob_threshold_rel,
            filter_sd=1,
            progress_bar=True
            )
        _, algo_counts_1 = count_blobs_from_peaks(real_peak_vals, blob_num)
        
        counter = blobCounter(blob_size=blob_size, blob_amplitude=(1/blob_num))
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

    fig = plt.figure(figsize=(5,3.5))
    fig.suptitle(f'samples: {sample_size}')
    plt.hist(true_counts, bins=bins, color='black', histtype='step', label='true')
    # plt.hist(algo_counts_1, bins=bins, color=('orange', 0.7), histtype='step', label='algorithm-old')
    plt.hist(algo_counts_2, bins=bins, color=('red', 0.9), linestyle='dashed', histtype='step', label='algorithm')
    plt.xlabel('blob counts')
    plt.ylabel('image counts')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{project_path}/{save_folder}/counting-algorithm-check-{sample_size}_{data_file_name}')
    plt.close()