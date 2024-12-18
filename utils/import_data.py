"""
Author: Nathan Teo

Copies run data from compiled data folder to run folder.
"""

import sys
import os
import platform

run = input('run folder: ')

sys.path.append(f'./cosmo_runs/{run}')
from config.model_params import generation_params

def init_param(config, param, default=None):
    """Initialize parameter and return default value if key is not found in config dictionary"""
    try:
        return config[param]
    except KeyError:
        return default

blob_num = generation_params['blob_num']
num_distribution = generation_params['num_distribution']
clustering = init_param(generation_params, 'clustering')
blob_amplitude = generation_params['blob_amplitude']
amplitude_distribution = generation_params['amplitude_distribution']
generation_seed = generation_params['seed']
blob_size = generation_params['blob_size']
sample_num = generation_params['sample_num']
image_size = generation_params['image_size']
gen_noise = generation_params['noise']
min_dist = init_param(generation_params, 'minimum_distance')

data_source_path = f'cosmo_data/{blob_num}_blob'
data_dest_path = f'cosmo_runs/{run}'
data_file_name = 'bn{}{}-cl{}-is{}-bs{}-ba{}{}-md{}-sn{}-sd{}-ns{}'.format(
        blob_num, num_distribution[0], 
        '{:.0e}_{:.0e}'.format(*clustering) if clustering is not None else '_',
        image_size, blob_size, 
        '{:.0e}'.format(blob_amplitude), amplitude_distribution[0],
        min_dist, f'{sample_num:.0e}', generation_seed, int(gen_noise)
)

source = f'./{data_source_path}'
dest = f'./{data_dest_path}/data'

if not os.path.exists(dest):
        os.makedirs(dest)

if platform.system()=='Windows':
        os.system(f'robocopy {source} {dest} {data_file_name}.npy /e')
elif platform.system()=='Linux':
        os.system(f'cp {source}/{data_file_name}.npy {dest}')

print('data moved')