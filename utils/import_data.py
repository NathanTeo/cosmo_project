import sys
import os
import platform

run = 'bug_finder'
root_path = "/Users/Idiot/Desktop/Research/OFYP/cosmo"

sys.path.append(f'{root_path}/cosmo_runs/{run}')

from config.model_params import generation_params

blob_num = generation_params['blob_num']
data_distribution = generation_params['distribution']
generation_seed = generation_params['seed']
blob_size = generation_params['blob_size']
sample_num = generation_params['sample_num']
image_size = generation_params['image_size']
gen_noise = generation_params['noise']

data_source_path = f'cosmo_data/{blob_num}_blob'
data_dest_path = f'cosmo_runs/{run}'
data_file_name = f'bn{blob_num}{data_distribution[0]}-is{image_size}-bs{blob_size}-sn{sample_num}-sd{generation_seed}-ns{int(gen_noise)}.npy'

source = f'{root_path}/{data_source_path}'
dest = f'{root_path}/{data_dest_path}/data'

if not os.path.exists(dest):
        os.makedirs(dest)

os.system(f'robocopy {source} {dest} {data_file_name} /e')

print('data moved')