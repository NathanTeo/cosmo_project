import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm.auto import tqdm
from utils.load_model import modelLoader

project_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo'
sys.path.append(f'{project_path}/cosmo_project')
from code_model.testers.eval_utils import *
from code_model.testers.plotting_utils import *
from code_data.utils import *

run = "cwgan_6e"
grid_size = 2

#########################################################################################
# Paths
save_path = f'{project_path}/misc_plots/inspect_checkpoints/{run}'
chkpt_path = f"C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_runs/{run}/checkpoints"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load samples
loader = modelLoader(run)

# Get checkpoints of models to be tested
filenames = os.listdir(chkpt_path)
filenames.sort()
print("checkpoints:")
print(filenames)

# Load models
models = loader.load_models(filenames)
all_samples, epochs = [], []
for i, model in enumerate(models):
    if 'diff' in run:    
        print(f'model {i+1}/{len(models)}', end= ' ')
    elif 'gan' in run:
        print(f'model {i+1}/{len(models)}')
    all_samples.append(loader.generate(model, int(grid_size**2)))

# Plot
titles = [file[:-4] for file in filenames]
for samples, title in tqdm(zip(all_samples, titles)):
    fig = plt.figure(figsize=(4,4))
    plot_img_grid(fig, samples, grid_size, title)
    plt.savefig(f'{save_path}/{title}')
    plt.close()