"""
Author: Nathan Teo

This script makes plots for the poster    
"""

run = "diffusion_7e"
epoch = 299

real_color = '#DFD7E7'
model_color = '#F9B268'

real_label = 'target'
model_label = 'model'

what_counting = None # load or fast
tasks = ['powerspec'] # counthist, amphist, powerspec

image_file_format = 'png'

#########################################################################################
"""Initialize"""
# Import
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from utils.load_model import modelLoader

project_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_project'
sys.path.append(project_path)
from code_model.testers.eval_utils import *
from code_model.testers.plotting_utils import *
from code_data.utils import *

# Paths
root_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo'
save_path = f'{root_path}/misc_plots/poster/{run}'

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Plot
mpl.rcParams['figure.facecolor'] = 'none'  # Transparent figure background
mpl.rcParams['axes.facecolor'] = 'none'    # Transparent axes background

mpl.rcParams['axes.edgecolor'] = '#DFD7E7'     # Color of the axes borders
mpl.rcParams['axes.labelcolor'] = '#DFD7E7'   # Color of the x and y labels
mpl.rcParams['xtick.color'] = '#DFD7E7'      # Color of the x-ticks
mpl.rcParams['ytick.color'] = '#DFD7E7'      # Color of the y-ticks
mpl.rcParams['text.color'] = '#DFD7E7'

# Set default font sizes globally
mpl.rcParams['axes.labelsize'] = 24  # Font size for axis labels
mpl.rcParams['axes.titlesize'] = 28  # Font size for the title
mpl.rcParams['xtick.labelsize'] = 20 # Font size for x-tick labels
mpl.rcParams['ytick.labelsize'] = 20 # Font size for y-tick labels
mpl.rcParams['legend.fontsize'] = 20  # Font size for legend text
mpl.rcParams['font.size'] = 20  # Default font size for text in the plot

custom_font_path = "C:/Users/Idiot/Desktop/Research/OFYP/misc/fonts/BAHNSCHRIFT.TTF"
custom_font = mpl.font_manager.FontProperties(fname=custom_font_path)
mpl.rcParams['font.family'] = custom_font.get_name()

# Loader
model = modelLoader(run, epoch=epoch, real_color=real_color, model_color=model_color)

# Load samples
print('loading samples...', end='\t')
model.load_samples()

real_samples = model.real_samples
model_samples = model.model_samples
print('complete')

# Load counts
if what_counting=='load':
    print('loading counts...', end='\t')
    model.load_counts()

    real_counts = model.real_blob_counts
    real_count_mean = model.real_blob_num_mean
    model_counts = model.model_blob_counts
    model_count_mean = model.model_blob_num_mean
    print('complete')
elif what_counting=='fast':
    print('counting...')
    blob_size = model.generation_params['blob_size']
    blob_amplitude = model.generation_params['blob_amplitude']
    
    """Count blobs using gaussian decomp"""
    print('counting blobs...')
    blob_threshold_rel = 0.6

    # Real
    _, real_counts, real_amplitudes = samples_blob_counter_fast(
        real_samples, 
        blob_size=blob_size, min_peak_threshold=blob_amplitude*blob_threshold_rel,
        method='zero', progress_bar=True
        )

    # Generated
    _, model_counts, model_amplitudes = samples_blob_counter_fast(
        model_samples, 
        blob_size=blob_size, min_peak_threshold=blob_amplitude*blob_threshold_rel,
        method='zero', progress_bar=True
    )
    
    # Find mean number of blobs
    real_count_mean = np.mean(real_counts)
    model_count_mean = np.mean(model_counts)
    print(f'mean number of target peaks: {real_count_mean}')
    print(f'mean number of generated peaks: {model_count_mean}')
else:
    pass

# Get image size
image_size = model_samples[0].shape[0]

"""Plot"""
print('plotting...')
'Sample'
print('task: samples', end='\t\t')
for i in range(5):
    fig = plt.figure(figsize=(10,10))
    plt.imshow(real_samples[i])
    plt.axis('off')
    plt.savefig(f'{save_path}/real_samples_{i}')
    plt.close()
    
    fig = plt.figure(figsize=(10,10))
    plt.imshow(model_samples[i])
    plt.axis('off')
    plt.savefig(f'{save_path}/model_samples_{i}')
    plt.close()
print('complete')

'Count histogram'
if 'counthist' in tasks:
    print('task: count histogram', end='\t')
    # Create figure
    fig = plt.figure(figsize=(7,5))

    # Bins for histogram
    bins = find_good_bins([real_counts, model_counts], method='arange',
                            spacing=(1.5,1.5))

    # Plot histogram
    real_hist, _, _ = plt.hist(real_counts, bins=bins, 
            histtype='step', label=real_label, color=(real_color,0.8), linewidth=4)
    plt.axvline(real_count_mean, color=(real_color,0.5), linestyle='dashed', linewidth=4)

    model_hist, _, _ = plt.hist(
        model_counts, bins=bins,
        histtype='step', label=model_label,
        color=(model_color,0.5), linewidth=3, linestyle='-',
        facecolor=(model_color,0.2), fill=True
        )
    plt.axvline(model_count_mean, color=(model_color,0.5), linestyle='dashed', linewidth=4)

    _, max_ylim = plt.ylim()
    plt.text(real_count_mean*1.05, max_ylim*0.9,
            'Mean: {:.2f}'.format(real_count_mean), color=(real_color,1))
    plt.text(model_count_mean*1.05, max_ylim*0.8,
            'Mean: {:.2f}'.format(model_count_mean), color=(model_color,1))

    # Format
    ax = plt.gca()  # Get the current axis
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(2, 2))
    if len(bins)<5:
        plt.xticks(bins[:-1]+0.5)
    plt.ylabel('sample count')
    plt.xlabel('blob count')
    # plt.suptitle(f"Histogram of blob count")
    plt.legend()
    plt.tight_layout()

    # Save
    plt.savefig(f'{save_path}/number-blobs-histogram.{image_file_format}')
    plt.close()
    print('complete')

'Power spectrum'
if 'powerspec' in tasks:
    print("task: power spectrum")
    # Get Cls
    image_size_angular = 1
    delta_ell = 500
    max_ell = 12000
    ell2d = ell_coordinates(image_size, image_size_angular/image_size)
    
    real_cl, real_err, bins = power_spectrum_stack(real_samples, 
                                                    delta_ell, max_ell, ell2d, image_size_angular,
                                                    progress_bar=True)
    model_cl, model_err, bins = power_spectrum_stack(model_samples, 
                                                    delta_ell, max_ell, ell2d, image_size_angular,
                                                    progress_bar=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7,5))
    
    # Plot
    plot_smooth_line(
        ax, real_cl, bins, real_err, 
        color=((real_color,1), (real_color,0.5)),
        linewidth=1, elinewidth=4, capsize=8, fmt='o',
        label='target', scale='semilog_x', errorbars=True, line=False
    )
    plot_smooth_line(
        ax, model_cl, bins, model_err, 
        color=((model_color,1), (model_color,1)),
        linewidth=1, elinewidth=2, capsize=4, fmt='.',
        label='model', scale='semilog_x', errorbars=True, line=False
    )
    
    # Format
    # fig.suptitle(f"Power spectrum")
    plt.xlabel(r'$l$')
    plt.ylabel(r'$C_l$')
    plt.xlim(np.min(bins)*.9,np.max(bins)*1.1)
    ax = plt.gca()  # Get the current axis
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))
    plt.tight_layout()
    
    plt.legend()

    # Save
    plt.savefig(f'{save_path}/power-spec.{image_file_format}')
    plt.close() 
    
    print('complete')

'Amp distribution'
if 'amphist' in tasks:
    print("task: amplitude histogram", end='\t')
    # Create figure
    fig = plt.figure(figsize=(7,5))
    
    real_amplitudes_concat = np.concatenate(real_amplitudes)
    model_amplitudes_concat = np.concatenate(model_amplitudes)
    
    # Bins for histogram
    bins = np.arange(0, 10+1, 0.5)
    
    # Plot histogram
    real_hist, _, _ = plt.hist(real_amplitudes_concat, bins=bins, 
            histtype='step', label=real_label, color=(real_color,0.8), linewidth=4)
    model_hist, _, _ = plt.hist(model_amplitudes_concat, bins=bins, 
        histtype='step', label=model_label,
        color=(model_color,0.5), linewidth=3, linestyle='-',
        facecolor=(model_color,0.2), fill=True
        )
    
    # Format
    ax = plt.gca()  # Get the current axis
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(2, 2))
    plt.ylabel('blob count')
    plt.xlabel('blob amplitude')
    # plt.suptitle(f"Histogram of blob amplitude")
    plt.legend()
    plt.tight_layout()

    # Save
    plt.savefig(f'{save_path}/amplitude-blobs-histogram.{image_file_format}')
    plt.close()
    
    print('complete')

print('complete')