import sys
import os

project_path = "C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_project"
sys.path.append(project_path)

from code_model.testers.eval_utils import *
from code_model.testers.plotting_utils import *
from code_data.utils import *

# Params
"""Params"""
generation_params = {
    'image_size': 64,
    'seed': 70,
    'blob_size': 5,
    'blob_amplitude': 1,
    'amplitude_distribution': 'delta',
    'sample_num': 50,
    'blob_num': 1000,
    'num_distribution': 'poisson',
    'clustering': (0.05, 0.8),
    'pad': 0,
    'noise': 0
}
# Initiate params
blob_num = generation_params['blob_num']
num_distribution = generation_params['num_distribution']
clustering = generation_params['clustering']
blob_amplitude = generation_params['blob_amplitude']
amplitude_distribution = generation_params['amplitude_distribution']
seed = generation_params['seed']
blob_size = generation_params['blob_size']
sample_num = generation_params['sample_num']
image_size = generation_params['image_size']
noise = generation_params['noise']

project_path = "C:/Users/Idiot/Desktop/Research/OFYP/cosmo"
data_file_name = 'bn{}{}-cl{}-is{}-bs{}-ba{}{}-sn{}-sd{}-ns{}'.format(
    blob_num, num_distribution[0], 
    '{:.0e}_{:.0e}'.format(*clustering) if clustering is not None else '_',
    image_size, blob_size, 
    '{:.0e}'.format(blob_amplitude), amplitude_distribution[0], 
    sample_num,
    seed, int(noise)
)
data_folder = f'cosmo_data/{blob_num}_blob'
save_folder = 'misc_plots/2-point-corr'

root_path = "C:/Users/Idiot/Desktop/Research/OFYP/cosmo"


if __name__=='__main__':
    try:    
        coords = np.load(f'{project_path}/{data_folder}/{data_file_name}_coords.npy', allow_pickle=True)
    except FileNotFoundError:
        print('data not found')
        dataset = blobDataset(**generation_params)
        dataset.realize(mode='single', batch=10)
        coords = dataset.sample_centers
    
    # Only look at n samples
    coords = coords[:50]
    
    """Correct 2-point correlation"""
    def func(x, a, b):
        return a * (x**(-b))
    
    x = np.linspace(0.001, 1, 100)
    
    """2-point correlation individual samples"""
    # plotting_sample_num = 3

    # sample_coords = coords[:plotting_sample_num]

    # fig, ax = plt.subplots() 
    
    # colors = [('C0','C0'), ('C1','C1'), ('C2','C2')]
    # for coord, color in tqdm(zip(sample_coords, colors)):
    #     # corrs, edges = two_point_correlation(coord, image_size, bins=20)

    #     # plot_histogram_stack(ax, corrs, edges, color=('r',0.2), label='custom', logscale=False)
        
    #     corrs, errs, edges = calculate_two_point(coord, image_size, bins=20)
        
    #     plot_smooth_line(ax, corrs, midpoints_of_bins(edges), errs, color=color, 
    #                      label='astroML', scale='log')
        
    # if clustering is None:
    #     pass    
    # else:
    #     plt.plot(x*image_size, func(x, clustering[0], clustering[1]))
    
    # # Format
    # # plt.ylim(-1,1)
    # fig.suptitle(f"2-point correlation, samples of {blob_num} blobs")
    # plt.xlabel('pair distance')
    # plt.ylabel('2-point correlation')
    # plt.tight_layout()
    

    # # Save
    # plt.show()
    # plt.close() 
    
    """2-point correlation one sample"""
    sample_coords = coords[0]

    fig, ax = plt.subplots() 
    if clustering is None:
        pass    
    else:
        plt.plot(x, func(x, clustering[0], clustering[1]), color='black', label='input function')

    corrs, errs, edges = calculate_two_point(sample_coords, image_size, bins=20)
    plot_smooth_line(ax, corrs, midpoints_of_bins(edges/image_size), errs, 
                     color=(('r', 0.8),('r')), label='realization', scale='log')
    
    # Format
    # plt.ylim(-1,1)
    fig.suptitle(f"2-point correlation")
    plt.xlabel('ratio of pair distance to image size')
    plt.ylabel('2-point correlation')
    plt.legend()
    plt.tight_layout()
    
    # Save
    plt.savefig(f'{save_folder}/2-point')
    plt.close() 

    """2-point correlation all samples"""
    # corrs, edges = stack_two_point_correlation(coords, image_size, bins=20, progress_bar=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(5,3.5))

    # Plot
    # plot_histogram_stack(ax, corrs, edges, color=('r',0.8), label='custom', logscale=False)
    
    corrs, errs, edges = two_point_stack(coords, image_size, bins=20, progress_bar=True)
    
    if clustering is None:
        pass
    else:
        plt.plot(x, func(x, clustering[0], clustering[1]), color=('black'), label='input function')
        
    plot_smooth_line(ax, corrs, midpoints_of_bins(edges/image_size), errs, 
                     color=(('r',0.8),('r')), label='realization', scale='log')

    
    # Format
    # ax.set_ylim(-1, 1)
    fig.suptitle(f"Mean of 2-point correlation over {len(coords)} samples")
    plt.xlabel('ratio of pair distance to image size')
    plt.ylabel('2-point correlation')
    plt.tight_layout()

    plt.legend()

    # Save
    plt.savefig(f'{save_folder}/2-point_mean')
    plt.close()
    

        