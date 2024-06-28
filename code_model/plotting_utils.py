"""
Author: Nathan Teo

This script contains functions used for plotting during model testing
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter

def plot_img_grid(subfig, imgs, grid_row_num, title, wspace=.2, hspace=.2, subplot_titles=None):
    """
    Plot a grid of sample images in a subfigure/figure
    """
    # Create grid of subplots for subfigure/figure
    axs = subfig.subplots(grid_row_num, grid_row_num)
    
    # Plot sample images in grid
    for i in range(grid_row_num):
        for j in range(grid_row_num):
            axs[i, j].imshow(imgs[(grid_row_num)*i+j], interpolation='none')
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].axis('off')
            if subplot_titles is not None:
                axs[i, j].set_title('{:.4f}'.format(subplot_titles[(grid_row_num)*i+j]))
    
    # Format
    subfig.subplots_adjust(wspace=wspace, hspace=hspace)         
    subfig.suptitle(title, y=0.95)

def find_peaks(imgs, min_distance=3, threshold_abs=0.05, filter_sd = 2):
    """
    Return coordinates of peaks for an array of images
    """
    peak_coords = []
    peak_nums = []
    
    for i in range(len(imgs)):
        # Smooth image to remove noise
        img = gaussian_filter(imgs[i], filter_sd, mode='nearest')
        
        # Find and record peaks
        img_peak_coords = peak_local_max(img, min_distance=min_distance, 
                                         threshold_abs=threshold_abs, exclude_border=False)
        peak_coords.append(img_peak_coords)
        peak_nums.append(len(img_peak_coords))
        
    return peak_coords, peak_nums
    

def plot_peak_grid(subfig, imgs, imgs_coords, grid_row_num, title, wspace=.2, hspace=.2, subplot_titles=None):
    """
    Plot a grid of images with detected peaks in a subfigure/figure
    """
    # Create grid of subplots for subfigure/figure
    axs = subfig.subplots(grid_row_num, grid_row_num)
    
    # Plot sample images in grid
    for i in range(grid_row_num):
        for j in range(grid_row_num):
            # Img and peak coords 
            img = imgs[(grid_row_num)*i+j]
            coords = np.array(imgs_coords[(grid_row_num)*i+j])
            coords_x = coords[:, 1]
            coords_y = coords[:, 0]
            peak_values = [img[coords_y[i],coords_x[i]] for i in range(len(coords_x))]
            
            # Plot
            axs[i, j].imshow(img, interpolation='none')
            axs[i, j].scatter(coords_x, coords_y, c='r', marker='x', alpha=0.5)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].axis('off')
            for k in range(len(peak_values)):    
                axs[i, j].annotate('{:.2f}'.format(peak_values[k]), (coords_x[k], coords_y[k]))
            
            if subplot_titles is not None:
                axs[i, j].set_title('{}'.format(subplot_titles[(grid_row_num)*i+j]))
    
    # Format
    subfig.subplots_adjust(wspace=wspace, hspace=hspace)         
    subfig.suptitle(title, y=1)

def marginal_sums(img):
    """
    Calculate marginal sums along x and y for sample images 
    """
    y_sum = [y for y in img.sum(axis=1)]
    x_sum = [x for x in img.sum(axis=0)]
    return y_sum, x_sum
        
def plot_marginal_sums(imgs, subfig, grid_row_num, title):
    """
    Plot marginal sums along x and y for sample images 
    """
    # Create subplots for subfigure/figure
    axs = subfig.subplots(grid_row_num*grid_row_num)
    
    # Plot marginal sums in subplots
    for i, ax in enumerate(axs):
        y_sum, x_sum = marginal_sums(imgs[i])
        ax.plot(y_sum, label='y')
        ax.plot(x_sum, label='x')
        
    # Format
    subfig.suptitle(title, y=0.95)

def stack_imgs(imgs):
    """
    Stack images into single image
    """
    for i, sample in enumerate(imgs):
        if i == 0:
            # Add first sample image to stacked image
            stacked_img = sample
        else:
            # Add subsquent samples to stacked image
            stacked_img = np.add(stacked_img, sample)
    return stacked_img

def plot_stacked_imgs(ax, stacked_img, title):
    """
    Plot stacked image
    """
    ax.imshow(stacked_img)
    ax.set_title(title)