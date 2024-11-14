"""
Author: Nathan Teo

This script contains functions used for model evaluation plotting in the modules.py file
"""

import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd

from code_model.testers.eval_utils import *

def plot_img_grid(subfig, imgs, grid_row_num, 
                  title, title_y=0.95, 
                  wspace=.2, hspace=.2, subplot_titles=None, 
                  vmin=-0.05, vmax=None, cmap='viridis'):
    """
    Plot a grid of sample images in a subfigure/figure
    """
    # Create grid of subplots for subfigure/figure
    axs = subfig.subplots(grid_row_num, grid_row_num)
    
    # Plot sample images in grid
    subplots = [[None for x in range(grid_row_num)] for y in range(grid_row_num)] 
    for i, j in itertools.product(range(grid_row_num), range(grid_row_num)):
        subplots[i][j] = axs[i, j].imshow(imgs[(grid_row_num)*i+j], interpolation='none', 
                                          vmin=vmin, vmax=vmax, cmap=cmap)
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        axs[i, j].axis('off')
        if subplot_titles is not None:
            axs[i, j].set_title('{:.4f}'.format(subplot_titles[(grid_row_num)*i+j]))

    # Format
    subfig.subplots_adjust(wspace=wspace, hspace=hspace)         
    subfig.suptitle(title, y=title_y)
    
    return subplots
    
def plot_min_num_peaks(ax, imgs, peak_nums, title=None, vmin=-0.05, vmax=None):
    """
    Plot the image with the minimum number of peaks
    """
    min_num_peaks = np.min(peak_nums)
    min_peak_idx = np.argmin(peak_nums)

    ax.imshow(imgs[min_peak_idx], vmin=vmin, vmax=vmax)
    ax.set_title(f"{title}\ncounts: {min_num_peaks}")
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    return min_num_peaks

def plot_extremum_num_blobs(subfig, imgs, imgs_coords, blob_nums, imgs_peak_counts=None,
                            extremum='min', k=3, title=None, title_y=0.92,
                            vmin=-0.05, vmax=None):
    """
    Plot the image with the minimum number of blobs
    """
    axs = subfig.subplots(k)
    
    if extremum=='min':
        min_idxs = np.argpartition(blob_nums, k)[:k]
    elif extremum=='max':
        min_idxs = np.argpartition(blob_nums, -k)[-k:]
    
    # Make k=1 case iterable for loop
    if k==1:
        axs = [axs]

    for ax, idx in zip(axs, min_idxs):
        ax.imshow(imgs[idx], vmin=vmin, vmax=vmax)
        ax.set_title(f"counts: {blob_nums[idx]}")
        
        coords = np.array(imgs_coords[idx])
        
        if coords.size: 
            coords_x = coords[:, 1]
            coords_y = coords[:, 0]
           
            ax.scatter(coords_x, coords_y, c='r', marker='x', alpha=0.5)
            
            if imgs_peak_counts is not None:
                peak_counts = imgs_peak_counts[idx]
                for k in range(len(peak_counts)):    
                    ax.annotate('{}'.format(peak_counts[k]), (coords_x[k], coords_y[k]))
        else:
            pass # If there are no peaks detected 
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
    subfig.suptitle(title, y=title_y)

def plot_peak_grid(subfig, imgs, imgs_coords,
                   grid_row_num, title, wspace=.2, hspace=.2, imgs_peak_values=None, subplot_titles=None, vmin=-0.05, vmax=None):
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
            
            # Plot
            axs[i, j].imshow(img, interpolation='none', vmin=vmin, vmax=vmax)
            if len(coords)>0: # skip if zero blob case    
                coords_x = coords[:, 1]
                coords_y = coords[:, 0]
                axs[i, j].scatter(coords_x, coords_y, c='r', marker='x', alpha=0.5)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].axis('off')
            
            if imgs_peak_values is not None and len(coords)>0:
                peak_values = imgs_peak_values[(grid_row_num)*i+j]
                for k in range(len(peak_values)):    
                    if isinstance(peak_values[0], int): # I don't like how this works. Find a more general way to do this?
                        axs[i, j].annotate('{}'.format(peak_values[k]), (coords_x[k], coords_y[k]))
                    else:
                        axs[i, j].annotate('{:.2f}'.format(peak_values[k]), (coords_x[k], coords_y[k]))
            
            if subplot_titles is not None:
                axs[i, j].set_title('{}'.format(subplot_titles[(grid_row_num)*i+j]))
    
    # Format
    subfig.subplots_adjust(wspace=wspace, hspace=hspace)         
    subfig.suptitle(title, y=1)
        
def plot_marginal_sums(marginal_sums, subfig, grid_row_num, title):
    """
    Plot marginal sums along x and y for sample images 
    """
    # Create subplots for subfigure/figure
    axs = subfig.subplots(grid_row_num*grid_row_num)
    
    # Plot marginal sums in subplots
    for i, ax in enumerate(axs):
        y_sum, x_sum = marginal_sums[i]
        ax.plot(y_sum, label='y')
        ax.plot(x_sum, label='x')
        
    # Format
    subfig.suptitle(title, y=0.95)

def plot_stacked_imgs(ax, stacked_img, title=None, vmin=-0.05, vmax=None):
    """
    Plot stacked image
    """
    ax.imshow(stacked_img, vmin=vmin, vmax=vmax)
    if title is not None:    
        ax.set_title(title)

def plot_pixel_histogram(ax, imgs, color, bins=None, logscale=True):
    """
    Plot individual image histograms on the same axes.
    """
    for img in imgs:
        ax.hist(img.ravel(), histtype='step', log=logscale, color=color, bins=bins)
    
    ax.set_ylabel('image count')
    ax.set_xlabel('pixel value')

def plot_histogram_stack(ax, hist, edges, 
                         color, linewidth=1, fill_color=None,
                         label=None, logscale=True):
    """
    Plots histogram from histogram data, n (value for each bar) and edges (x values of each bar).
    """
    # Create points from histogram data
    x, y = [edges[0]], [0]
    for i in range(len(hist)):
        x.extend((edges[i], edges[i+1]))
        y.extend((hist[i], hist[i]))
    x.append(edges[-1])
    y.append(0)
    
    # Plot
    ax.plot(x, y, color=color, label=label, linewidth=linewidth)
    
    if logscale:
        ax.set_yscale('log')
    
    # Fill colour
    if fill_color is not None:
        ax.fill_between(x, 0, y, color=fill_color)
        
def plot_smooth_line(ax, y, x, errs=None, interp='cubic',
                   color=(('darkorange', 1), ('darkorange', 0.5)), linewidth=1, capsize=2, elinewidth=1,
                   label=None, errorbars=True, scale='semilog_x'):
    """Plots the interpolated 2 point correlation with errors is available"""
    # Interpolate
    smooth = interp1d(x, y, kind=interp)
    
    # Plot
    lin = np.linspace(x[0], x[-1], 100)
    ax.plot(lin, smooth(lin), color=color[0], linewidth=linewidth, label=label)
    
    if errorbars:
        ax.errorbar(x, y, yerr=errs, fmt='.', capsize=capsize, elinewidth=elinewidth, color=color[1])
        
    if scale=='log' or scale=='semilog_y':
      ax.set_yscale('log')
    if scale=='log' or scale=='semilog_x':
      ax.set_xscale('log')
        
def midpoints_of_bins(edges):
    """Returns midpoints of bin edges for plotting"""
    return (edges[:-1]+edges[1:])/2 

def set_linewidth(current_iter, total_iter, minor=0.3, major=1.2):
    return minor if current_iter!=int(total_iter-1) else major

def millify(n, rounding=1):
    millnames = ['','k','M','B','T']
    
    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(np.floor(0 if n == 0 else np.log10(abs(n))/3))))

    return '{value:.{rounding}f}{millname}'.format(value=n / 10**(3 * millidx), rounding=rounding, millname=millnames[millidx])    

def find_good_bins(arrs, spacing=(1.5, 1.5), 
                   method='arange', step=1, num_bins=20,
                   ignore_outliers=False, percentile_range=(1,99)):
    """Returns reasonable bins for histogram"""
    arr = np.concatenate(arrs)
    
    if ignore_outliers:
        bin_min = np.floor(np.percentile(arr, percentile_range[0]))
        bin_max = np.ceil(np.percentile(arr, percentile_range[1]))
    else:        
        bin_min = np.min(arr)
        bin_max = np.max(arr)
    
    if method=='arange': 
        return np.arange(bin_min-spacing[0], bin_max+spacing[1], step)
    elif method=='linspace':
        return np.linspace(bin_min-spacing[0], bin_max+spacing[1], num_bins)

def blank_plot(ax):
    ax.set_axis_off()

def save_log_dict(file_path, dict):
    """Saves the log dictionary as a csv"""
    df = pd.DataFrame(dict)
    df.to_csv(file_path)

def capword(word):
    """Capitalize first letter of word"""
    lst = list(word)
    lst[0] = lst[0].upper()
    return "".join(lst)