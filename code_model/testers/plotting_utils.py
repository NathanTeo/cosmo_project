"""
Author: Nathan Teo

This script contains functions used for plotting during model testing
"""

import matplotlib.pyplot as plt
import numpy as np

from code_model.testers.eval_utils import *

def plot_img_grid(subfig, imgs, grid_row_num, title, wspace=.2, hspace=.2, subplot_titles=None, vmin=-0.05, vmax=None):
    """
    Plot a grid of sample images in a subfigure/figure
    """
    # Create grid of subplots for subfigure/figure
    axs = subfig.subplots(grid_row_num, grid_row_num)
    
    # Plot sample images in grid
    for i in range(grid_row_num):
        for j in range(grid_row_num):
            axs[i, j].imshow(imgs[(grid_row_num)*i+j], interpolation='none', vmin=vmin, vmax=vmax)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].axis('off')
            if subplot_titles is not None:
                axs[i, j].set_title('{:.4f}'.format(subplot_titles[(grid_row_num)*i+j]))
    
    # Format
    subfig.subplots_adjust(wspace=wspace, hspace=hspace)         
    subfig.suptitle(title, y=0.95)
    
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
                            extremum='min', k=3, title=None, vmin=-0.05, vmax=None):
    """
    Plot the image with the minimum number of blobs
    """
    axs = subfig.subplots(k)
    
    if extremum=='min':
        min_idxs = np.argpartition(blob_nums, k)[:k]
    elif extremum=='max':
        min_idxs = np.argpartition(blob_nums, -k)[-k:]

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
        
    subfig.suptitle(title, y=0.92)

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
            coords_x = coords[:, 1]
            coords_y = coords[:, 0]
            
            
            # Plot
            axs[i, j].imshow(img, interpolation='none', vmin=vmin, vmax=vmax)
            axs[i, j].scatter(coords_x, coords_y, c='r', marker='x', alpha=0.5)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].axis('off')
            
            if imgs_peak_values is not None:
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

def plot_stacked_imgs(ax, stacked_img, title, vmin=-0.05, vmax=None):
    """
    Plot stacked image
    """
    ax.imshow(stacked_img, vmin=vmin, vmax=vmax)
    ax.set_title(title)

def plot_pixel_histogram(ax, imgs, color, bins=None):
    """
    Plot individual image histograms on the same axes.
    """
    for img in imgs:
        ax.hist(img.ravel(), histtype='step', log=True, color=color, bins=bins)
    
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
        
def plot_two_point(ax, corrs, edges, errs=None, interp='cubic',
                   color=(('darkorange', 1), ('darkorange', 0.5)), linewidth=1,
                   label=None, errorbars=True):
    """Plots the interpolated 2 point correlation with errors is available"""
    # Get midpoints and interpolate
    midpoints = midpoints_of_bins(edges)
    smooth = interp1d(midpoints, corrs, kind=interp)
    
    # Plot
    x = np.linspace(midpoints[0], midpoints[-1], 100)
    ax.plot(x, smooth(x), color=color[0], linewidth=linewidth, label=label)
    if errorbars:
        ax.errorbar(midpoints, corrs, yerr=errs, fmt='.', color=color[1])


def midpoints_of_bins(edges):
    """Returns midpoints of bin edges for plotting"""
    return (edges[:-1]+edges[1:])/2 

def set_linewidth(current_iter, total_iter, minor=0.3, major=1.2):
    return minor if current_iter!=(total_iter-1) else major
    