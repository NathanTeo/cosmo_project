"""
Author: Nathan Teo

This script contains functions used for plotting during model testing
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm
from scipy.stats import multivariate_normal
import random

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

def create_circular_mask(h, w, center, radius=None):
    """
    Creates a circular mask on an image of size (h,w) at a specified coordinate
    """
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def find_local_peaks(img, min_distance=1, threshold_abs=0):
    """
    Returns coordinates of all local peaks of an image.
    Minimum distance between peaks (size of mask) and absolute minimum threshold can be specified
    """
    # Create a temporary padded image so that mask can be iterated through each point
    temp_canvas = np.pad(img, (min_distance,min_distance), mode='constant', constant_values=(0,0))
    
    # Find local peak coordinates
    img_peak_coords = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Mask image to only find maximum of local region
            mask = create_circular_mask(
                temp_canvas.shape[1], temp_canvas.shape[0],
                (j+min_distance,i+min_distance),
                radius = min_distance
                )
            masked_canvas = temp_canvas.copy()
            masked_canvas[~mask] = 0

            # Coordinates of local maxima
            canvas_peak_coord = np.unravel_index(masked_canvas.argmax(), masked_canvas.shape)
            
            # Record coordinates if the local maxima is the center of the mask and above absolute threshold
            if canvas_peak_coord==(i+min_distance, j+min_distance):
                if masked_canvas[canvas_peak_coord[0], canvas_peak_coord[1]]>=threshold_abs:
                    img_peak_coords.append([coord-min_distance for coord in canvas_peak_coord])
    
    return np.array(img_peak_coords)

def imgs_peak_finder(imgs, min_distance=3, threshold_abs=0.05, filter_sd = None, progress_bar=True):
    """
    Return coordinates of peaks for an array of images
    """
    peak_coords = []
    peak_nums = []
    peak_vals = []
    
    for img in tqdm(imgs, disable=(not progress_bar)):
        # Smooth image to remove noise
        if filter_sd is not None:    
            img = gaussian_filter(img, filter_sd, mode='nearest')
        
        # Find and record peaks
        img_peak_coords = find_local_peaks(img, min_distance=min_distance, threshold_abs=threshold_abs)
        peak_coords.append(img_peak_coords)
        peak_nums.append(len(img_peak_coords))
        peak_vals.append([img[coord[0],coord[1]] for coord in img_peak_coords])
        
    return peak_coords, peak_nums, peak_vals

def guassian_decomposition(img, blob_size, min_peak_threshold=0.08, max_iters=20):
    img_decomp = img.copy()
    peak_coords = []
    peak_vals = []
    
    for _ in range(max_iters):
        peak_coord = np.unravel_index(img_decomp.argmax(), img_decomp.shape)
        peak_val = np.max(img_decomp)
    
        x, y = np.mgrid[0:img_decomp.shape[0]:1, 0:img_decomp.shape[1]:1]
        pos = np.dstack((x, y))
        gaussian = normalize_2d(multivariate_normal(peak_coord, [[blob_size, 0], [0, blob_size]]).pdf(pos))*peak_val

        img_decomp = np.subtract(img_decomp, gaussian)        
        
        if peak_val<=min_peak_threshold:
            break
        
        mask_radius = int(np.round(blob_size/2))
        temp_canvas = np.pad(img_decomp, (mask_radius,mask_radius), mode='constant', constant_values=(0,0))
        mask = create_circular_mask(
            temp_canvas.shape[1], temp_canvas.shape[0],
            np.add((peak_coord[1], peak_coord[0]),mask_radius),
            radius = mask_radius
        )
        
        masked_canvas = temp_canvas.copy()
        masked_canvas[~mask] = 0

        mask_size = np.count_nonzero(mask)
        
        if masked_canvas.sum()/mask_size<=-0.01:
            img_decomp[peak_coord[0], peak_coord[1]] = 0
            continue
        else:
            peak_coords.append(peak_coord)
            peak_vals.append(peak_val)
        
    return peak_coords, peak_vals

def imgs_blob_finder(imgs, blob_size, min_peak_threshold, max_iters=20, filter_sd=None, progress_bar=True):
    """
    Return coordinates of blobs and the corresponding peak values for an array of images
    """
    blob_coords = []
    blob_nums = []
    peak_vals = []
    
    for img in tqdm(imgs, disable=(not progress_bar)):
        # Smooth image to remove noise
        if filter_sd is not None:    
            img = gaussian_filter(img, filter_sd, mode='nearest')
        
        # Find and record blobs
        img_blob_coords, img_peak_vals = guassian_decomposition(img, blob_size, min_peak_threshold, max_iters)
        blob_coords.append(img_blob_coords)
        blob_nums.append(len(img_blob_coords))
        peak_vals.append(img_peak_vals)

    return blob_coords, blob_nums, peak_vals

def count_blobs(imgs_peak_vals, generation_blob_number):
    """
    Find the counts for each peak and the total number of peaks for a series of images
    """
    imgs_peak_counts = []
    imgs_total_blob_counts = []
    
    for peak_vals in imgs_peak_vals:
        peak_count_vals = [int(np.round(vals*generation_blob_number)) for vals in peak_vals]
        imgs_peak_counts.append(peak_count_vals)
        imgs_total_blob_counts.append(np.sum(peak_count_vals))
    
    return imgs_peak_counts, imgs_total_blob_counts
    
def plot_min_num_peaks(ax, imgs, peak_nums, title=None):
    """
    Plot the image with the minimum number of peaks
    """
    min_num_peaks = np.min(peak_nums)
    min_peak_idx = np.argmin(peak_nums)

    ax.imshow(imgs[min_peak_idx])
    ax.set_title(f"{title}\ncounts: {min_num_peaks}")
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    return min_num_peaks

def plot_extremum_num_blobs(subfig, imgs, blob_nums, extremum='min', k=3, title=None):
    """
    Plot the image with the minimum number of blobs
    """
    axs = subfig.subplots(k)
    
    if extremum=='min':
        min_idxs = np.argpartition(blob_nums, k)[:k]
    elif extremum=='max':
        min_idxs = np.argpartition(blob_nums, -k)[-k:]

    for ax, idx in zip(axs, min_idxs):
        ax.imshow(imgs[idx])
        ax.set_title(f"counts: {blob_nums[idx]}")
    
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    
    subfig.suptitle(title, y=0.92)

def plot_peak_grid(subfig, imgs, imgs_coords, imgs_peak_values, grid_row_num, title, wspace=.2, hspace=.2, subplot_titles=None):
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
            peak_values = imgs_peak_values[(grid_row_num)*i+j]
            
            # Plot
            axs[i, j].imshow(img, interpolation='none')
            axs[i, j].scatter(coords_x, coords_y, c='r', marker='x', alpha=0.5)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].axis('off')
            
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

def plot_pixel_histogram(ax, imgs, color, bins=None):
    """
    Plot individual image histograms on the same axes.
    """
    for img in imgs:
        ax.hist(img.ravel(), histtype='step', log=True, color=color, bins=bins)
    
    ax.set_ylabel('counts')
    ax.set_xlabel('pixel value')
        
def stack_histograms(imgs, bins=np.arange(-0.1,1,0.05), mean=True, progress_bar=True):
    """
    Find the mean/total of a series of histograms.
    Returns edges for plotting purposes. 
    """
    if isinstance(bins, int):
        stack = np.zeros(bins)
    else:
        stack = np.zeros(len(bins)-1)
        
    for img in tqdm(imgs, disable=(not progress_bar)):
        # Get histogram for single image    
        n, edges, _ = plt.hist(img.ravel(), bins=bins)
        # Add histogram to stack
        stack = np.add(stack, n)

    if mean:
        # Mean histogram
        return stack/len(imgs), edges
    else:
        # Total histogram
        return stack, edges

def plot_histogram_stack(ax, hist, edges, color, label=None, fill_color='r', fill=False, xlabel='counts', ylabel='pixel value'):
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
    ax.plot(x, y, color=color, label=label)
    
    # Labels
    ax.set_ylabel(xlabel)
    ax.set_xlabel(ylabel)
    
    # Fill colour
    if fill:
        ax.fill_between(x, 0, y, color=fill_color)

def normalize_2d(matrix):
    """
    Normalized entire matrix to minimum and maximum of the matrix
    """
    return (matrix-np.min(matrix))/(np.max(matrix)-np.min(matrix))

def euclidean_dist(coord_1, coord_2):
    """
    Calculates euclidean distance between 2 points
    """
    return ((coord_1[0]-coord_2[0])**2+(coord_1[1]-coord_2[1])**2)**0.5

def generate_random_coords(image_size, n):
    """
    Generate n number of random 2D coordinates within the range of the image size
    """
    coords = []
    for _ in range(n):
        coords.append([random.randint(0, image_size-1), random.randint(0, image_size-1)])
    return np.array(coords)

def find_pair_distances(sample_1, sample_2):
    """
    Find distances between all pairs of points given two samples.
    If both samples are the same, the distances to all pairs of points within one sample will be calculated.
    """
    if (len(sample_1)==len(sample_2)) and ((sample_1==sample_2).all()):
        sample = sample_1
        sample_temp = np.array(sample).copy()
        distances = []
        
        # Calculate distances
        for coord in sample:
            for i in range(len(sample_temp)-1):
                distances.append(euclidean_dist(coord, sample_temp[i+1]))
            sample_temp = sample_temp[1:]
    else:
        distances = []
        for coord_1 in sample_1:
            for coord_2 in sample_2:
                distances.append(euclidean_dist(coord_1, coord_2))
                
    return distances

def find_pair_hist(sample_1, sample_2, bins):
    """
    Returns the histogram of distances given two samples of points.
    """
    distances = find_pair_distances(sample_1, sample_2)
    n, edges, _ = plt.hist(distances, bins=bins)
    
    return n/len(distances), edges

def two_point_correlation(sample, image_size, bins=10, rel_random_n=5):
    """
    Calculates the two point correlation using the Landy Szalay estimator given an image sample
    """
    random_sample = generate_random_coords(image_size, len(sample)*rel_random_n)
    
    dd_norm, edges = find_pair_hist(sample, sample, bins)
    rr_norm, _ = find_pair_hist(random_sample, random_sample, bins)
    dr_norm, _ = find_pair_hist(sample, random_sample, bins)
    
    return (dd_norm-2*dr_norm+rr_norm)/rr_norm, edges

def stack_two_point_correlation(point_coords, image_size, bins=10, rel_random_n=5, progress_bar=False):
    """
    Calculates the two point correlation using the Landy Szalay estimator given a series of image samples
    """
    bins = np.linspace(0, image_size,bins)
    dd_dists = []
    rr_dists = []
    dr_dists = []
    
    for sample in tqdm(point_coords, disable=not progress_bar):
        sample = np.array(sample)
        random_sample = generate_random_coords(image_size, len(sample)*rel_random_n)
        
        # Calculate distances
        dd_dists.extend(find_pair_distances(sample, sample))
        rr_dists.extend(find_pair_distances(random_sample, random_sample))
        dr_dists.extend(find_pair_distances(sample, random_sample))
    
    dd, edges, _ = plt.hist(dd_dists, bins=bins)
    rr, edges, _ = plt.hist(rr_dists, bins=bins)
    dr, edges, _ = plt.hist(dr_dists, bins=bins)
    
    dd_norm = dd/len(dd_dists)
    rr_norm = rr/len(rr_dists)
    dr_norm = dr/len(dr_dists)

    corr = (dd_norm-2*dr_norm+rr_norm)/rr_norm
    
    return corr, edges

def find_total_fluxes(samples, progress_bar=False):
    """Returns an array of total image flux for an array of image samples"""
    fluxes = []
    for sample in tqdm(samples, disable=not progress_bar):
        fluxes.append(sample.sum())
    
    return fluxes