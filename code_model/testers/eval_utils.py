"""
Author: Nathan Teo

This script contains functions useful for model evaluation
"""

import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm
from scipy.stats import multivariate_normal
import random



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

def marginal_sums(img):
    """
    Calculate marginal sums along x and y for sample images 
    """
    y_sum = [y for y in img.sum(axis=1)]
    x_sum = [x for x in img.sum(axis=0)]
    return y_sum, x_sum

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