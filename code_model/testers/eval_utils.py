"""
Author: Nathan Teo

This script contains functions and classes for model evaluation in the modules.py file
"""

import numpy as np
import torch
from numpy.linalg import norm
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from scipy import spatial
from scipy.stats import entropy
from tqdm.auto import tqdm
from scipy.stats import multivariate_normal
from scipy.signal.windows import general_gaussian, tukey
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from astroML.correlation import two_point_angular, bootstrap_two_point_angular
import time
import concurrent.futures
from multiprocessing import Manager

"""Ungrouped"""
inv_transform_dict = {
    torch.log10: lambda x: 10**x
}

def update_progress_bar(queue, total, pbar):
    """This function updates the progress bar from the main thread"""
    while pbar.n < total:
        queue.get()  # Block until there is progress to update
        pbar.update(1)  # Update the progress bar by 1 unit

def init_param(config, param, default=None):
    """Initialize parameter and return default value if key is not found in config dictionary"""
    try:
        return config[param]
    except KeyError:
        return default

def marginal_sums(img):
    """Calculate marginal sums along x and y for sample images """
    y_sum = [y for y in img.sum(axis=1)]
    x_sum = [x for x in img.sum(axis=0)]
    return y_sum, x_sum

def stack_imgs(imgs):
    """Stack images into single image"""
    for i, sample in enumerate(imgs):
        if i == 0:
            # Add first sample image to stacked image
            stacked_img = sample
        else:
            # Add subsquent samples to stacked image
            stacked_img = np.add(stacked_img, sample)
    return stacked_img

def stack_histograms(imgs, bins=np.arange(-0.1,1,0.05), mean=True, progress_bar=True):
    """Find the mean/total of a series of histograms.
    Returns edges for plotting purposes."""
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
    """Normalize 2d matrix to [0,1]"""
    return (matrix-np.min(matrix))/(np.max(matrix)-np.min(matrix))

def find_total_fluxes(samples):
    """Returns an array of total image flux for an array of image samples"""
    samples = samples.reshape(samples.shape[0], -1)
    return np.sum(samples, axis=1)
    
def make_gaussian(center, var, image_size):
    """Make a square gaussian kernel"""
    x = np.arange(0, image_size, 1, float)
    y = x[:,np.newaxis]

    x0 = center[1]
    y0 = center[0]

    return np.exp(-0.5 * ((x-x0)**2 + (y-y0)**2) / var)

def create_blobs(centers, image_size, blob_size, blob_amplitude):
    """Create a sample of gaussian blobs"""
    if len(centers)==0:
        return np.zeros((image_size, image_size))
    else:
        return np.array([normalize_2d(make_gaussian(coord, blob_size, image_size))*blob_amplitude
                            for coord in centers]).sum(axis=0)
    
def get_residuals(samples, blob_coords, image_size, blob_size, blob_amplitude):
    """Calculate residuals"""
    # Realize gaussian sample from generated blob center coordinates
    coords_to_gaussian_samples = np.array([create_blobs(centers, image_size, blob_size, blob_amplitude)
                                           for centers in tqdm(blob_coords)])
    
    # Get residuals
    return coords_to_gaussian_samples - samples

def JSD(P, Q):
    """Calculates the Jensen-Shannon divergence"""
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

"""Power spectrum"""
def cosine_window(N):
    """Makes a cosine window for apodizing to avoid edges effects in the 2d FFT"""
    # make a 2d coordinate system
    N=int(N) 
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.)/N *np.pi ## eg runs from -pi/2 to pi/2
    X = np.outer(ones,inds)
    Y = np.transpose(X)
  
    # make a window map
    window_map = np.cos(X) * np.cos(Y)
   
    # return the window map
    return(window_map)

def super_gaussian_window(N):
    """Makes a super gaussian window for apodizing to avoid edges effects in the 2d FFT"""
    return np.outer(general_gaussian(N,6,N*0.45-1),general_gaussian(N,6,N*0.45-1))

def tukey_window(N, alpha=0.5):
    """Makes a tuckey window for apodizing to avoid edges effects in the 2d FFT"""
    return np.outer(tukey(N, alpha), tukey(N, alpha))

def apodize(sample, method='tukey'):
    """Apodize a sample, supports cosine, supergaussian and tukey windows"""
    if method=='cosine':
        return cosine_window(sample.shape[0]) * sample

    elif method=='supergaussian':
        return super_gaussian_window(sample.shape[0]) * sample

    elif method=='tukey':
        return tukey_window(sample.shape[0]) * sample

    else:
        return sample

def fourier_transform_samples(samples, progress_bar=False):
    """Performs a 2d FFT on the image"""
    return [np.abs(np.fft.fftshift(np.fft.fft2(apodize(sample)))) 
            for sample in tqdm(samples, disable=not progress_bar)]

def ell_coordinates(image_size_pixel, pixel_size_deg):
    """Make a 2d ell coordinate system for a FFT sample""" 
    ones = np.ones(image_size_pixel)
    inds  = (np.arange(image_size_pixel)+.5 - image_size_pixel/2.) /(image_size_pixel-1.)
    kX = np.outer(ones,inds) / (pixel_size_deg * np.pi/180.)
    kY = np.transpose(kX)
    K = np.sqrt(kX**2. + kY**2.)
    ell_scale_factor = 2. * np.pi
    return K * ell_scale_factor
    
def power_spectrum(Map1, Map2, delta_ell, ell_max, ell2d=None, image_size_angular=1, 
                   taper=True, detrend='constant'):
    """Calcualtes the power spectrum of a 2d map by FFTing, squaring, and azimuthally averaging"""
    image_size_pixel = Map1.shape[0]
    pixel_size_angular = image_size_angular / image_size_pixel
    
    # Option to find ell2d outside this function is added to avoid repeating this initialization 
    # for multiple samples
    if ell2d is None: # Find ell coordinates if not provided 
        # Make a 2d ell coordinate system 
        ell2d = ell_coordinates(image_size_pixel, pixel_size_angular)
    
    # Make an array to hold the power spectrum results
    N_bins = int(ell_max/delta_ell)
    ell_array = np.arange(N_bins)
    CL_array = np.zeros(N_bins)
    
    # Get the 2d fourier transform of the map
    if detrend=='constant':
        Map1 = Map1 - np.mean(Map1)
        Map2 = Map2 - np.mean(Map2)
    if taper:
        Map1 = apodize(Map1)
        Map2 = apodize(Map2)
    FMap1 = np.fft.ifft2(np.fft.fftshift(Map1))
    FMap2 = np.fft.ifft2(np.fft.fftshift(Map2))
    PSMap = np.fft.fftshift(np.real(np.conj(FMap1) * FMap2))
    
    # Fill out the spectra
    for i in range(N_bins):
        ell_array[i] = (i + 0.5) * delta_ell
        idxs_in_bin = ((ell2d >= (i* delta_ell)) * (ell2d < ((i+1)* delta_ell))).nonzero()
        CL_array[i] = np.mean(PSMap[idxs_in_bin])

    # Return the power spectrum and ell bins
    return ell_array, CL_array * np.sqrt(pixel_size_angular * np.pi/180.) * 2.

def power_spectrum_stack(samples, 
                         delta_ell=500, ell_max=15000, ell2d=None, image_size_angular=1,
                         errorbar='percentile', percentile_range=(25,75),
                         progress_bar=False):
    """Calculate the mean power spectrum and variance given a set of samples"""
    # Find power spectrum of samples
    cls = []   
    for sample in tqdm(samples, disable=not progress_bar):
        bins, cl = power_spectrum(sample, sample, delta_ell, ell_max, ell2d, image_size_angular, taper=True)
        cls.append(cl)
    
    # Find mean and std dev
    mean = np.mean(cls, axis=0)
    if errorbar=='std':
        errs = np.std(cls, axis=0, ddof=1) # ddof=1 for sample estimate of popln std dev
    if errorbar=='percentile':
        lower = np.percentile(cls, percentile_range[0], axis=0)
        upper = np.percentile(cls, percentile_range[1], axis=0)
        errs = [lower, upper]
    
    return mean, errs, bins

"""2-point correlation"""
def calculate_two_point(coords, image_size, angular_size=1, bins=10, bootstrap=True, logscale=True):
    """Calculate the two point correaltion with astroML"""
    # Get edges of bins
    if logscale:
        edges = 10**np.linspace(np.log10(0.001), np.log10(angular_size), bins)
    else:    
        edges = np.linspace(0, angular_size, bins)

    coords_scaled = np.array(coords) / image_size * angular_size

    # Two point correlation
    if bootstrap:
        corrs, errs, _ = bootstrap_two_point_angular(*zip(*coords_scaled), edges)
        return corrs, errs, edges/angular_size*image_size
    else:
        corrs = two_point_angular(*zip(*coords_scaled), edges)
        return corrs, None, edges/angular_size*image_size
    
def two_point_stack(samples, image_size, bins=10, bootstrap=True, progress_bar=False, logscale=True):
    """Calculate the mean two point correlation for a set of samples with astroML"""
    # Initialize arrays
    all_corrs = []
    all_errs = []

    # Two point correlation
    for coords in tqdm(samples, disable=not progress_bar):
        if len(coords)==0:
            continue
        corrs, errs, edges = calculate_two_point(coords, image_size, bins=bins, bootstrap=bootstrap, logscale=logscale)
        all_corrs.append(corrs)
        all_errs.append(errs)
        
    # Calculate mean, ignoring nan values in corr
    corrs = np.nanmean(all_corrs, axis=0)
    errs = np.nanstd(all_corrs, axis=0, ddof=1) # ddof=1 for sample estimate of popln std dev
    ''' Don't use the bootstrap errors
    nonnan_counts = np.count_nonzero(~np.isnan(all_corrs), axis=0)
    all_errs = np.where(np.isnan(all_corrs), np.nan, all_errs)
    errs = np.sqrt(np.nansum(((1/nonnan_counts)*np.array(all_errs))**2, axis=0)) if bootstrap else None
    '''
    return corrs, errs, edges



"""Counting"""
def create_circular_mask(h, w, center, radius=None):
    """Creates a circular mask on an image of size (h,w) at a specified coordinate"""
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def gaussian_decomposition(img, blob_size, min_peak_threshold=0.08, max_iters=50, method='subtract', check_shape=False):
    """Gaussian decomposition on a single image for blob counting and blob coordinates"""
    # Initiate variables
    img_decomp = img.copy()
    peak_coords = []
    peak_vals = []
    
    # Gaussian decomposition, find maxima -> subtract gaussian from maxima -> repeat
    for _ in range(max_iters):
        # Find maximum
        peak_coord = np.unravel_index(img_decomp.argmax(), img_decomp.shape)
        peak_val = np.max(img_decomp)

        if method=='subtract':
            # Create gaussian with amplitude of maximum
            x, y = np.mgrid[0:img_decomp.shape[0]:1, 0:img_decomp.shape[1]:1]
            pos = np.dstack((x, y))
            gaussian = normalize_2d(multivariate_normal(peak_coord, [[blob_size, 0], [0, blob_size]]).pdf(pos))*peak_val

            # Subtract created gaussian from image
            img_decomp = np.subtract(img_decomp, gaussian)
        elif method=='zero':
            mask_inner = create_circular_mask(img.shape[0], img.shape[0], peak_coord[::-1], blob_size+1)
            mask_outer = create_circular_mask(img.shape[0], img.shape[0], peak_coord[::-1], blob_size+2)
            # mask_outer2 = create_circular_mask(img.shape[0], img.shape[0], peak_coord[::-1], blob_size+4)
            
            # img_decomp = np.where(mask_outer2, img_decomp*0.8, img_decomp)
            img_decomp = np.where(mask_outer, img_decomp-peak_val*0.15, img_decomp)
            img_decomp[mask_inner] = 0
        
        # Stop once threshold maximum is below threshold
        if peak_val<=min_peak_threshold:
            break
        
        if check_shape:
            # Find if residual from subtraction is relatively flat, ignore point if residual is very negative
            mask_radius = int(np.round(blob_size/2))
            temp_canvas = np.pad(img_decomp, (mask_radius,mask_radius), mode='constant', constant_values=(0,0))
            mask = create_circular_mask(
                temp_canvas.shape[1], temp_canvas.shape[0],
                np.add(peak_coord[::-1],mask_radius),
                radius = mask_radius
            )
            
            masked_canvas = temp_canvas.copy()
            masked_canvas[~mask] = 0

            mask_size = np.count_nonzero(mask)
            
            if masked_canvas.sum()/mask_size<=-0.01:
                img_decomp[peak_coord[0], peak_coord[1]] = 0
                continue
        
        peak_coords.append(peak_coord)
        peak_vals.append(peak_val)
        
    return peak_coords, peak_vals

def samples_blob_counter_fast(imgs, blob_size, min_peak_threshold, max_iters=50, filter_sd=None, method='subtract', progress_bar=True):
    """Return coordinates of blobs and the corresponding peak values for an array of images"""
    blob_coords = []
    blob_nums = []
    peak_vals = []
    
    for img in tqdm(imgs, disable=(not progress_bar)):
        # Smooth image to remove noise
        if filter_sd is not None:    
            img = gaussian_filter(img, filter_sd, mode='nearest')
        
        # Find and record blobs
        img_blob_coords, img_peak_vals = gaussian_decomposition(img, blob_size, min_peak_threshold, 
                                                                max_iters=max_iters, method=method)
        blob_coords.append(img_blob_coords)
        blob_nums.append(len(img_blob_coords))
        peak_vals.append(img_peak_vals)

    return blob_coords, blob_nums, peak_vals

def count_blobs_from_peaks(imgs_peak_vals, generation_blob_number):
    """Find the counts for each peak and the total number of peaks for a series of images"""
    imgs_peak_counts = []
    imgs_total_blob_counts = []
    
    for peak_vals in imgs_peak_vals:
        peak_count_vals = [int(np.round(vals*generation_blob_number)) for vals in peak_vals]
        imgs_peak_counts.append(peak_count_vals)
        imgs_total_blob_counts.append(np.sum(peak_count_vals))
    
    return imgs_peak_counts, imgs_total_blob_counts
    
def circle_points(r, n):
    """
    Takes in an array of radii and number of points at each radius
    Returns coordinates of equally spaced points on a circle at the specified radii 
    """
    circles = []
    for r, n in zip(r, n):
        rand_dir = np.random.uniform(low=0, high=2*np.pi)
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        x = r * np.cos(t + rand_dir)
        y = r * np.sin(t + rand_dir)
        circles.extend(np.c_[y, x])
    return np.array(circles)
 
def mse(A, B):
    """Mean squared error of two arrays"""
    return (np.square(A - B)).mean(axis=None)

def mse_from_residuals(residuals):
    """Find the mean squared errors from residuals"""
    return np.mean(np.reshape(
        np.square(residuals), (residuals.shape[0],residuals.shape[1]**2)), axis=1)

def n_slices(n, list_):
    for i in range(len(list_) + 1 - n):
        yield list_[i:i+n]

def argSublist(list_, sub_list):
    indexes=[]
    for i, slice_ in enumerate(n_slices(len(sub_list), list_)):
        if (slice_==sub_list).all():
            indexes.append(i+1)
    return indexes

def find_local_min(y, x, nonnegative=True):
    """Find first local minima found in a list"""
    # Sort by x without duplicates, taking minimum of y for duplicates
    sorted_x, idxs = np.unique(x, return_inverse=True)
    sorted_y = np.full(len(sorted_x), None)
    for i, idx in enumerate(idxs):
        if sorted_y[idx] is None:
            sorted_y[idx] = y[i]
        else:
            sorted_y[idx] = min(sorted_y[idx], y[i])

    # If nonnegative, ensure 0 can be a local minima
    if nonnegative:
        sorted_x = np.insert(sorted_x, 0, -1)
        sorted_y= np.insert(sorted_y, 0, 1e10)

    # Find local minima with differences
    diff = sorted_x[1:] - sorted_x[:-1]
    indexes = argSublist(diff, np.array([1,1]))
    if len(indexes)!=0:
        for index in indexes:
            if sorted_y[index]<sorted_y[index-1] and sorted_y[index]<sorted_y[index+1]:
                return sorted_x[index]
    
    return None

def find_dist_to_next_consec(x, val, direction=1):
    """Find nearest consecutive number not in a list, 
    starting from a specific value and going in a specified direction
    eg. x=[5,7,6,0] val=5, direction=1 => 8"""
    if val not in x:
      return None

    # Sort x
    if direction==1: # Ascending
        sorted_x = np.unique(x)
    elif direction==-1: # Descending
        sorted_x = -np.unique(-np.array(x))

    # Truncate, values in the opposite direction are irrelavant
    sorted_x = sorted_x[int(np.where(sorted_x==val)[0][0]):]

    # Find consecutive values by finding differences of 1/-1 (depending on direction)
    diff = sorted_x[1:] - sorted_x[:-1]
    consec = np.append(np.where(diff==direction, 1, 0), [0])

    # Find first zero when consecutive chain stops
    first_zero_idx = int(np.where(consec==0)[0][0])

    # Distance to next consecutive is the sum to the first zero + 1
    return consec[:first_zero_idx].sum() + 1
    

class blobFitter():
    """Fits for the number of blobs for each sample given a set of samples
    Blobs on samples must have the same amplitude"""
    def __init__(self, blob_size, blob_amplitude, jit=0.5, error_scaling=1e7):
        """Initialize params"""
        self.blob_size = blob_size
        self.blob_amplitude = blob_amplitude
        self.jit = jit
        self.error_scaling = error_scaling
        
    
    def load_samples(self, samples):
        """Loads samples and initialize sample params"""
        self.samples = samples      
        self.image_size = samples[0].shape[0]
        self.sample_size = len(samples)
        
        self.bounds = (-0.5,self.image_size-0.5)
        self.single_blob_l1, self.single_blob_l2 = self._single_blob_error()
 
    def _make_gaussian(self, center, var, image_size):
        """Make a 2D symmetric gaussian"""
        x = np.arange(0, image_size, 1, float)
        y = x[:,np.newaxis]

        x0 = center[1]
        y0 = center[0]

        return np.exp(-0.5 * ((x-x0)**2 + (y-y0)**2) / var)

    def _create_blobs(self, centers):
        """Create a sample of gaussian blobs"""
        if len(centers)==0: # Zero case
            return np.zeros((self.image_size, self.image_size))
        else:
            return np.array([normalize_2d(self._make_gaussian(coord, self.blob_size, self.image_size))*self.blob_amplitude
                            for coord in centers]).sum(axis=0)

    def plot_fit(self, sample, centers):
        """Plot the best fit and the residual"""
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        y, x = zip(*centers)
        axs[0].imshow(sample)
        axs[0].scatter(x, y, c='r', alpha=0.5)

        residual = sample-self._create_blobs(centers)
        max = np.unravel_index(residual.argmax(), residual.shape)
        min = np.unravel_index(residual.argmin(), residual.shape)
        cmap_bounds = np.max([np.abs(residual.min()), np.abs(residual.max())])
        
        axs[1].set_title('residual')
        axs[1].imshow(residual, cmap='RdBu', vmin=-cmap_bounds, vmax=cmap_bounds)
        axs[1].scatter(x, y, c='r', alpha=0.5)
        axs[1].scatter(max[1], max[0], c='blue', marker='+')
        axs[1].scatter(min[1], min[0], c='white', marker='+')

        fig.suptitle(f'{len(centers)} blobs fit')
        plt.tight_layout()
        plt.show()
        plt.close()
        
    def _jitter(self, coords, jit):
        num = len(coords)
        push = circle_points([jit], [num])
        return np.array(coords) + push

    def _single_blob_error(self):
        single_blob = self._create_blobs([(int(self.image_size/2),int(self.image_size/2))])
        l1 = single_blob.sum()
        l2 = mse(single_blob, np.zeros((self.image_size, self.image_size)))
        return l1, l2

    def find_guess(self, sample, rel_peak_threshold=0.8, max_iters=500):
        """Find an initial guess using gaussian decomposition"""
        img_decomp = sample.copy()
        peak_coords = []
        peak_counts = []

        # Gaussian decomposition
        for _ in range(int(max_iters)):
            # Find max
            peak_coord = np.unravel_index(img_decomp.argmax(), img_decomp.shape)
            peak_val = np.max(img_decomp)
            
            # Stopping criterion
            if peak_val<=self.blob_amplitude*rel_peak_threshold:
                break
            
            # Remove gaussian with amplitude of maximum pixel
            x, y = np.mgrid[0:img_decomp.shape[0]:1, 0:img_decomp.shape[1]:1]
            pos = np.dstack((x, y))
            gaussian = normalize_2d(multivariate_normal(peak_coord, [[self.blob_size, 0], [0, self.blob_size]]).pdf(pos))*peak_val

            img_decomp = np.subtract(img_decomp, gaussian)

            # Record peak value and number of blobs for peak
            peak_coords.append(peak_coord)
            peak_counts.append(int(np.round(peak_val/self.blob_amplitude*1.2))) # NOTE: scalar multiplier chosen arbitrarily, come back to this

        # Record guess, one coordinate for each count
        guess = []
        for coord, count in zip(peak_coords, peak_counts):
            if count==1:
                guess.append(coord)
            else:
                # Jitter, put coord equally spaced on circle centered at maximum
                coords = [coord for _ in range(count)]
                coords = self._jitter(coords, self.jit)
                guess.extend(coords)

        if len(guess)>0:
            guess = np.array(guess).clip(*self.bounds)

        return guess
        
    def count_sample(self, sample,
                method='SLSQP', max_iters=20, err_threshold_rel=0.2,
                plot_progress=False):
        # Get guess of coordinates
        initial_guess = self.find_guess(sample)

        # Count
        guesses, errs, _ = self.count_recursive(
            [initial_guess], [], [], sample, method, 
            curr_iter=0, max_iters=max_iters, err_threshold_rel=err_threshold_rel,
            plot_progress=plot_progress
            )
        counts = [len(guess) for guess in guesses]

        # Index of best fit
        idx = np.argmin(errs)
        
        return guesses[idx], errs, counts
            
    def count_sample_single_step(self, sample, queue=None, # queue must be third param 
                                 method='SLSQP', max_iters=50,  
                                 plot_guess=False, plot_progress=False):
        """
        Count the number of blobs in a sample by fitting
        Keep fitting for counts at increments of 1 until a local error minima is found. 
        """

        # Get guess of coordinates
        initial_guess = self.find_guess(sample)
        guess = initial_guess.copy()

        # Plots points of the initial guess
        if plot_guess:
            y, x = zip(*guess)
            plt.imshow(sample)
            plt.scatter(x, y, c='r')
            plt.title(f'Initial guess: {len(guess)} blobs counted')
            plt.show()
            plt.close()

        def fit_objective(*args):
            """
            Function to minimize for fitting gaussian blobs
            """
            # args (centers, img), the centers must be in format (y0, y1, ... , x0, x1, ...)
            centers = (args[0][:int(len(args[0])/2)], args[0][int(len(args[0])/2):])
            centers = list(zip(*centers))
            img = args[1]

            # Grid for gaussian blob
            blobs = self._create_blobs(centers)

            # Calculate error
            error = mse(blobs, img)

            return error*self.error_scaling

        def minimize_objective(guess, sample, method='SLSQP', plot_progress=False):
            """ 
            Finds the best fit for a given guess of gaussian centers and sample image
            """
            # Print blob fitting
            if plot_progress:
                print(f'fitting for {len(guess)} blobs...', end=' ')

            minimize_start = time.time()

            # Sort for constraints
            guess = guess[guess[:, 0].argsort()]

            # Reshape centers guess to be [y0, y1, ..., x0, x1, ...]
            guess_transfm = np.concatenate(list(zip(*guess)))

            # Contraints and bounds
            n = len(guess)
            D = np.eye(n) - np.eye(n, k=-1)
            cons = [{'type': 'ineq', 'fun': lambda x: D @ x[:n]}] 

            bnds = np.array([self.bounds for _ in guess_transfm])
            
            # Minimize
            result = minimize(fit_objective, guess_transfm, args=(sample),
                            method=method, bounds=bnds, constraints=cons)

            # Reshape centers fit to be [(y0, x0), (y1, x1), ...]
            fit = (result.x[:int(len(result.x)/2)], result.x[int(len(result.x)/2):])
            fit = np.array(list(zip(*fit))).clip(*self.bounds) 

            minimize_end = time.time()
            
            if plot_progress:
                print('{0:.4f}s'.format(minimize_end-minimize_start))
                self.plot_fit(sample, fit)
                print(result)

            return fit, result.fun

        def add_guess(sample, centers):
            """
            Adds a guess at a resonable location
            """
            # Add guess at the pixel where the residual (image-fit) is the largest
            guess_img = self._create_blobs(centers)
            
            residual = sample - guess_img
            
            new_guess = np.unravel_index(residual.argmax(), residual.shape)
            
            return np.append(guess, [new_guess], axis=0), new_guess

        def remove_guess(sample, centers):
            """
            Removes the worst guess 
            """
            # Clip fit to within image
            centers_copy = centers.copy().clip(-0.5,len(sample)-0.5)
            
            # Remove guess where the residual (sample-fit) at the center pixel is the most negative 
            guess_img = self._create_blobs(centers_copy)
            
            sample_value_at_center = np.array([sample[int(center[0]), int(center[1])] for center in centers_copy])
            guess_value_at_center = np.array([guess_img[int(center[0]), int(center[1])] for center in centers_copy])
            residual = sample_value_at_center - guess_value_at_center
            
            worst_guess_idx = residual.argmin()
            
            return np.delete(guess, worst_guess_idx, axis=0), guess[worst_guess_idx]

        errs = [[], []]

        # Get initial fit using original guess
        if len(guess)>0:
            guess, err = minimize_objective(initial_guess, sample, method=method, plot_progress=plot_progress)
            errs[0].append(err)
            errs[1].append(len(guess))
        # 0 blob case
        else:
            guess = np.empty((0,2))
            err = mse(sample, np.zeros(sample.shape))
            errs[0].append(err)
            errs[1].append(len(guess))

        # Initialize check for reducing number of blobs
        check_lower_count = False

        # Add blobs until fit deproves
        for i in range(1, max_iters):
            # Remember previous guess
            prev_guess, prev_err = guess, err

            # Add blob
            guess, _ = add_guess(sample, guess)

            # Fit
            guess, err = minimize_objective(guess, sample, method=method, plot_progress=plot_progress)
            errs[0].append(err)
            errs[1].append(len(guess))

            if err>prev_err:
                # If fit immediately deproves, reduce counts
                if i==1 and len(prev_guess)>0:
                    check_lower_count = True
                    guess, err = prev_guess, prev_err
                    break
                # Stop when error increases
                else:
                    if plot_progress:
                        print('best fit found')
                    # For multiprocessing tracking
                    if queue is not None:
                        queue.put(1)
                    return prev_guess, errs

        # Remove blobs until fit deproves
        if check_lower_count==True:
            # Remember previous guess
            prev_guess, prev_err = guess, err            
            
            # 0 blob case
            if len(guess)==1:
                guess = np.empty((0,2))
                err = mse(sample, np.zeros(sample.shape))
                errs[0].append(err)
                errs[1].append(len(guess))
                # For multiprocessing tracking
                if queue is not None:
                    queue.put(1)
                # Return lowest error
                if err>prev_err:
                    return prev_guess, errs
                else:
                    return guess, errs

            # Remove blob
            guess, _ = remove_guess(sample, guess)

            # Fit
            guess, err = minimize_objective(guess, sample, method=method, plot_progress=plot_progress)
            errs[0].append(err)
            errs[1].append(len(guess))

            # Stop when error increases
            if err>prev_err:
                if plot_progress:
                    print('best fit found')
                # For multiprocessing tracking
                if queue is not None:
                    queue.put(1)
                return prev_guess, errs

        # Return last guess and err if maximum iterations reached
        if plot_progress:
            print('max iters reached')
            
        # For multiprocessing tracking
        if queue is not None:
            queue.put(1)
            
        return guess, errs
    
    def count(self, err_threshold_rel=0.3, method='SLSQP', mode='multi', plot_progress=False, progress_bar=True):
        """
        Count blobs of the loaded samples
        """
        # Time track
        start = time.time()
        
        # Count using single core
        if mode=='single':
            fit_coords = []
            counts = []
            
            for sample in tqdm(self.samples, disable=not progress_bar):
                fit_coord, errs, _ = self.count_sample_single_step(sample, method=method, plot_progress=plot_progress)
                
                fit_coords.append(fit_coord)
                count = len(fit_coord)
                counts.append(count)

        # Count using multi core    
        if mode=='multi':
            with Manager() as manager:
                queue = manager.Queue()
                
                # Initialize the progress bar
                if progress_bar:
                    pbar = tqdm(total=self.sample_size, desc="fitting", unit="sample") if progress_bar else None

                # Using ProcessPoolExecutor to run tasks in parallel
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    # Submit tasks to the executor, passing the queue for progress updates
                    futures = [executor.submit(self.count_sample_single_step, sample, queue) for sample in self.samples]

                    # Start a thread or a separate process to handle progress bar updates in the main thread
                    if progress_bar:    
                        update_progress_bar(queue, self.sample_size, pbar)

                    # Wait for all tasks to finish and gather results
                    results = [future.result() for future in futures]
                # Extract the fit_coords and errors from the results
                fit_coords = [result[0] for result in results]
                errs = [result[1] for result in results]
                counts = [len(coord) for coord in fit_coords]
        
        # Time track
        end = time.time()

        if not progress_bar and plot_progress:
            print(f'fit {self.sample_size} samples')
            print('per sample: {:4f}s | total: {:4f}s'.format((end-start)/self.sample_size, end-start))
            print()
            
        return fit_coords, np.array(counts)
    
    """
    Not in use
    New counting method that takes steps (of count) based on the gradient,
    Too volatile for small number of blobs and not fast enough for large number of blobs
    Error threshold stopping criterion is very dependent on the blob count --> needs to be carefully set  
    """
    def minimize_objective(self,
        guess, sample, size, amplitude,
        method='SLSQP',
        plot_progress=False
        ):
        """Minimize the objective function"""
        
        def fit_objective(*args):
            """Function to minimize for fitting gaussian blobs"""
            # args (centers, img, size, amplitude), the centers must be in format (y0, y1, ... , x0, x1, ...)
            centers = (args[0][:int(len(args[0])/2)], args[0][int(len(args[0])/2):])
            centers = list(zip(*centers))
            img = args[1]
            size = args[2]
            amplitude = args[3]
            image_size = img.shape[0]

            # Grid for gaussian blob
            blobs = create_blobs(centers, image_size, size, amplitude)

            # Calculate error
            error = mse(blobs, img)

            return error*1e7
        
        # Print blob fitting
        if plot_progress:
            print(f'fitting for {len(guess)} blobs...', end=' ')

        minimize_start = time.time()

        # Sort for constraints
        guess = guess[guess[:, 0].argsort()]

        # Reshape centers guess to be [y0, y1, ..., x0, x1, ...]
        guess_transfm = np.concatenate(list(zip(*guess)))

        # Contraints and bounds
        n = len(guess)
        D = np.eye(n) - np.eye(n, k=-1)
        cons = [{'type': 'ineq', 'fun': lambda x: D @ x[:n]}]

        bnds = np.array([self.bounds for _ in guess_transfm])

        # Minimize
        result = minimize(fit_objective, guess_transfm, args=(sample, size, amplitude),
                            method=method, bounds=bnds, constraints=cons)

        # Reshape centers fit to be [(y0, x0), (y1, x1), ...]
        fit = (result.x[:int(len(result.x)/2)], result.x[int(len(result.x)/2):])
        fit = list(zip(*fit))

        minimize_end = time.time()
        if plot_progress:
            print('{0:.4f}s'.format(minimize_end-minimize_start))

        if plot_progress:
            self.plot_fit(sample, fit)
            print(result)

        return fit, result.fun

    def add_guess(self, residual, guess):
        """"Add single guess"""
        # New guess is put at location with the largest error
        new_guess = np.unravel_index(residual.argmax(), residual.shape)
        return np.append(guess, [new_guess], axis=0)

    def remove_guess(self, residual, guess):
        """Remove single guess"""
        min = np.unravel_index(residual.argmin(), residual.shape)
        # Nearest center to minimum residual point is removed
        worst_guess = guess[spatial.KDTree(guess).query(min)[1]]
        return np.delete(guess, np.where((guess==worst_guess).all(axis=1))[0], axis=0)

    def remove_guesses(self, residual, guess, n):
        """Remove multiple guesses"""
        # Clip to avoid index errors
        guess_clip = guess.copy().clip(-0.5, self.image_size-0.5)
        
        # Get residual at center
        residual_at_centers = np.array([residual[int(center[0]), int(center[1])] for center in guess_clip])
        
        # Mask to only look at values sufficiently negative
        masked_residual = residual_at_centers[residual_at_centers<=-self.blob_amplitude*0.5]
    
        # Find minimum centers    
        worst_guess_idxs = masked_residual.argsort()[:np.min([n,len(guess),len(masked_residual)])]
        
        return np.delete(guess, worst_guess_idxs, axis=0)

    def add_guesses(self, residual, guess, n):
        """Add multiple guesses using Gaussian decomp"""
        img_decomp = residual.copy()
        peak_coords = []
        peak_counts = []

        # Gaussian decomposition
        for _ in range(n):
            # Find max
            peak_coord = np.unravel_index(img_decomp.argmax(), img_decomp.shape)
            peak_val = np.max(img_decomp)
            
            # Early stopping criterion
            if peak_val<=self.blob_amplitude*0.5:
                break

            # Remove gaussian with amplitude of maximum pixel
            x, y = np.mgrid[0:img_decomp.shape[0]:1, 0:img_decomp.shape[1]:1]
            pos = np.dstack((x, y))
            gaussian = normalize_2d(multivariate_normal(peak_coord, [[self.blob_size, 0], [0, self.blob_size]]).pdf(pos))*peak_val

            img_decomp = np.subtract(img_decomp, gaussian)

            # Record peak value and number of blobs for peak
            peak_coords.append(peak_coord)
            peak_counts.append(np.max([int(np.round(peak_val/self.blob_amplitude)*1.3), 1])) # NOTE: scaling value chosen arbitrarily, come back to this

            # Stop when n guesses are added
            if np.sum(peak_counts)>=n:
                break
        
        # Record guess, one coordinate for each count
        new_centers = []
        for coord, count in zip(peak_coords, peak_counts):
            if count==1:
                new_centers.append(coord)
            else:
                # Jitter, put coord equally spaced on circle centered at maximum
                coords = [coord for _ in range(count)]
                coords = self._jitter(coords, 0.5)
                new_centers.extend(coords)

        new_centers = np.array(new_centers).clip(*self.bounds)
        
        if len(new_centers)==0: # No new added centers case
            new_guess = guess
        else:
            new_guess = np.append(guess, new_centers, axis=0)
        
        return new_guess 

    def count_recursive(self,
        guesses, errs, counts,
        sample, method,
        step='error', curr_iter=0, max_iters=10, err_threshold_rel=0.3,
        plot_progress=False
        ):
        """
        Count the number of blobs on a sample recursively by fitting.
        Keep fitting for counts at increments decided based on the error until certain conditions are met.
        """
        # Initiate state
        state = 'run'
        if plot_progress:
            print()
            print(f'iteration {curr_iter}')
        
        # Find count
        count = len(guesses[-1])
        counts.append(count)

        # Fit
        if len(guesses[-1])>0:
            fit_guess, err = self.minimize_objective(
                np.array(guesses[-1]), sample, 
                self.blob_size, self.blob_amplitude,
                method=method, plot_progress=plot_progress)
            fit_guess = np.array(fit_guess)
            guesses[-1] = fit_guess
            errs.append(err)
        else: # zero case
            fit_guess = np.empty((0,2))
            err = mse(sample, np.zeros(self.image_size))
            guesses[-1] = fit_guess
            errs.append(err)
    
        # Stop if exceed set max iterations
        if curr_iter > max_iters:
            if plot_progress:
                print('max iters reached')
            state = 'complete'
            return fit_guess, errs, state 
        curr_iter += 1
    
        # Stop if local minimum is found
        best_count = find_local_min(errs, counts)
        if best_count is not None:
            if plot_progress:
                print('local minima found')
            state = 'complete'
            return guesses, errs, state

        # Find residual
        residual = sample - self._create_blobs(fit_guess)
        residual_sum = residual.sum()
        residual_l1 = np.abs(residual).sum()
        residual_l1 = np.where(residual_l1>self.single_blob_l1*0.05, 
                               residual_l1, 0) # Remove sources of small error, simple low-pass (not sure if this helps?)
        
        # Check error threshold
        if residual_l1<self.single_blob_l1*err_threshold_rel: 
            if plot_progress:
                print('err threshold reached')
            state = 'complete'
            return guesses, errs, state

        # Add/remove multiple guesses
        if step=='error' and state=='run':
            # Check condition to change step state
            if residual_l1<self.single_blob_l1*5:
                # Change step state
                step='single'
                if plot_progress:
                    print('error threshold reached: switch to single step')
                
            else:
                if residual_sum<0:
                    # Get estimate of number of guesses to remove
                    masked_l1 = -np.where(residual<=-self.blob_amplitude/10, residual, 0).sum()
                    n = np.min([int(np.round(masked_l1/self.single_blob_l1)), count]) # Count cannot be negative
                    
                    # Remove a maximum of n guesses
                    if n!=0: 
                        new_guess = self.remove_guesses(residual, fit_guess, n)
                    else: # No guesses to remove
                        new_guess = fit_guess
                    
                    # Actual number of guesses removed    
                    n_removed = len(fit_guess) - len(new_guess)
                    
                    # Switch to single step if only 1 guess is removed
                    if n_removed>1:
                        if plot_progress:
                            print(f'removed {n_removed} guesses')
                    else:
                        step = 'single'
                        if plot_progress:
                            print('only 1 guess to remove: switch to single step')

                elif residual_sum>=0:
                    # Get estimate of number of guesses to add
                    masked_l1 = np.where(residual>=self.blob_amplitude/10, residual, 0).sum()
                    n = int(np.round(masked_l1/self.single_blob_l1))
                    
                    # Add a maximum of n guesses
                    if n!=0: 
                        new_guess = self.add_guesses(residual, fit_guess, n)
                    else: # No guesses to add
                        new_guess = fit_guess
                        
                    # Actual number of guesses added     
                    n_added = len(new_guess) - len(fit_guess)
                    
                    # Switch to single step if only 1 guess is added
                    if n_added>1:
                        if plot_progress:
                            print(f'added {n_added} guesses')
                    else:
                        step = 'single'
                        if plot_progress:
                            print('only 1 guess to add: switch to single step')
            
            # Fit new guess             
            if step=='error':
                # Append new guess
                guesses.append(new_guess)
                
                # Keep fitting
                guesses, errs, state = self.count_recursive(
                    guesses, errs, counts, sample,
                    method, step, curr_iter, max_iters, err_threshold_rel, 
                    plot_progress
                    )
            elif step=='single':
                # Step from lowest error
                min_idx = np.argmin(errs)
                fit_guess = guesses[min_idx]
                
                # Take first single step
                if residual_sum<0 and count!=0:
                    # Remove
                    if plot_progress:
                        print('removing 1 guess')
                    guesses.append(self.remove_guess(residual, fit_guess))
                    
                    # Keep fitting
                    guesses, errs, state = self.count_recursive(
                        guesses, errs, counts, sample,
                        method, step, curr_iter, max_iters, err_threshold_rel, 
                        plot_progress
                        )
                elif residual_sum>=0 or count==0:
                    # Add
                    if plot_progress:
                        print('adding 1 guess')
                    guesses.append(self.add_guess(residual, fit_guess))
                    
                    # Keep fitting
                    guesses, errs, state = self.count_recursive(
                        guesses, errs, counts, sample,
                        method, step, curr_iter, max_iters, err_threshold_rel, 
                        plot_progress
                        )
                
        # Add/remove single guess
        if step=='single' and state=='run':
            # Find gradient
            grad = (errs[-1] - errs[-2]) / (counts[-1] - counts[-2])

            if grad<0 or count==0: # Add for negative gradient
                # Find nearest count that has not been fit
                target_count = count + find_dist_to_next_consec(counts, count, 1)
                
                # Add
                fit_guess = guesses[int(np.where(counts==target_count-1)[0][0])]
                if plot_progress:
                    print('adding 1 guess')
                guesses.append(self.add_guess(residual, fit_guess))
                
                # Keep fitting
                guesses, errs, state = self.count_recursive(
                    guesses, errs, counts, sample,
                    method, step, curr_iter, max_iters, err_threshold_rel, 
                    plot_progress
                    )
            elif grad>=0: # Remove for positive gradient
                # Find nearest count that has not been fit
                target_count = np.max([count - find_dist_to_next_consec(counts, count, -1), 0])
                
                # Remove
                fit_guess = guesses[int(np.where(counts==target_count+1)[0][0])]
                if plot_progress:
                    print('removing 1 guess')
                guesses.append(self.remove_guess(residual, fit_guess))
                
                # Keep fitting
                guesses, errs, state = self.count_recursive(
                    guesses, errs, counts, sample,
                    method, step, curr_iter, max_iters, err_threshold_rel, 
                    plot_progress
                    )

        # Exit once complete state is reached
        if state=='complete':
            return guesses, errs, state        


"""Depreciated"""

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

def euclidean_dist(coord_1, coord_2):
    """
    Calculates euclidean distance between 2 points
    """
    return ((coord_1[0]-coord_2[0])**2+(coord_1[1]-coord_2[1])**2)**0.5

def generate_random_coords(image_size, n):
    """
    Generate n number of random 2D coordinates within the range of the image size
    """
    return np.random.rand(n, 2)*image_size-0.5

def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]

def find_pair_distances(sample_1, sample_2):
    """
    Find distances between all pairs of points given two samples.
    If both samples are the same, the distances to all pairs of points within one sample will be calculated.
    """
    
    ## SLOW METHOD
    # if (len(sample_1)==len(sample_2)) and ((sample_1==sample_2).all()):
    #     sample = sample_1
    #     sample_temp = np.array(sample).copy()
    #     distances = []
        
    #     # Calculate distances
    #     for coord in sample:
    #         for i in range(len(sample_temp)-1):
    #             distances.append(euclidean_dist(coord, sample_temp[i+1]))
    #         sample_temp = sample_temp[1:]
    # else:
    #     distances = []
    #     for coord_1 in sample_1:
    #         for coord_2 in sample_2:
    #             distances.append(euclidean_dist(coord_1, coord_2))
    
    if (len(sample_1)==len(sample_2)) and ((sample_1==sample_2).all()):
        distances = upper_tri_masking(cdist(sample_1, sample_2))
    else:
        distances = cdist(sample_1, sample_2).flatten()
                
    return distances

def find_pair_hist(sample_1, sample_2, bins):
    """
    Returns the histogram of distances given two samples of points.
    """
    distances = find_pair_distances(sample_1, sample_2)
    n, edges = np.histogram(distances, bins=bins)
    
    return n/len(distances), edges

def two_point_correlation(sample, image_size, bins=10, rel_random_n=5):
    """
    Calculates the two point correlation using the Landy Szalay estimator given an image sample
    """
    random_sample = generate_random_coords(image_size,int(max(len(sample)*rel_random_n, 1e3)))
    
    dd_norm, edges = find_pair_hist(sample, sample, bins)
    rr_norm, _ = find_pair_hist(random_sample, random_sample, bins)
    dr_norm, _ = find_pair_hist(sample, random_sample, bins)
    
    return (dd_norm-2*dr_norm+rr_norm)/rr_norm, edges

def stack_two_point_correlation(point_coords, image_size, bins=10, rel_random_n=5, progress_bar=False):
    """
    Calculates the two point correlation using the Landy Szalay estimator given a series of image samples
    NOTE: THIS IS WRONG. SHOULD NOT BE STACKING SAMPLES
    """
    bins = np.linspace(0, image_size,bins)
    dd_dists = []
    rr_dists = []
    dr_dists = []
    
    for sample in tqdm(point_coords, disable=not progress_bar):
        sample = np.array(sample)
        random_sample = generate_random_coords(image_size, int(max(len(sample)*rel_random_n, 1e3)))
        
        # Calculate distances
        dd_dists.extend(find_pair_distances(sample, sample))
        rr_dists.extend(find_pair_distances(random_sample, random_sample))
        dr_dists.extend(find_pair_distances(sample, random_sample))
    
    dd, edges = np.histogram(dd_dists, bins=bins)
    rr, edges = np.histogram(rr_dists, bins=bins)
    dr, edges = np.histogram(dr_dists, bins=bins)
    
    dd_norm = dd/len(dd_dists)
    rr_norm = rr/len(rr_dists)
    dr_norm = dr/len(dr_dists)

    corr = (dd_norm-2*dr_norm+rr_norm)/rr_norm
    
    return corr, edges

