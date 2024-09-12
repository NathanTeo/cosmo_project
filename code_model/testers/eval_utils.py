"""
Author: Nathan Teo

This script contains functions useful for model evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm
from scipy.stats import multivariate_normal
import random
import time
import concurrent.futures
from scipy.optimize import minimize

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
    return np.random.rand(n, 2)*image_size-0.5

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
        random_sample_size = len(sample)*rel_random_n + 1 # +1 to prevent 0 case
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

"""Depreciated"""
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

def gaussian_decomposition(img, blob_size, min_peak_threshold=0.08, max_iters=20):
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

        # Create gaussian with amplitude of maximum
        x, y = np.mgrid[0:img_decomp.shape[0]:1, 0:img_decomp.shape[1]:1]
        pos = np.dstack((x, y))
        gaussian = normalize_2d(multivariate_normal(peak_coord, [[blob_size, 0], [0, blob_size]]).pdf(pos))*peak_val

        # Subtract created gaussian from image
        img_decomp = np.subtract(img_decomp, gaussian)        
        
        # Stop once threshold maximum is below threshold
        if peak_val<=min_peak_threshold:
            break
        
        # Find if residual from subtraction is relatively flat, ignore point if residual is very negative
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
        img_blob_coords, img_peak_vals = gaussian_decomposition(img, blob_size, min_peak_threshold, max_iters)
        blob_coords.append(img_blob_coords)
        blob_nums.append(len(img_blob_coords))
        peak_vals.append(img_peak_vals)

    return blob_coords, blob_nums, peak_vals

def count_blobs_from_peaks(imgs_peak_vals, generation_blob_number):
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
    
def circle_points(r, n):
    circles = []
    for r, n in zip(r, n):
        rand_dir = np.random.uniform(low=0, high=2*np.pi)
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        x = r * np.cos(t + rand_dir)
        y = r * np.sin(t + rand_dir)
        circles.extend(np.c_[y, x])
    return np.array(circles)
 
def mse(A, B):
    """Mean squared error"""
    return (np.square(A - B)).mean(axis=None)

class blobCounter():
    def __init__(self, blob_size, blob_amplitude, jit=0.5, error_scaling=1e7):

        self.blob_size = blob_size
        self.blob_amplitude = blob_amplitude
        self.jit = jit
        self.error_scaling = error_scaling
    
    def load_samples(self, samples):
        self.samples = samples      
        self.image_size = samples[0].shape[0]
        self.sample_size = len(samples)
 
    def _make_gaussian(self, center, var, image_size):
        """ Make a square gaussian kernel"""
        x = np.arange(0, image_size, 1, float)
        y = x[:,np.newaxis]

        x0 = center[1]
        y0 = center[0]

        return np.exp(-0.5 * ((x-x0)**2 + (y-y0)**2) / var)

    def _create_blobs(self, centers):
        """Create an image of gaussian blobs"""
        return np.array([normalize_2d(self._make_gaussian(coord, self.blob_size, self.image_size))*self.blob_amplitude
                            for coord in centers]).sum(axis=0)

    def plot_fit(self, sample, centers):
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        y, x = zip(*centers)
        axs[0].imshow(sample)
        axs[0].scatter(x, y, c='r', alpha=0.5)

        residual = sample-self._create_blobs(centers, self.blob_size, self.blob_amplitude, sample.shape[0])
        max = np.unravel_index(residual.argmax(), residual.shape)
        min = np.unravel_index(residual.argmin(), residual.shape)

        axs[1].set_title('residual')
        axs[1].imshow(residual)
        axs[1].scatter(x, y, c='r', alpha=0.5)
        axs[1].scatter(max[1], max[0], c='blue', marker='+')
        axs[1].scatter(min[1], min[0], c='white', marker='+')

        fig.suptitle(f'{len(centers)} blobs fit')
        plt.tight_layout()
        plt.show()
        plt.close()
        
    def _jitter(self, coords):
        num = len(coords)
        push = circle_points([self.jit], [num])
        return np.array(coords) + push

    def find_guess(self, sample, rel_peak_threshold=0.8, max_iters=30):
        """Find an initial guess using gaussian decomp"""
        img_decomp = sample.copy()
        peak_coords = []
        peak_counts = []

        # Gaussian decomposition
        for _ in range(max_iters):
            # Find max
            peak_coord = np.unravel_index(img_decomp.argmax(), img_decomp.shape)
            peak_val = np.max(img_decomp)

            # Remove gaussian with amplitude of maximum pixel
            x, y = np.mgrid[0:img_decomp.shape[0]:1, 0:img_decomp.shape[1]:1]
            pos = np.dstack((x, y))
            gaussian = normalize_2d(multivariate_normal(peak_coord, [[self.blob_size, 0], [0, self.blob_size]]).pdf(pos))*peak_val

            img_decomp = np.subtract(img_decomp, gaussian)

            # Stopping criterion
            if peak_val<=self.blob_amplitude*rel_peak_threshold:
                break

            # Record peak value and number of blobs for peak
            peak_coords.append(peak_coord)
            peak_counts.append(int(np.round(peak_val/self.blob_amplitude)*1.1)) # NOTE: 1.1 chosen arbitrarily, come back to this

        # Record guess, one coordinate for each count
        guess = []
        for coord, count in zip(peak_coords, peak_counts):
            if count==1:
                guess.append(coord)
            else:
                # Jitter, put coord equally spaced on circle centered at maximum
                coords = [coord for _ in range(count)]
                coords = self._jitter(coords)
                guess.extend(coords)

        guess = np.array(guess).clip(-0.5, self.image_size-0.5)

        return guess

    def count_sample(self, sample, method='SLSQP', max_iters=10, plot_guess=False, plot_progress=False):
        """Count the number of blobs in a sample by fitting"""

        # Get guess of coordinates
        initial_guess = self.find_guess(sample)
        guess = initial_guess.copy()

        if plot_guess:
            y, x = zip(*guess)
            plt.imshow(sample)
            plt.scatter(x, y, c='r')
            plt.title(f'Initial guess: {len(guess)} blobs counted')
            plt.show()
            plt.close()

        def fit_objective(*args):
            """Function to minimize for fitting gaussian blobs"""
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

            bnds = np.array([(-0.5, self.image_size-0.5) for _ in guess_transfm])
            
            # Minimize
            result = minimize(fit_objective, guess_transfm, args=(sample),
                            method=method, bounds=bnds, constraints=cons)

            # Reshape centers fit to be [(y0, x0), (y1, x1), ...]
            fit = (result.x[:int(len(result.x)/2)], result.x[int(len(result.x)/2):])
            fit = list(zip(*fit))

            minimize_end = time.time()
            
            if plot_progress:
                print('{0:.4f}s'.format(minimize_end-minimize_start))
                self.plot_fit(sample, fit)
                print(result)

            return fit, result.fun

        def add_guess(sample, centers):
            # New guess is put at location with the largest error
            guess_img = self._create_blobs(centers)
            residual = sample - guess_img
            new_guess = np.unravel_index(residual.argmax(), residual.shape)
            # print(f'{new_guess} added')
            return np.append(guess, [new_guess], axis=0), new_guess

        def remove_guess(sample, centers):
            # Guess with the largest error is removed
            guess_img = self._create_blobs(centers)
            sample_value_at_center = np.array([sample[int(center[0]), int(center[1])] for center in centers])
            guess_value_at_center = np.array([guess_img[int(center[0]), int(center[1])] for center in centers])
            residual = sample_value_at_center - guess_value_at_center
            worst_guess_idx = residual.argmin()
            # print(f'{guess[worst_guess_idx]} removed')
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
                return prev_guess, errs

        # Return last guess and err if maximum iterations reached
        if plot_progress:
            print('max iters reached')
        return guess, errs
    
    def count(self, method='SLSQP', mode='multi', plot_progress=False, progress_bar=True):
        # Time track
        start = time.time()
        
        # Count using single core
        if mode=='single':
            fit_coords = []
            counts = []
            
            for sample in tqdm(self.samples, disable=not progress_bar):
                fit_coord, errs = self.count_sample(sample, method=method, plot_guess=False, plot_progress=plot_progress)
                
                fit_coords.append(fit_coord)
                count = len(fit_coord)
                counts.append(count)

        # Count using multi core    
        if mode=='multi':
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(tqdm(executor.map(self.count_sample, self.samples),
                                    disable=not progress_bar, total=self.sample_size         
                                    ))

                fit_coords = []
                errs = []

                for result in results:
                    fit_coords.append(result[0])
                    errs.append(result[1])

            counts = [len(coord) for coord in fit_coords]
        
        # Time track
        end = time.time()

        if not progress_bar and plot_progress:
            print(f'fit {self.sample_size} samples')
            print('per sample: {:4f}s | total: {:4f}s'.format((end-start)/self.sample_size, end-start))
            print()
            
        return fit_coords, np.array(counts) 

    def plot_sample(self, sample, coord, count):
        y, x = zip(*coord)
        plt.imshow(sample)
        plt.scatter(x, y, c='r', alpha=0.5)
        plt.title(f'{count} blobs counted')
        plt.show()
        plt.close()