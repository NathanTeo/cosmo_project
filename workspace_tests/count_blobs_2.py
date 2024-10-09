import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from tqdm.auto import tqdm
import time
import concurrent.futures

from scipy.optimize import minimize

def normalize_2d(matrix):
    return (matrix-np.min(matrix))/(np.max(matrix)-np.min(matrix))

def make_gaussian(center, var, image_size):
        """ Make a square gaussian kernel"""
        x = np.arange(0, image_size, 1, float)
        y = x[:,np.newaxis]

        x0 = center[1]
        y0 = center[0]

        return np.exp(-0.5 * ((x-x0)**2 + (y-y0)**2) / var)

def create_blobs(centers, blob_size, blob_amplitude, image_size):
    """Create an image of gaussian blobs"""
    return np.array([normalize_2d(make_gaussian(coord, blob_size, image_size))*blob_amplitude
                        for coord in centers]).sum(axis=0)

def create_samples(num_of_samples, num_of_blobs, blob_size, blob_amplitude, image_size):
    """Create samples of gaussians blobs"""
    centers = np.random.rand(num_of_samples, num_of_blobs, 2)*image_size    
    samples = [create_blobs(center, blob_size, blob_amplitude, image_size) for center in centers]
    
    return np.array(samples)
    
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
    def __init__(self, samples, blob_size, blob_amplitude, jit=0.5, error_scaling=1e7):
        self.samples = samples
        self.blob_size = blob_size
        self.blob_amplitude = blob_amplitude
        self.image_size = samples[0].shape[0]
        self.sample_size = len(samples)
        
        self.jit = jit
        self.error_scaling = error_scaling
       

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

        residual = sample-self._create_blobs(centers)
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

    def find_guess(self, sample, rel_peak_threshold=0.8, max_iters=100):
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
            peak_counts.append(int(np.round(peak_val/self.blob_amplitude*1.3))) # NOTE: 1.1 chosen arbitrarily, come back to this

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
                            method=method, bounds=bnds, constraints=cons,
                            options={'max_iters': 40})

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
        guess, err = minimize_objective(initial_guess, sample, method=method, plot_progress=plot_progress)
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
                if i==1:
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
            for i in range(1, max_iters):
                # No blobs case
                if len(guess)==1:
                    guess = np.array([])
                    err = mse(sample, np.zeros(sample.shape))
                    return guess, errs

                # Remember previous guess
                prev_guess, prev_err = guess, err

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
    
    def count(self, method='SLSQP', mode='multi', plot_progress=False):
        # Time track
        start = time.time()
        
        # Count using single core
        if mode=='single':
            fit_coords = []
            counts = []
            
            for sample in self.samples:
                fit_coord, errs = self.count_sample(sample, method=method, plot_guess=False, plot_progress=plot_progress)
                
                fit_coords.append(fit_coord)
                count = len(fit_coord)
                counts.append(count)

        # Count using multi core    
        if mode=='multi':
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = executor.map(self.count_sample, self.samples)

                fit_coords = []
                errs = []

                for result in results:
                    fit_coords.append(result[0])
                    errs.append(result[1])

            counts = [len(coord) for coord in fit_coords]
        
        # Time track
        end = time.time()

        print(f'fit {self.sample_size} samples')
        print('mean time per sample: {:4f}s'.format((end-start)/self.sample_size))
        print('total: {:4f}s'.format(end-start))
        print()
            
        return fit_coords, np.array(counts) 

    def plot_sample(self, sample, coord, count):
        y, x = zip(*coord)
        plt.imshow(sample)
        plt.scatter(x, y, c='r', alpha=0.5)
        plt.title(f'{count} blobs counted')
        plt.show()
        plt.close()
        
def plot_sample(sample, coord, count):
    y, x = zip(*coord)
    plt.imshow(sample)
    plt.scatter(x, y, c='r', alpha=0.5)
    plt.title(f'{count} blobs counted')
    plt.show()
    plt.close()

num_of_imgs = 1
num_of_blobs = 50
blob_size = 5
blob_amplitude = 1/num_of_blobs
image_size = 128

if __name__=="__main__":
    print('test')
    
    samples = create_samples(num_of_imgs, num_of_blobs, blob_size, blob_amplitude, image_size)

    counter = blobCounter(samples, blob_size, blob_amplitude)
    
    print('---------')
    print('SLSQP')
    print('---------')
    coords, counts = counter.count(method='SLSQP', mode='single', plot_progress=True)
    # coords, counts = counter.count(method='SLSQP', mode='multi')
    print(f'sample counts: {counts}')
    print(f'accuracy: {np.where(counts==num_of_blobs, 1, 0).mean()}')
    plot_sample(samples[0], coords[0], counts[0])
    '''
    print('---------')
    print('COBYLA')
    print('---------')
    coords, counts = counter.count(method='COBYLA', mode='single')
    coords, counts = counter.count(method='COBYLA', mode='multi')
    print(f'sample counts: {counts}')
    print(f'accuracy: {np.where(counts==num_of_blobs, 1, 0).mean()}')
    plot_sample(samples[0], coords[0], counts[0])
 
    print('---------')
    print('trust_constr')
    print('---------')
    coords, counts = counter.count(method='trust-constr', mode='single')
    coords, counts = counter.count(method='trust-constr', mode='multi')
    print(f'sample counts: {counts}')
    print(f'accuracy: {np.where(counts==num_of_blobs, 1, 0).mean()}')
    plot_sample(samples[0], coords[0], counts[0])
    '''