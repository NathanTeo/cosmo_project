"""
Author: Nathan Teo

This script contains all functions and classes used to realize samples in the script blob_realization.py
"""

import random
import os
import shutil
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import astropy.units as u
from scipy.special import gamma
import concurrent.futures
from multiprocessing import Manager
import time

def update_progress_bar(queue, total, pbar):
    """This function updates the progress bar from the main thread"""
    while pbar.n < total:
        queue.get()  # Block until there is progress to update
        pbar.update(1)  # Update the progress bar by 1 unit

def sample_power_law(a, b, p, size=1, rng=np.random.default_rng()):
    """Sample x^{-p} for a<=x<=b"""
    r = rng.random(size=size)*(b-a) + a
    return r**(-p)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def init_param(config, param, default=None):
    """Initialize parameter and return default value if key is not found in config dictionary"""
    try:
        return config[param]
    except KeyError:
        return default

class blobDataset():
    """Create real dataset for training, realize blobs"""
    def __init__(self, **params):
        """Initialize parameters"""
        self.image_size = params['image_size']
        seed = params['seed']
        self.blob_size = params['blob_size']
        self.sample_num = params['sample_num']
        self.blob_num =  params['blob_num']
        self.num_distribution = params['num_distribution']
        self.blob_amplitude = init_param(params, 'blob_amplitude', 1)
        self.amplitude_distribution = init_param(params, 'amplitude_distribution', 'delta')
        self.clustering = init_param(params, 'clustering')
        self.pad = init_param(params, 'pad', 0)
        self.noise = init_param(params, 'noise', 0)
        self.min_dist = init_param(params, 'minimum_distance')
        
        if self.pad == 'auto':
            self.pad = self.blob_size 
        self.generation_matrix_size = self.image_size + self.pad*2
        
        self.file_name = 'bn{}{}-cl{}-is{}-bs{}-ba{}{}-md{}-sn{}-sd{}-ns{}'.format(
            self.blob_num, self.num_distribution[0], 
            '{:.0e}_{:.0e}'.format(*self.clustering) if self.clustering is not None else '_',
            self.image_size, self.blob_size, 
            '{:.0e}'.format(self.blob_amplitude), self.amplitude_distribution[0],
            self.min_dist, f'{self.sample_num:.0e}', seed, self.noise
        )
        
        '''
        How seeding works:
        The seed RNG is initialized here with the chosen generation seed.
        The seed RNG produces random integer seeds, one for each sample.
        These seeds are used to initialize a local RNG for each sample made.
        This is done so that during multiprocessing the RNGs are all unqiue, 
        using np.random may result in the same RNG state to be passed to two or more processes
        '''
        # Seed rng
        seed_rng = np.random.default_rng(seed=seed)
        self.seeds = seed_rng.choice(int(self.sample_num*5), size=self.sample_num, replace=False)
  
        # Initiate center generator
        self.center_generator = centerGenerator(self.blob_num, self.image_size)
        
    def make_gaussian(self, center, var, image_size):
            """ Make a symmetric 2D gaussian"""
            x = np.arange(0, image_size, 1, float)
            y = x[:,np.newaxis]

            x0 = center[1]
            y0 = center[0]

            return np.exp(-0.5 * ((x-x0)**2 + (y-y0)**2) / var)

    def create_blobs(self, centers, blob_amplitudes=None):
        """Create an image of gaussian blobs given the centers and amplitudes of the blobs"""
        if centers.shape[0]>0:
            if blob_amplitudes is None: # realize blobs of the same size
                return np.array([normalize_2d(self.make_gaussian(coord, self.blob_size, self.generation_matrix_size))*self.blob_amplitude
                                for coord in centers]).sum(axis=0)
            else:
                return np.array([normalize_2d(self.make_gaussian(coord, self.blob_size, self.generation_matrix_size))*blob_amplitude
                    for coord, blob_amplitude in zip(centers, blob_amplitudes)]).sum(axis=0)
            
        else: # Empty image if there are no blob centers
            return np.zeros((self.generation_matrix_size, self.generation_matrix_size))
    
    def realize_sample(self, seed=None, queue=None):
        """Realize a single sample for the dataset"""
        # Set new random seed, necessary for multiprocessing to ensure each task is assigned a unique rng
        local_rng = np.random.default_rng(seed)
        
        # Generate points for centers
        centers = self.center_generator.generate(self.num_distribution, self.clustering, self.min_dist, rng=local_rng)
        count = centers.shape[0]
        
        # Sample amplitudes
        if self.amplitude_distribution=='delta':
            amps = None
        elif self.amplitude_distribution=='power':
            amps = sample_power_law(0.1, 1, 1, count, rng=local_rng)*self.blob_amplitude # minimum of 1 maximum of 10
        
        # Create blobs
        sample = self.create_blobs(centers, amps)
        
        # Add noise
        if self.noise!=0:
            noise_img = local_rng.normal(0, self.noise, (self.image_size, self.image_size))
            sample = np.add(sample, noise_img)
            
        # Unpad
        pad_sample = sample
        if self.pad != 0:
            sample = sample[self.pad:-self.pad,self.pad:-self.pad]
            
        # For multiprocessing tracking
        if queue is not None:
            queue.put(1)
        
        return sample, centers, count, amps
    
    def realize(self, batch=1000, temp_root_path="temp", mode='single', progress_bar=True):
        """Realize all samples by batches"""
        # Generate in batches, generation slows significantly when entire dataset is generated in one go, probably taking too much ram       
        num_of_batches = int(np.ceil(self.sample_num / batch))
        remainder = self.sample_num % batch
        if remainder==0:
            batch_sizes = np.repeat(batch, num_of_batches)
        else:
            batch_sizes = np.append(np.repeat(batch, num_of_batches),[remainder])
        # Batch seeds
        seeds = chunks(self.seeds, batch)
        
        # Create temporary save path
        temp_path = f'{temp_root_path}/temp'
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        
        # Realize
        for i, (batch_size, batch_seeds) in enumerate(zip(batch_sizes, seeds)):
            # Reset sample list
            samples = []
            sample_counts = []
            sample_centers = []
            sample_amps = []
            
            # Realize samples using single core
            if mode=='single':
                for j in tqdm(range(batch_size), desc=f'batch {i+1}/{num_of_batches}'):
                    # Realize sample
                    sample, centers, count, amps = self.realize_sample(seed=batch_seeds[j])
                    
                    # Add sample to list
                    samples.append(sample)
                    
                    # Log
                    sample_counts.append(count)
                    sample_centers.append(centers)
                    sample_amps.append(amps)

            # Realize samples using multi core    
            elif mode=='multi':
                with Manager() as manager:
                    queue = manager.Queue()
                    
                    # Initialize the progress bar
                    if progress_bar:
                        pbar = tqdm(desc=f'batch {i+1}/{num_of_batches}', total=int(batch_size), unit="sample")

                    # Using ProcessPoolExecutor to run tasks in parallel
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        # Submit tasks to the executor, passing the queue for progress updates
                        futures = [executor.submit(self.realize_sample, seed, queue) for seed in batch_seeds]

                        # Start a thread or a separate process to handle progress bar updates in the main thread
                        if progress_bar:    
                            update_progress_bar(queue, int(batch_size), pbar)

                        # Wait for all tasks to finish and gather results
                        results = [future.result() for future in futures]
                    
                    for result in results:
                        # Save in temp
                        samples.append(result[0])
                        sample_centers.append(result[1])
                        sample_counts.append(result[2])
                        sample_amps.append(result[3])
            else:
                raise Exception("Processing mode not supported")
            
            # Save samples in temp
            np.savez(f'{temp_path}/temp{i}', 
                     samples=samples, 
                     centers=np.array(sample_centers, dtype=object), 
                     counts=sample_counts,
                     amps=np.array(sample_amps, dtype=object))
            
        # Combine batches into single dataset
        print('combining dataset...')
        self.samples, self.sample_centers, self.sample_counts, self.sample_amps = [], [], [], []
        for file in tqdm(sorted(os.listdir(temp_path))):
            with np.load(f'{temp_path}/{file}', allow_pickle=True) as batch:
                self.samples.extend(batch['samples'])
                self.sample_centers.extend(list(batch['centers']))
                self.sample_counts.extend(batch['counts'])
                self.sample_amps.extend(list(batch['amps']))
            
        # Delete temporary folder
        shutil.rmtree(temp_path)
        print('temporary files deleted')
    
    def save(self, path):
        """Save data"""
        np.save(f'{path}/{self.file_name}', self.samples)
        np.save(f'{path}/{self.file_name}_counts', self.sample_counts)
        np.save(f'{path}/{self.file_name}_coords', np.array(self.sample_centers, dtype=object))
    
    def plot_example(self, sample, path, file_type='png'):
        """Plot first sample"""
        plt.title(f'num of blobs: {self.blob_num} | image size: {self.image_size} | blob size: {self.blob_size}')
        plt.imshow(sample)
        plt.colorbar()
        plt.savefig(f'{path}/sample_{self.file_name}.{file_type}')
        plt.close()
    
    def plot_count_distribution(self, path, file_type='png'):
        """Plot distribution"""
        bins = np.arange(np.min(self.sample_counts)-1, np.max(self.sample_counts)+1, 1)
        
        fig = plt.figure()
        fig.suptitle(f'Histogram of blob counts in the dataset')
        plt.hist(self.sample_counts, bins=bins)
        plt.xlabel('blob counts')
        plt.ylabel('image counts')
        plt.savefig(f'{path}/count_distr_{self.file_name}.{file_type}')
        plt.close()
        
    def plot_amplitude_distribution(self, path, file_type='png'):
        """Plot distribution"""
        amps = np.concatenate(self.sample_amps)
        bins = np.arange(np.min(amps)-1, np.max(amps)+1, 0.5)
        
        fig = plt.figure()
        fig.suptitle(f'Histogram of all blob amplitudes in the dataset')
        plt.hist(amps, bins=bins)
        plt.xlabel('blob amplitudes')
        plt.ylabel('image counts')
        plt.savefig(f'{path}/amp_distr_{self.file_name}.{file_type}')
        plt.close()
        
    def _normalize_2d(self, matrix):
        """Normalize to [0, 1]"""
        return (matrix-np.min(matrix))/(np.max(matrix)-np.min(matrix)) 


class centerGenerator():
    """Generates 2D center coordinates for blobs, supports clustering"""
    def __init__(self, num_centers, image_size):
        self.num_centers = num_centers
        self.image_size = image_size

    def unclustered_centers(self, num_distribution, rng=np.random.default_rng()):
        """Generate random coordinates"""
        if num_distribution=='delta':
            # Generate center coordinates
            centers = rng.random(size=(self.num_centers,2))*self.image_size - 0.5
        elif num_distribution=='poisson':
            # Get number of blobs from poisson distribution
            current_center_num = rng.poisson(self.num_centers)
            # Generate center coordinates
            if current_center_num==0:
                centers = np.empty((0,2))
            else:
                centers = rng.random(size=(current_center_num,2))*self.image_size - 0.5
         
        return centers
    
    def random_with_min_dist(self, n, min_dist, size=(1,2), rng=np.random.default_rng()):
        centers = []
        counter = 0
        while True:
            # Generate new center coordinate
            new_center = rng.random(size=size)*self.image_size - 0.5
            # Calculate distances to all other centers
            dist = np.array([np.linalg.norm(new_center - center) for center in centers])
            # Add blob if it is at the minimum distance from all other centers
            if (dist<min_dist).sum()>0:
                pass
            else:
                centers.append(*new_center)
            # Iterate counter
            counter += 1
            # Break when desired number of coordinates generated
            if len(centers)==n:
                break
            # Break if unable to find coordinate with minimum distance to all other points 
            if counter>(n*50):
                print("Error - unable to find coordiante that satisfies minimum distance constraint")
                return None 
        return np.array(centers)
    
    def non_overlapping_centers(self, num_distribution, min_dist, rng=np.random.default_rng()):
        """Generate random coordinates with minimum distance"""
        if num_distribution=='delta':
            # Generate center coordinates
            centers = self.random_with_min_dist(self.num_centers, min_dist, rng=rng)
        elif num_distribution=='poisson':
            # Get number of blobs from poisson distribution
            current_center_num = rng.poisson(self.num_centers)
            # Generate center coordinates
            if current_center_num==0:
                centers = np.empty((0,2))
            else:
                centers = self.random_with_min_dist(current_center_num, min_dist, rng=rng)
        return centers
    
    def clustered_centers(self, num_distribution, clustering, rng=np.random.default_rng(), track_progress=False):
        """Generate clustered coordinates according to a power-law 2-point correlation function"""
        # Survey configuration, assume all samples are 1' by 1' patches
        lxdeg = 1                    # Length of x-dimension [deg]
        lydeg = 1                    # Length of y-dimension [deg]
        nx = 1000                    # Grid size of x-dimension
        ny = 1000                    # Grid size of y-dimension
        # Input correlation function w(theta) = wa*theta[deg]^(-wb)
        wa = clustering[0]
        wb = clustering[1]
        
        if track_progress:
            phi = wa**(1/wb)
            print((phi*u.radian).to(u.degree))
        
        # Initializations
        lxrad,lyrad = np.radians(lxdeg),np.radians(lydeg)

        def transpk2d(nx,ny,lx,ly,kmod,pkmod):
            """Transform the power spectrum P --> P' so that the lognormal
            realization of P' will be the same as a Gaussian realization of P"""
            area,nc = lx*ly,float(nx*ny)
            # Obtain 2D grid of k-modes
            kx = 2.*np.pi*np.fft.fftfreq(nx,d=lx/nx)
            ky = 2.*np.pi*np.fft.fftfreq(ny,d=ly/ny)[:int(ny/2+1)]
            kspec = np.sqrt(kx[:,np.newaxis]**2 + ky[np.newaxis,:]**2)
            # Create power spectrum array
            pkspec = np.interp(kspec,kmod,pkmod) # interpolate the pk onto a finer grid
            pkspec[0,0] = 0.
            pkspec = pkspec/area + 0.*1j # ? normalization
            # Inverse Fourier transform to the correlation function
            xigrid = nc*np.fft.irfftn(pkspec)
            # Transform the correlation function
            xigrid = np.log(1.+xigrid)
            # Fourier transform back to the power spectrum
            pkspec = np.real(np.fft.rfftn(xigrid))
            return pkspec

        def gendens2d(nx,ny,lx,ly,pkspec,wingrid):
            """Generate a 2D log-normal density field of a Gaussian power spectrum"""
            # Generate complex Fourier amplitudes
            nc = float(nx*ny)
            ktot = pkspec.size
            pkspec[pkspec < 0.] = 0.
            rangauss = np.reshape(rng.normal(0.,1.,ktot),(nx,int(ny/2+1)))
            realpart = np.sqrt(pkspec/(2.*nc))*rangauss
            rangauss = np.reshape(rng.normal(0.,1.,ktot),(nx,int(ny/2+1)))
            imagpart = np.sqrt(pkspec/(2.*nc))*rangauss
            deltak = realpart + imagpart*1j
            # Make sure complex conjugate properties are in place
            doconj2d(nx,ny,deltak)
            # Do Fourier transform to produce overdensity field
            deltax = nc*np.fft.irfftn(deltak)
            # Produce density field
            lmean = np.exp(0.5*np.var(deltax))
            # print(deltax.shape, wingrid.shape)
            meangrid = wingrid*np.exp(deltax)/lmean
            return meangrid

        def doconj2d(nx,ny,deltak):
            """Impose complex conjugate properties on Fourier amplitudes"""
            for ix in range(int(nx/2+1),nx):
                deltak[ix,0] = np.conj(deltak[nx-ix,0])
                deltak[ix,int(ny/2)] = np.conj(deltak[nx-ix,int(ny/2)])
                deltak[0,0] = 0. + 0.*1j
                deltak[int(nx/2),0] = np.real(deltak[int(nx/2),0]) + 0.*1j
                deltak[0,int(ny/2)] = np.real(deltak[0,int(ny/2)]) + 0.*1j
                deltak[int(nx/2),int(ny/2)] = np.real(deltak[int(nx/2),int(ny/2)]) + 0.*1j
            return

        def genpos2d(nx,ny,lx,ly,datgrid):
            """Convert 2D number grid to positions"""
            # Create grid of x and y positions
            dx,dy = lx/nx,ly/ny
            x,y = dx*np.arange(nx),dy*np.arange(ny)

            xgrid,ygrid = np.meshgrid(x,y,indexing='ij')

            # Get coordinates where grid has points
            datgrid1,xgrid,ygrid = datgrid[datgrid > 0.].astype(int),xgrid[datgrid > 0.],ygrid[datgrid > 0.] 
            
            xgrid = np.repeat(xgrid, datgrid1)
            ygrid = np.repeat(ygrid, datgrid1)

            # Jitter
            xpos = xgrid + rng.uniform(0.,dx, size=len(xgrid)) 
            ypos = ygrid + rng.uniform(0.,dy, size=len(ygrid)) 
            
            return xpos, ypos
        
        # Convert to angular power spectrum
        ca = 2*np.pi*((np.pi/180.)**wb)*wa*(2.**(1.-wb))*gamma(1.-wb/2.)/gamma(wb/2.)
        cb = 2.-wb
        
        if track_progress:
            print('Clustering function:')
            print('w(theta) =',wa,'theta[deg]^(-',wb,')')
            print('C_K =',ca,'K[rad]^(-',cb,')')
        
        # Model angular power spectrum
        kminmod = min(2.*np.pi/lxrad,2.*np.pi/lyrad)
        kmaxmod = np.sqrt((np.pi*nx/lxrad)**2+(np.pi*ny/lyrad)**2) # in x and y direction
        nkmod = 1000
        kmod = np.linspace(kminmod,kmaxmod,nkmod)
        pkmod = ca*(kmod**(-cb))
        
        # Transform the power spectrum P --> P' so that the lognormal
        # realization of P' will be the same as a Gaussian realization of P
        pkin = transpk2d(nx,ny,lxrad,lyrad,kmod,pkmod)
        
        # Generate a log-normal density field of a Gaussian power spectrum
        wingrid = np.ones((nx,ny))
        wingrid /= np.sum(wingrid)
        meangrid = gendens2d(nx,ny,lxrad,lyrad,pkin,wingrid)
        
        # Sample number grid
        if num_distribution=='delta':
            # Ravel
            mean_ravel = meangrid.ravel()/float(meangrid.sum())

            # Sample coordinates using density field as probability
            idxs = rng.choice(np.arange(len(mean_ravel)), size=self.num_centers, p=mean_ravel).astype(int)
            idxs, counts = np.unique(idxs, return_counts=True)
            
            # Place coordinates on flattened grid 
            dat_list = np.zeros(len(mean_ravel), dtype=int)
            np.put(dat_list, idxs, counts)
            
            # Unravel
            datgrid = np.reshape(dat_list, meangrid.shape)
        
        elif num_distribution=='poisson':
            # Sample each point using Poisson
            meangrid *= float(self.num_centers)
            datgrid = rng.poisson(meangrid).astype(float)
        
        # Convert 2D number grid to positions
        xpos,ypos = genpos2d(nx ,ny,lxrad,lyrad,datgrid)
        
        # Convert positions to degrees
        centers = np.array(list(zip(
            np.degrees(xpos)*self.image_size/lxdeg,
            np.degrees(ypos)*self.image_size/lydeg,
            ))
        )
        
        return centers
    
    def generate(self, num_distribution, clustering, min_dist=None, rng=np.random.default_rng()):
        """Generate coordinates for one sample"""
        if min_dist is not None:
            centers = self.non_overlapping_centers(num_distribution, min_dist, rng=rng)
        elif clustering is None:
            centers = self.unclustered_centers(num_distribution, rng=rng)
        else:
            centers = self.clustered_centers(num_distribution, clustering, rng=rng)
        return centers



"""Depreciated"""

def normalize_2d(matrix): 
    return (matrix-np.min(matrix))/(np.max(matrix)-np.min(matrix)) 

def create_blob_sample(pos, generation_matrix_size, blob_num, blob_size, blob_amplitude=1):
    for j in range(blob_num):
        # Random coordinate for blob
        mean_coords = [random.randint(0, generation_matrix_size-1), random.randint(0, generation_matrix_size-1)]
        if j==0:
            # Add first blob to image
            sample = multivariate_normal(mean_coords, [[blob_size, 0], [0, blob_size]]).pdf(pos)
            # Normalize
            sample = normalize_2d(sample)*blob_amplitude
        
        if j!=0:
            # Add subsequent single blob to image
            sample_next = multivariate_normal(mean_coords, [[blob_size, 0], [0, blob_size]]).pdf(pos)
            # Normalize
            sample_next = normalize_2d(sample_next)*blob_amplitude
            sample = np.add(sample, sample_next)
    
    if blob_num==0:
        return np.zeros((generation_matrix_size, generation_matrix_size))
    else:
        return sample
    
def create_point_sample(point_num, image_size):
    coords = np.random.randint(28, size=(point_num, 2))
    sample = np.zeros((image_size, image_size))
    
    for coord in coords:
        sample[coord[0], coord[1]] += 1
        
    return sample