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

def sample_power_law(a, b, p, size=1):
    """Sample x^{-p} for a<=x<=b"""
    r = np.random.random(size=size)*(b-a) + a
    print(r)
    return r**(-p)

class blobDataset():
    """generate real data for training"""
    def __init__(self, **params):
        """Initialize parameters"""
        self.image_size = params['image_size']
        seed = params['seed']
        self.blob_size = params['blob_size']
        self.sample_num = params['sample_num']
        self.blob_num =  params['blob_num']
        self.num_distribution = params['num_distribution']
        self.blob_amplitude = params['blob_amplitude']
        self.amplitude_distribution = params['amplitude_distribution']
        self.clustering = params['clustering']
        self.pad = params['pad']
        self.noise = params['noise']
        
        if self.pad == 'auto':
            self.pad = self.blob_size 
        self.generation_matrix_size = self.image_size + self.pad*2
        
        self.file_name = 'bn{}{}-cl{}-is{}-bs{}-ba{}{}-sn{}-sd{}-ns{}'.format(
            self.blob_num, self.num_distribution[0], 
            '{:.0e}_{:.0e}'.format(*self.clustering) if self.clustering is not None else '_',
            self.image_size, self.blob_size, 
            '{:.0e}'.format(self.blob_amplitude), self.amplitude_distribution[0], 
            self.sample_num,
            seed, self.noise
        )
        
        # Set seed
        np.random.seed(seed)
  
        # Initiate center generator
        self.center_generator = centerGenerator(self.blob_num, self.image_size)
        
    def make_gaussian(self, center, var, image_size):
            """ Make a square gaussian kernel"""
            x = np.arange(0, image_size, 1, float)
            y = x[:,np.newaxis]

            x0 = center[1]
            y0 = center[0]

            return np.exp(-0.5 * ((x-x0)**2 + (y-y0)**2) / var)

    def create_blobs(self, centers, blob_amplitudes=None):
        """Create an image of gaussian blobs""" ### Add in blob_amplitudes, if fixed - send in array of same blob amps
        if centers.shape[0]>0:
            if blob_amplitudes is None:
                return np.array([normalize_2d(self.make_gaussian(coord, self.blob_size, self.generation_matrix_size))*self.blob_amplitude
                                for coord in centers]).sum(axis=0)
            else:
                return np.array([normalize_2d(self.make_gaussian(coord, self.blob_size, self.generation_matrix_size))*blob_amplitude
                    for coord, blob_amplitude in zip(centers, blob_amplitudes)]).sum(axis=0)
            
        else:
            return np.zeros((self.generation_matrix_size, self.generation_matrix_size))
    
    def realize_batch(self, dummy):
        # Generate points for centers
        centers = self.center_generator.generate(self.num_distribution, self.clustering)
        count = centers.shape[0]
        
        # Sample amplitudes
        if self.amplitude_distribution=='delta':
            amps = None
        elif self.amplitude_distribution=='power':
            amps = sample_power_law(0.1, 1, 1, count)*self.blob_amplitude # minimum of 1 maximum of 10
        
        # Create blobs
        sample = self.create_blobs(centers, amps)
        
        # Add noise
        if self.noise!=0:
            noise_img = np.random.normal(0, self.noise, (self.image_size, self.image_size))
            sample = np.add(sample, noise_img)
            
        # Unpad
        pad_sample = sample
        if self.pad != 0:
            sample = sample[self.pad:-self.pad,self.pad:-self.pad]
        
        return sample, centers, count, amps
    
    def realize(self, batch=1000, temp_root_path="temp", mode='single'):
        """Generate all samples"""
        # Generate in batches, generation slows significantly when entire dataset is generated in one go, probably taking too much ram       
        num_of_batches = int(np.ceil(self.sample_num / batch))
        remainder = self.sample_num % batch
        if remainder==0:
            batch_sizes = np.repeat(batch, num_of_batches)
        else:
            batch_sizes = np.append(np.repeat(batch, num_of_batches),[remainder])
        
        # Create temporary save path
        temp_path = f'{temp_root_path}/temp'
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        
        # Realize
        for i, batch_size in enumerate(batch_sizes):
            # Reset sample list
            samples = []
            sample_counts = []
            sample_centers = []
            sample_amps = []
            
            # Single core
            if mode=='single':
                for _ in tqdm(range(batch_size), desc=f'batch {i+1}/{num_of_batches}'):
                    # Realize sample
                    sample, centers, count, amps = self.realize_batch(self.center_generator)
                    
                    # Add sample to list
                    samples.append(sample)
                    
                    # Log
                    sample_counts.append(count)
                    sample_centers.append(centers)
                    sample_amps.append(amps)

            # Count using multi core    
            if mode=='multi':
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = list(tqdm(
                        executor.map(self.realize_batch, np.repeat(1,batch_size)),
                        desc=f'batch {i+1}/{num_of_batches}', total=int(batch_size)         
                        ))

                    for result in results:
                        # Save in temp
                        samples.append(result[0])
                        sample_centers.append(result[1])
                        sample_counts.append(result[2])
                        sample_amps.append(result[3])
            
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
        bins = np.linspace(np.min(self.sample_amps)-1, np.max(self.amps)+1, 40)
        
        fig = plt.figure()
        fig.suptitle(f'Histogram of all blob amplitudes in the dataset')
        plt.hist(np.concatenate(self.sample_amps), bins=bins)
        plt.xlabel('blob amplitudes')
        plt.ylabel('image counts')
        plt.savefig(f'{path}/amp_distr_{self.file_name}.{file_type}')
        plt.close()
        
    def _normalize_2d(self, matrix): 
        return (matrix-np.min(matrix))/(np.max(matrix)-np.min(matrix)) 


class centerGenerator():
    def __init__(self, num_centers, image_size):
        self.num_centers = num_centers
        self.image_size = image_size

    def unclustered_centers(self, num_distribution):
        if num_distribution=='delta':
            # Create sample
            self.centers = np.random.rand(self.num_centers,2)*self.image_size - 0.5
        elif num_distribution=='poisson':
            # Get number of blobs from poisson distribution
            current_center_num = np.random.poisson(self.num_centers)
            
            # Create sample
            if current_center_num==0:
                self.centers = np.empty((0,2))
            else:
                self.centers = np.random.rand(current_center_num,2)*self.image_size - 0.5
         
        return self.centers
    
    def clustered_centers(self, num_distribution, clustering, track_progress=False):
        # Survey configuration
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
                # Transform the power spectrum P --> P' so that the lognormal
        # realization of P' will be the same as a Gaussian realization of P
        def transpk2d(nx,ny,lx,ly,kmod,pkmod):
            # print('Transforming to input P(k)...')
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

        # Generate a 2D log-normal density field of a Gaussian power spectrum
        def gendens2d(nx,ny,lx,ly,pkspec,wingrid):
            # print('Generating lognormal density field...')
            # Generate complex Fourier amplitudes
            nc = float(nx*ny)
            ktot = pkspec.size
            pkspec[pkspec < 0.] = 0.
            rangauss = np.reshape(np.random.normal(0.,1.,ktot),(nx,int(ny/2+1)))
            realpart = np.sqrt(pkspec/(2.*nc))*rangauss
            rangauss = np.reshape(np.random.normal(0.,1.,ktot),(nx,int(ny/2+1)))
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

        # Impose complex conjugate properties on Fourier amplitudes
        def doconj2d(nx,ny,deltak):
            for ix in range(int(nx/2+1),nx):
                deltak[ix,0] = np.conj(deltak[nx-ix,0])
                deltak[ix,int(ny/2)] = np.conj(deltak[nx-ix,int(ny/2)])
                deltak[0,0] = 0. + 0.*1j
                deltak[int(nx/2),0] = np.real(deltak[int(nx/2),0]) + 0.*1j
                deltak[0,int(ny/2)] = np.real(deltak[0,int(ny/2)]) + 0.*1j
                deltak[int(nx/2),int(ny/2)] = np.real(deltak[int(nx/2),int(ny/2)]) + 0.*1j
            return

        # Convert 2D number grid to positions
        def genpos2d(nx,ny,lx,ly,datgrid):
            # print('Populating density field...')

            # Create grid of x and y positions
            dx,dy = lx/nx,ly/ny
            x,y = dx*np.arange(nx),dy*np.arange(ny)

            xgrid,ygrid = np.meshgrid(x,y,indexing='ij')

            # Get coordinates where grid has points
            datgrid1,xgrid,ygrid = datgrid[datgrid > 0.].astype(int),xgrid[datgrid > 0.],ygrid[datgrid > 0.] 
            
            xgrid = np.repeat(xgrid, datgrid1)
            ygrid = np.repeat(ygrid, datgrid1)

            # Jitter
            xpos = xgrid + np.random.uniform(0.,dx, size=len(xgrid)) 
            ypos = ygrid + np.random.uniform(0.,dy, size=len(ygrid)) 
            
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
            idxs = np.random.choice(np.arange(len(mean_ravel)), size=self.num_centers, p=mean_ravel).astype(int)
            idxs, counts = np.unique(idxs, return_counts=True)
            
            # Place coordinates on flattened grid 
            dat_list = np.zeros(len(mean_ravel), dtype=int)
            np.put(dat_list, idxs, counts)
            
            # Unravel
            datgrid = np.reshape(dat_list, meangrid.shape)
        
        elif num_distribution=='poisson':
            # Sample each point using Poisson
            meangrid *= float(self.num_centers)
            datgrid = np.random.poisson(meangrid).astype(float)
        
        # Convert 2D number grid to positions
        xpos,ypos = genpos2d(nx ,ny,lxrad,lyrad,datgrid)
        
        # Convert positions to degrees
        self.centers = np.array(list(zip(
            np.degrees(xpos)*self.image_size/lxdeg,
            np.degrees(ypos)*self.image_size/lydeg,
            ))
        )
        
        return self.centers
    
    def generate(self, num_distribution, clustering):
        if clustering is None:
            self.unclustered_centers(num_distribution)
        else:
            self.clustered_centers(num_distribution, clustering)
        
        return self.centers



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