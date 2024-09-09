import random
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

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
        self.pad = params['pad']
        self.noise = params['noise']
        
        if self.pad == 'auto':
            self.pad = self.blob_size 
        self.generation_matrix_size = self.image_size + self.pad*2
        self.blob_amplitude = 1/self.blob_num
        
        self.file_name = f'bn{self.blob_num}{self.num_distribution[0]}-is{self.image_size}-bs{self.blob_size}-sn{self.sample_num}-sd{seed}-ns{self.noise}'
        
        random.seed(seed)
        
    def make_gaussian(self, center, var, image_size):
            """ Make a square gaussian kernel"""
            x = np.arange(0, image_size, 1, float)
            y = x[:,np.newaxis]

            x0 = center[1]
            y0 = center[0]

            return np.exp(-0.5 * ((x-x0)**2 + (y-y0)**2) / var)

    def create_blobs(self, centers):
        """Create an image of gaussian blobs"""
        return np.array([normalize_2d(self.make_gaussian(coord, self.blob_size, self.generation_matrix_size))*self.blob_amplitude
                            for coord in centers]).sum(axis=0)
        
    def generate(self):
        """Generate all samples"""
        self.samples = []
        self.sample_counts = []
        for i in tqdm(range(self.sample_num)):
            if self.num_distribution=='uniform':
                # Create sample
                centers = np.random.rand(self.blob_num,2)*self.generation_matrix_size
                sample = self.create_blobs(centers)
            elif self.num_distribution=='poisson':
                # Get number of blobs from poisson distribution
                current_blob_num = np.random.poisson(self.blob_num)
                self.sample_counts.append(current_blob_num)
                
                # Create sample
                if current_blob_num==0:
                    sample = np.zeros((28,28))
                else:
                    centers = np.random.rand(current_blob_num,2)*self.generation_matrix_size
                    sample = self.create_blobs(centers)
                    
            # Add noise
            if self.noise!=0:
                noise_img = np.random.normal(0, self.noise, (self.image_size, self.image_size))
                sample = np.add(sample, noise_img)
                
            # Unpad
            pad_sample = sample
            if self.pad != 0:
                sample = sample[self.pad:-self.pad,self.pad:-self.pad]
            
            # Add sample to list
            self.samples.append(sample)
   
        
    def save(self, path):
        """Save data"""
        np.save(f'{path}/{self.file_name}', self.samples)
        if self.num_distribution!='uniform':
            np.save(f'{path}/{self.file_name}_counts', self.sample_counts)
    
    def plot_example(self, sample, path, file_type='png'):
        """Plot first sample"""
        plt.title(f'num of blobs: {self.blob_num} | image size: {self.image_size} | blob size: {self.blob_size}')
        plt.imshow(sample)
        plt.colorbar()
        plt.savefig(f'{path}/sample_{self.file_name}.{file_type}')
        plt.close()
    
    def plot_distribution(self, path, file_type='png'):
        """Plot distribution"""
        bins = np.arange(np.min(self.sample_counts)-1, np.max(self.sample_counts)+1, 1)
        
        fig = plt.figure()
        fig.suptitle(f'num of blobs: {self.blob_num} | image size: {self.image_size} | blob size: {self.blob_size}')
        plt.hist(self.sample_counts, bins=bins)
        plt.xlabel('blob counts')
        plt.ylabel('image counts')
        plt.savefig(f'{path}/distr_{self.file_name}.{file_type}')
        plt.close()
        
    def _normalize_2d(self, matrix): 
        return (matrix-np.min(matrix))/(np.max(matrix)-np.min(matrix)) 

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
        return np.zeros((28,28))
    else:
        return sample
    
def create_point_sample(point_num, image_size):
    coords = np.random.randint(28, size=(point_num, 2))
    sample = np.zeros((image_size, image_size))
    
    for coord in coords:
        sample[coord[0], coord[1]] += 1
        
    return sample