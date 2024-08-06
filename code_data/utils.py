import random
from scipy.stats import multivariate_normal
import numpy as np

def normalize_2d(matrix):
    return (matrix-np.min(matrix))/(np.max(matrix)-np.min(matrix)) 

def create_blob_sample(pos, blob_num, blob_size, generation_matrix_size):
    for j in range(blob_num):
        # Random coordinate for blob
        mean_coords = [random.randint(0, generation_matrix_size-1), random.randint(0, generation_matrix_size-1)]
        if j==0:
            # Add first blob to image
            sample = multivariate_normal(mean_coords, [[blob_size, 0], [0, blob_size]]).pdf(pos)
            # Normalize
            sample = normalize_2d(sample)
        
        if j!=0:
            # Add subsequent single blob to image
            sample_next = multivariate_normal(mean_coords, [[blob_size, 0], [0, blob_size]]).pdf(pos)
            # Normalize
            sample_next = normalize_2d(sample_next)
            sample = np.add(sample, sample_next)
    
    return sample

def create_point_sample(point_num, image_size):
    coords = np.random.randint(28, size=(point_num, 2))
    sample = np.zeros((image_size, image_size))
    
    for coord in coords:
        sample[coord[0], coord[1]] += 1
        
    return sample