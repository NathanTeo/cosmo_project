"""
Author: Nathan Teo

This script looks at power spectrum outlier samples
"""

run = "diffusion_6b"

##############################################################################################
# Import
import numpy as np
import matplotlib.pyplot as plt
import sys
from utils.load_model import modelLoader

project_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo/cosmo_project'
sys.path.append(project_path)
from code_model.testers.eval_utils import *
from code_model.testers.plotting_utils import *
from code_data.utils import *

# Load samples
model = modelLoader(run)
model.load_samples()

real_samples = model.real_samples
model_samples = model.model_samples

# Get image size
image_size = model_samples[0].shape[0]

# Find power spectrum of all samples
ell2d = ell_coordinates(image_size, 1/image_size)
cls = []   
for sample in tqdm(model_samples, desc='power spectrum'):
    bins, cl = power_spectrum(sample, sample, 
                              delta_ell=500, ell_max=10000, ell2d=ell2d, 
                              image_size_angular=1, taper=True)
    plt.plot(bins, cl, color=('black', 0.1))
    cls.append(cl)
cls = np.array(cls)
plt.show()
plt.close()

# Get samples of first bin to look for outliers
bin1 = cls[:,0]

# Top k outliers
k = 10
outliers_idxs = np.argpartition(bin1, -k)[-k:]
print(outliers_idxs)
outliers = model_samples[outliers_idxs]
print(outliers.shape)

# Plot top k outliers
fig, axs = plt.subplots(int(np.ceil(k/5)), 5)
for i, outlier in enumerate(outliers):
    axs[int(np.floor(i/5)), i%5].imshow(outlier)
plt.show()
plt.close()

x = 10
outlier_count = np.sum(np.where(bin1>x*np.mean(bin1), 1, 0))
print(f"{outlier_count} samples {x}x mean, {outlier_count/len(model_samples)} of total")

outlier_idxs = np.argwhere(bin1>x*np.mean(bin1)).squeeze()
not_outliers = np.delete(cls, outlier_idxs, axis=0)
outliers = cls[outlier_idxs]

# print(outlier_idxs)
# print(cls.shape)
# print(outliers.shape)

for cl in not_outliers:
    plt.plot(bins, cl, color=('black', 0.1))
for cl in outliers:
    plt.plot(bins, cl, color=('r', 0.2))

plt.show()
plt.close()