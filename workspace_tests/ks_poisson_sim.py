"""
Author: Nathan Teo

This script runs poisson simulations to get proper p-values for the discrete ks test
"""

n = 10_000
mean = 10
sample_size = 5_000
seed = 32

ks_stat_obs = 0.1331

##################################################################################################

# Import
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os

# Paths
project_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo'
save_path = f'{project_path}/misc_plots/ks_poisson_sim/mean-{mean}_samples-{sample_size}_sims-{n}'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
# RNG
rng = np.random.default_rng(seed=seed)

# Simulate
print('simulating...', end='\t\t')
sims = rng.poisson(10, size=(n, sample_size))
print('complete')

# Range
print('finding range...', end='\t')
lower, upper = np.min(sims), np.max(sims)
x = np.arange(lower, upper, 1)
print('complete')

# Calculate ks statistics
if 'simulation.npy' not in os.listdir(save_path):
    ks_stats = np.array([stats.kstest(sim, stats.poisson(mean).cdf).statistic
            for sim in tqdm(sims, desc='calculating statistics')])
    np.save(f'{save_path}/simulation.npy', ks_stats)
else:
    print('loading simulations...', end='\t')
    ks_stats = np.load(f'{save_path}/simulation.npy', allow_pickle=True)
    print('complete')

# Check
bins = np.arange(np.min(sims[0])-2, np.max(sims[0])+2, 1)
plt.hist(sims[0], bins=bins, histtype='step')
plt.title('Distribution of Poisson sample')
plt.xlabel('random variable')
plt.ylabel('count')
plt.savefig(f'{save_path}/rv_histogram')
plt.close()

plt.step(x, stats.poisson.cdf(x, mu=mean), where='post', color=('black', 0.5))
plt.ecdf(sims[0], color=('r', 0.5), linestyle='--')
plt.title('CDF of Poisson sample')
plt.xlabel('random variable')
plt.ylabel('probability')
plt.savefig(f'{save_path}/rv_cdf')
plt.close()

# Plot
plt.hist(ks_stats, histtype='step', bins=20)
plt.title('Distribution of simulated KS statistic')
plt.xlabel('KS statistic')
plt.ylabel('count')
plt.savefig(f'{save_path}/stat_histogram')
plt.close()

if ks_stat_obs is not None:
    p_val = np.where(ks_stats>=ks_stat_obs, 1, 0).mean()
    
    plt.hist(ks_stats, histtype='step', bins=20)
    plt.axvline(ks_stat_obs, color=('r',0.5), linestyle='dashed', linewidth=1)
    plt.title(f'observed KS statisic: {ks_stat_obs} | p-value: {p_val}')
    plt.xlabel('KS statistic')
    plt.ylabel('count')
    plt.savefig(f'{save_path}/stat_{str(ks_stat_obs).replace(".", "-")}')
    plt.close()
    
    print(f'ks-statistic: {ks_stat_obs} | p-value: {p_val}')
    
print('test complete')
