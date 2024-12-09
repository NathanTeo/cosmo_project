"""
Author: Nathan Teo

This script runs poisson simulations to get proper p-values for the discrete ks test
"""

n = 100_000
mean = 10
sample_size = 5_000
seed = 32

ks_stat_obs = 0.015

restart = False

##################################################################################################

# Import
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm.auto import tqdm
import os

# Paths
project_path = 'C:/Users/Idiot/Desktop/Research/OFYP/cosmo'
save_path = f'{project_path}/misc_plots/ks_poisson_sim/mean-{mean}_samples-{sample_size}_sims-{n}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def ecdf(a):
    """Return bins and ecdf"""
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]

def ks_poisson(rvs, mean):
    """KS test for poisson"""
    x, counts = ecdf(rvs)
    poisson = stats.poisson.cdf(x, mu=mean)
    diff = np.abs(poisson-counts)
    return np.max(diff)

# Run simulations
if 'simulation.npy' not in os.listdir(save_path) or restart:
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
    
    # Calculate KS stats
    ks_stats = np.array([ks_poisson(sim, mean)
            for sim in tqdm(sims, desc='calculating statistics')])
    np.save(f'{save_path}/simulation.npy', ks_stats)
    
    # Check
    bins = np.arange(np.min(sims[0])-2, np.max(sims[0])+2, 1)
    plt.hist(sims[0], bins=bins, histtype='step')
    plt.title('Distribution of Poisson sample')
    plt.xlabel('random variable')
    plt.ylabel('count')
    plt.savefig(f'{save_path}/rv_histogram')
    plt.close()

    plt.step(x, stats.poisson.cdf(x, mu=mean), where='post', color=('black', 0.5), label='theory')
    plt.ecdf(sims[0], color=('r', 0.5), linestyle='--', label='sim')
    plt.title(f'CDF of Poisson sample, KS stat: {ks_stats[0]}')
    plt.xlabel('random variable')
    plt.ylabel('probability')
    plt.legend()
    plt.savefig(f'{save_path}/rv_cdf')
    plt.close()
    
else:
    print('loading simulations...', end='\t')
    ks_stats = np.load(f'{save_path}/simulation.npy', allow_pickle=True)
    print('complete')


# Crit statistic
ks_stats = np.sort(ks_stats)
crit_stat = np.percentile(ks_stats, 95)

'Plot'
fig = plt.figure(figsize=(4,3))

# Histogram
bins = np.linspace(0,0.03,40)
hist, _, _ = plt.hist(ks_stats, histtype='step', bins=bins, color=('black',.9), linewidth=1.1)

# Color crit region
cdf = np.cumsum(hist)
cdf = cdf / cdf[-1]
bin_idx = np.argmax(cdf >= 0.95)
tail_bins = bins[bin_idx:-1]
tail_bins[0] = crit_stat
plt.fill_between(tail_bins, hist[bin_idx:], color=('red', 0.2), step='post', label='p-value of 0.05')

# Crit value
plt.axvline(crit_stat, color=('red',0.5), linestyle='dashed', linewidth=1.2)
_, max_ylim = plt.ylim()
plt.text(crit_stat*1.05, max_ylim*0.4, 
         f'critical statistic\n{crit_stat:.4f}', color='red')

plt.title('Distribution of simulated KS statistic')
ax = plt.gca()
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis='y', style='scientific', scilimits=(3, 3))
plt.xlabel('KS statistic')
plt.ylabel('count')
plt.tight_layout()
plt.legend()
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
