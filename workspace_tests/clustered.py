import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import astropy.units as u
import time
from astroML.correlation import two_point, bootstrap_two_point, bootstrap_two_point_angular

def generate_cluster_centers(num_of_centers, image_size, clustering, num_distribution='uniform', track_progress=False):
    # Survey configuration
    lxdeg = 1               # Length of x-dimension [deg]
    lydeg = 1               # Length of y-dimension [deg]
    nx = 1200               # Grid size of x-dimension
    ny = 1200               # Grid size of y-dimension
    # Input correlation function w(theta) = wa*theta[deg]^(-wb)
    wa = clustering[0]
    wb = clustering[1]
    
    if track_progress:
        phi = wa**(1/wb)
        print((phi*u.radian).to(u.degree))
    
    # Initializations
    lxrad,lyrad = np.radians(lxdeg),np.radians(lydeg)
    
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
    if num_distribution=='uniform':
        # Ravel
        mean_ravel = meangrid.ravel()/float(meangrid.sum())

        # Sample coordinates using density field as probability
        idxs = np.random.choice(np.arange(len(mean_ravel)), size=num_of_centers, p=mean_ravel).astype(int)
        idxs, counts = np.unique(idxs, return_counts=True)
        
        # Place coordinates on flattened grid 
        dat_list = np.zeros(len(mean_ravel), dtype=int)
        np.put(dat_list, idxs, counts)
        
        # Unravel
        datgrid = np.reshape(dat_list, meangrid.shape)
    
    elif num_distribution=='poisson':
        # Sample each point using Poisson
        meangrid *= float(num_of_centers)
        datgrid = np.random.poisson(meangrid).astype(float)
    
    # Convert 2D number grid to positions
    xpos,ypos = genpos2d(nx ,ny,lxrad,lyrad,datgrid)
    
    # Convert positions to degrees
    xpos,ypos = np.degrees(xpos),np.degrees(ypos)
    
    # print(len(xpos),'galaxies generated')
    return xpos, ypos

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

def midpoints_of_bins(edges):
    """Returns midpoints of bin edges for plotting"""
    return (edges[:-1]+edges[1:])/2 

if __name__ == '__main__':
    num_samples = 1
    num_centers = 1e4
    image_size = 1200
    clustering = (0.5, 0.8)
    num_distribution = 'poisson'
    

    np.random.seed(50) 
    
    # num_blobs = []
    # for _ in tqdm(range(num_samples)):    
    #     x, y = generate_cluster_centers(num_centers, image_size, clustering, num_distribution=num_distribution)
    #     coords = np.array(list(zip(x, y)))
    #     num_blobs.append(len(x))
    
    # fig, axs = plt.subplots(1,2, figsize=(12,5)) 
    
    # axs[0].scatter(x, y, marker='.')
    # axs[0].set_title(f'example sample, {len(x)} number of centers')
    # axs[0].set_aspect('equal', 'box')
        
    # axs[1].hist(num_blobs, bins=np.arange(min(num_blobs)-3.5,max(num_blobs)+3.5,1),histtype='step')
    # axs[1].set_title('distribution')
    # axs[1].set_xlabel('center counts')
    # axs[1].set_ylabel('sample counts')
    
    # fig.tight_layout()
    
    # plt.show()
    # plt.close()
    
    # x = np.linspace(0.005, 1, 40)
    
    # def func(x, wa, wb):
    #     return (wa*(x**(-wb)))

    # plt.plot(x, func(x, clustering[0], clustering[1]), label='input')

    # bins = np.linspace(0,1,20)
    # corr, err = bootstrap_two_point(coords, bins, method='landy-szalay')
    # plt.errorbar(midpoints_of_bins(bins), corr, yerr=err, fmt='.', label='generated')

    # corr = two_point(coords, bins, method='landy-szalay')
    # plt.errorbar(midpoints_of_bins(bins), corr, yerr=None, fmt='.', label='generated')

    # plt.title(f'2-point\nw(theta) = {clustering[0]}theta[deg]^(-{clustering[1]})\n{len(coords)} points generated')

    # plt.tight_layout()
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.legend()
    # plt.show()
    # plt.close()

    
    'Look at different clusterings'
    logscale = True
    clusterings = [(0.0001881, 0.8), (0.001, 0.8), (0.005, 0.8)]

    coords = []
    for clustering in clusterings:
        x, y = generate_cluster_centers(num_centers, image_size, clustering, num_distribution=num_distribution)
        coords.append(np.array(list(zip(x, y))))
     
    x = np.linspace(0.001, 1, 40)
    
    def func(x, wa, wb):
        return (wa*(x**(-wb)))

    colors = ['C0', 'C1', 'C2']
    for coord, clustering, color in tqdm(zip(coords, clusterings, colors)):
        plt.plot(x, func(x, clustering[0], clustering[1]), color=color, 
                 label=f'input: {clustering[0]}theta[deg]^-{clustering[1]}')

        if logscale:
            bins = 10**np.linspace(np.log10(0.001),np.log10(1),20)
        else:
            bins = np.linspace(0,1,20)

        corr, err, _ = bootstrap_two_point_angular(*zip(*coord), bins, method='landy-szalay')
        plt.errorbar(midpoints_of_bins(bins), corr, yerr=err, fmt='.', color=color, capsize=2, 
                     label=f'realization: {clustering[0]}theta[deg]^-{clustering[1]}')

    # Format
    if logscale:
        plt.xscale('log')
        plt.yscale('log')
    plt.title(f'2-point with different input functions, {num_centers} centers')
    plt.xlabel('theta (degrees)')
    plt.ylabel('2 point correlation')
    plt.legend()
    plt.tight_layout()
    
    plt.show()
    plt.close()

       