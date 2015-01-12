from sstanalysis.popmodel import POPFile
import numpy as np

base_dir = '/Volumes/Bucket1/Data/hybrid_v5_rel04_BC5_ne120_t12_pop62/'
fname = 'hybrid_v5_rel04_BC5_ne120_t12_pop62.pop.h.nday1.0046-01-01.nc'

try:
    p =  POPFile(base_dir + fname)
except RuntimeError:
    base_dir = '/Users/rpa/RND/Data/hybrid_v5_rel04_BC5_ne120_t12_pop62/'
    print base_dir + fname
    p = POPFile(base_dir + fname)    
    
# compute variables needed for FFT
Nt, Nx, Ny, k, l, PSD_sum = p.power_spectrum_2d()

# take time mean
PSD_2d = PSD_sum/Nt

# reduce dimensions
PSDx = PSD_2d.mean(axis=0)/Nx
PSDy = PSD_2d.mean(axis=1)/Ny

np.testing.assert_almost_equal(PSDx.mean(), 206.088243177)
np.testing.assert_almost_equal(PSDy.mean(), 203.649329056)
