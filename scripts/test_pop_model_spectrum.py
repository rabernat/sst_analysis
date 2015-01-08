from sstanalysis import popanalysis
reload(popanalysis)

base_dir = '/Volumes/Bucket1/Data/hybrid_v5_rel04_BC5_ne120_t12_pop62/'
fname = base_dir + 'hybrid_v5_rel04_BC5_ne120_t12_pop62.pop.h.nday1.0046-02-01.nc'

p = popanalysis.POPFile(fname)

# load variable
T = p.nc.variables['SST']

# compute variables needed for FFT
Nt, Nx, Ny, k, l, PSD_sum = p.power_spectrum_2d(T)

# take time mean
PSD_2d = PSD_sum/Nt

# reduce dimensions
PSDx = PSD_2d.mean(axis=0)/Nx
PSDy = PSD_2d.mean(axis=1)/Ny
