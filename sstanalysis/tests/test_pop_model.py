from sstanalysis import popmodel
import numpy as np

ddir = '/Users/rpa/RND/Data/hybrid_v5_rel04_BC5_ne120_t12_pop62/'
fname = 'hybrid_v5_rel04_BC5_ne120_t12_pop62.pop.h.nday1.0046-01-01.nc'

p = popmodel.POPFile(ddir + fname)

# set up fields needed to calculate gradients
p.initialize_gradient_operator()

# load a tracer field
T = p.nc.variables['SST'][0]

# take gradient modulus
gradT = p.gradient_modulus(T)

# take spatial mean of gradient moduls
mean_gradT = p.mask_field(gradT * p.tarea).sum() / p.mask_field(p.tarea).sum()
np.testing.assert_almost_equal(mean_gradT, 1.71409599582e-05)

