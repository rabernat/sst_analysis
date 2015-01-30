import numpy as np
import os
from matplotlib import pyplot as plt
%matplotlib inline
from Idealized_Model import idealmodel

# load data files
base_dir = 'Idealized_Model/Model_output/Eady/Eady/'
fname = os.path.join(base_dir,'b00.mat')
p = idealmodel.IDEALFile(fname)

# 2D spectra
nbins, Nx, Ny, k, l, PSD_2d, Ki, iso_spec, area = p.power_spectrum_2d()

# Structure Function
N, Struc_funci, Struc_funcj = p.structure_function()
