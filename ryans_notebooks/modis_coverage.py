import numpy as np
from satdatatools import ghrsst_L2P_dataset as dset
from satdatatools import aggregator

ddir = '/Volumes/Bucket1/Data/ghrsst/data/L2P/MODIS_A/JPL'
lla = aggregator.LatLonAggregator()
count = lla.zeros(np.dtype('i8'))

gc = dset.GHRSSTCollection(ddir)
for f in gc.iterate():
    print f.fname
    pc = f.proximity_confidence
    if isinstance(f, np.ma.masked_array):
        pc = pc.filled(0.)
    goodpts = (pc >= 4).astype('i4')
    print goodpts.sum()
    count += lla.binsum(goodpts, f.lon, f.lat)

np.savez('goodpts_count', count=count, lon=lla.lon, lat=lla.lat)

