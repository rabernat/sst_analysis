from IPython.parallel import Client
rc = Client()
dview = rc[:] # use all engines

with dview.sync_imports():
    import numpy
    from satdatatools import ghrsst_L2P_dataset, aggregator

# set up the aggregator object on each engine
dview.execute('lla = aggregator.LatLonAggregator()')

# define a remote function
@dview.parallel(block=False)
def count_goodpts(fname):
    f = ghrsst_L2P_dataset.GHRSSTFile(fname)
    pc = f.proximity_confidence
    if isinstance(f, numpy.ma.masked_array):
        pc = pc.filled(0.)
    goodpts = (pc >= 4).astype('i4')
    return lla.binsum(goodpts, f.lon, f.lat)

# define a reduce function
def verbose_reduce_sum(A,B):
    print A.sum()
    return A + B

# new do the look
ddir = '/Volumes/Bucket1/Data/ghrsst/data/L2P/MODIS_A/JPL'
lla = aggregator.LatLonAggregator()
#count = lla.zeros(np.dtype('i8'))

gc = ghrsst_L2P_dataset.GHRSSTCollection(ddir)
#count = reduce( verbose_reduce_sum,
#        count_goodpts.map(gc.iterate()) )
filelist = []
while len(filelist) < 16:
    filelist.append(gc.iterate(yield_fname=True).next())

#count = reduce( verbose_reduce_sum,
#         count_goodpts.map(filelist) )

#np.savez('goodpts_count_par', count=count, lon=lla.lon, lat=lla.lat)

