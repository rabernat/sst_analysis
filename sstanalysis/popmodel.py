import numpy as np
import netCDF4
from warnings import warn

class POPFile(object):
    
    def __init__(self, fname, ah=-3e17, is3d=False):
        """Wrapper for POP model netCDF files"""
        self.nc = netCDF4.Dataset(fname)
        self.Ny, self.Nx = self.nc.variables['TAREA'].shape     
        self._ah = ah
        
        # mask
        self.mask = self.nc.variables['KMT'][:] <= 1

        self.is3d = is3d
        if self.is3d:
            self.z_t = nc.variables['z_t'][:]
            self.z_w_top = nc.variables['z_w_top'][:]
            self.z_w_bot = nc.variables['z_w_bop'][:]
            self.Nz = len(self.z_t)
            kmt = p.nc.variables['KMT'][:]
            self.mask3d = np.zeros((self.Nz, self.Ny, self.Nx), dtype='b')
            Nz = mask3d.shape[0]
            for k in range(Nz):
                self.mask3d[k] = (kmt<=k)

    def mask_field(self, T):
        """Apply mask to tracer field T"""
        return np.ma.masked_array(T, self.mask)
        
    def initialize_gradient_operator(self):
        """Needs to be called before calculating gradients"""
        # raw grid geometry
        work1 = (self.nc.variables['HTN'][:] /
                 self.nc.variables['HUW'][:])
        tarea = self.nc.variables['TAREA'][:]
        self.tarea = tarea
        tarea_r = np.ma.masked_invalid(tarea**-1).filled(0.)
        dtn = work1*tarea_r
        dts = np.roll(work1,-1,axis=0)*tarea_r
        
        work1 = (self.nc.variables['HTE'][:] /
                 self.nc.variables['HUS'][:])
        dte = work1*tarea_r
        dtw = np.roll(work1,-1,axis=1)*tarea_r
        
        # boundary conditions
        kmt = self.nc.variables['KMT'][:] > 1
        kmtn = np.roll(kmt,-1,axis=0)
        kmts = np.roll(kmt,1,axis=0)
        kmte = np.roll(kmt,-1,axis=1)
        kmtw = np.roll(kmt,1,axis=1)
        self._cn = np.where( kmt & kmtn, dtn, 0.)
        self._cs = np.where( kmt & kmts, dts, 0.)
        self._ce = np.where( kmt & kmte, dte, 0.)
        self._cw = np.where( kmt & kmtw, dtw, 0.)
        self._cc = -(self._cn + self._cs + self._ce + self._cw)
        
        # mixing coefficients
        #self._ah = -0.2e20*(1280.0/self.Nx)
        j_eq = np.argmin(self.nc.variables['ULAT'][:,0]**2)
        self._ahf = (tarea / self.nc.variables['UAREA'][j_eq,0])**1.5
        self._ahf[self.mask] = 0.   
        
        # stuff for gradient
        # reciprocal of dx and dy (in meters)
        self._dxtr = 100.*self.nc.variables['DXT'][:]**-1
        self._dytr = 100.*self.nc.variables['DYT'][:]**-1
        self._kmaske = np.where(kmt & kmte, 1., 0.)
        self._kmaskn = np.where(kmt & kmtn, 1., 0.)
        
        self._dxu = self.nc.variables['DXU'][:]
        self._dyu = self.nc.variables['DYU'][:]
                
    def laplacian(self, T):
        """Returns the laplacian of T at the tracer point."""
        return (
            self._cc*T +
            self._cn*np.roll(T,-1,axis=0) +
            self._cs*np.roll(T,1,axis=0) +
            self._ce*np.roll(T,-1,axis=1) +
            self._cw*np.roll(T,1,axis=1)          
        )
        
    def gradient_modulus(self, T):
        """Return the modulus of the gradient of T at the tracer point."""
        dTx = self._kmaske * (np.roll(T,-1,axis=0) - T)
        dTy = self._kmaskn * (np.roll(T,-1,axis=1) - T)
        
        return np.sqrt( 0.5 *
                    (dTx**2 + np.roll(dTx,1,axis=0)**2) * self._dxtr**2
                   +(dTy**2 + np.roll(dTy,1,axis=1)**2) * self._dytr**2
        )        
        
    def power_spectrum_2d(self, varname='SST', lonrange=(180,200), latrange=(30,50)):
        """Calculate a two-dimensional power spectrum of netcdf variable 'varname'
           in the box defined by lonrange and latrange.
        """
    
        lon = self.nc.variables['TLONG'][:]
        lat = self.nc.variables['TLAT'][:]
        
        # step 1: figure out the box indices
        lonmask = (lon >= lonrange[0]) & (lon < lonrange[1])
        latmask = (lat >= latrange[0]) & (lat < latrange[1])
        boxidx = lonmask & latmask # this won't necessarily be square
        irange = np.where(boxidx.sum(axis=0))[0]
        imin, imax = irange.min(), irange.max()
        jrange = np.where(boxidx.sum(axis=1))[0]
        jmin, jmax = jrange.min(), jrange.max()
        Nx = imax - imin
        Ny = jmax - jmin

        # step 2: load the data
        T = self.nc.variables[varname][..., jmin:jmax, imin:imax]
        
        # step 3: figure out if there is too much land in the box
        MAX_LAND = 0.01 # only allow up to 1% of land
        region_mask = self.mask[jmin:jmax, imin:imax]
        land_fraction = region_mask.sum().astype('f8') / (Ny*Nx)
        if land_fraction==0.:
            # no problem
            pass
        elif land_fraction >= MAX_LAND:
            raise ValueError('The sector has too much land. land_fraction = ' + str(land_fraction))
        else:    
            # do some interpolation
            errstr = 'The sector has land (land_fraction=%g) but we are interpolating it out.' % land_fraction
            warn(errstr)
            # have to figure out how to actually do it
            # Ti = ...
        
        # step 4: figure out FFT parameters (k, l, etc.) and set up result variable

        ##########################################
        ### Start looping through each time step #
        ##########################################
        Nt = T.shape[0]
        for n in range(Nt):
            Ti = T[n]
            
            # step 5: interpolate the missing data (only of necessary)
        
            # step 6: detrend the data in two dimensions
        
            # step 7: window the data
        
            # step 8: do the FFT for each timestep and aggregate the results
        
        
        # step 9: return the results
            
        
        
        
        

