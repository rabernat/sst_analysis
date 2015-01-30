import numpy as np
import netCDF4
from warnings import warn
from scipy import linalg as lin
from scipy import signal as sig
from scipy import fftpack as fft
from scipy import interpolate as naiso
from scipy import io
import gfd

class IDEALFile(object):
    
    def __init__(self, fname):
        """Wrapper for POP model netCDF files"""
        self.mat = io.loadmat(fname)
        self.Ny, self.Nx = self.mat['bC'].shape     
        #self._ah = ah
        
        # mask
        #self.mask = self.nc.variables['KMT'][:] <= 1

        #self.is3d = is3d
        #if self.is3d:
        #    self.z_t = nc.variables['z_t'][:]
        #    self.z_w_top = nc.variables['z_w_top'][:]
        #    self.z_w_bot = nc.variables['z_w_bop'][:]
        #    self.Nz = len(self.z_t)
        #    kmt = p.nc.variables['KMT'][:]
        #    self.mask3d = np.zeros((self.Nz, self.Ny, self.Nx), dtype='b')
        #    Nz = mask3d.shape[0]
        #    for k in range(Nz):
        #        self.mask3d[k] = (kmt<=k)

    #def mask_field(self, T):
    #    """Apply mask to tracer field T"""
    #    return np.ma.masked_array(T, self.mask)
        
    #def initialize_gradient_operator(self):
    #    """Needs to be called before calculating gradients"""
    #    # raw grid geometry
    #    work1 = (self.nc.variables['HTN'][:] /
    #             self.nc.variables['HUW'][:])
    #    tarea = self.nc.variables['TAREA'][:]
    #    self.tarea = tarea
    #    tarea_r = np.ma.masked_invalid(tarea**-1).filled(0.)
    #    dtn = work1*tarea_r
    #    dts = np.roll(work1,-1,axis=0)*tarea_r
    #    
    #    work1 = (self.nc.variables['HTE'][:] /
    #             self.nc.variables['HUS'][:])
    #    dte = work1*tarea_r
    #    dtw = np.roll(work1,-1,axis=1)*tarea_r
    #    
    #    # boundary conditions
    #    kmt = self.nc.variables['KMT'][:] > 1
    #    kmtn = np.roll(kmt,-1,axis=0)
    #    kmts = np.roll(kmt,1,axis=0)
    #    kmte = np.roll(kmt,-1,axis=1)
    #    kmtw = np.roll(kmt,1,axis=1)
    #    self._cn = np.where( kmt & kmtn, dtn, 0.)
    #    self._cs = np.where( kmt & kmts, dts, 0.)
    #    self._ce = np.where( kmt & kmte, dte, 0.)
    #    self._cw = np.where( kmt & kmtw, dtw, 0.)
    #    self._cc = -(self._cn + self._cs + self._ce + self._cw)
    #    
    #    # mixing coefficients
    #    #self._ah = -0.2e20*(1280.0/self.Nx)
    #    j_eq = np.argmin(self.nc.variables['ULAT'][:,0]**2)
    #    self._ahf = (tarea / self.nc.variables['UAREA'][j_eq,0])**1.5
    #    self._ahf[self.mask] = 0.   
    #    
    #    # stuff for gradient
    #    # reciprocal of dx and dy (in meters)
    #    self._dxtr = 100.*self.nc.variables['DXT'][:]**-1
    #    self._dytr = 100.*self.nc.variables['DYT'][:]**-1
    #    self._kmaske = np.where(kmt & kmte, 1., 0.)
    #    self._kmaskn = np.where(kmt & kmtn, 1., 0.)
    #    
    #    self._dxu = self.nc.variables['DXU'][:]
    #    self._dyu = self.nc.variables['DYU'][:]
                
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
        
    def power_spectrum_2d(self, varname='bT', nbins=256, detre=False, windw=False):
        """Calculate a two-dimensional power spectrum of netcdf variable 'varname'
           in the box defined by lonrange and latrange.
        """
        
        # step 1: load the data
        #T = np.roll(self.nc.variables[varname],-1000)[..., jmin:jmax, imin:imax]
        T = self.mat[varname][:]
        Ny, Nx = T.shape
        dx = 1e3
        dy = 1e3

        # Wavenumber step
        k = fft.fftshift(fft.fftfreq(Nx, dx))

        l = fft.fftshift(fft.fftfreq(Ny, dy))

        ##########################################
        ### Start looping through each time step #
        ##########################################
        #Nt = T.shape[0]
        PSD_2d = np.zeros((Ny,Nx))
        #for n in range(Nt):
        Ti = T.copy()
            
        # step 2: detrend the data in two dimensions (least squares plane fit)
        if detre:
            print 'Detrending Data'
            d_obs = np.reshape(Ti, (Nx*Ny,1))
            G = np.ones((Ny*Nx,3))
            for i in range(Ny):
                G[Nx*i:Nx*i+Nx, 0] = i+1
                G[Nx*i:Nx*i+Nx, 1] = np.arange(1, Nx+1)    
            m_est = np.dot(np.dot(lin.inv(np.dot(G.T, G)), G.T), d_obs)
            d_est = np.dot(G, m_est)
            Lin_trend = np.reshape(d_est, (Ny, Nx))
            Ti -= Lin_trend

        # step 3: window the data
        # Hanning window
        if windw:
            print 'Windowing Data'
            windowx = sig.hann(Nx)
            windowy = sig.hann(Ny)
            window = windowx*windowy[:,np.newaxis] 
            Ti *= window

        # step 4: do the FFT for each timestep and aggregate the results
        Tif = fft.fftshift(fft.fft2(Ti))
        PSD_2d += np.real(Tif*np.conj(Tif))

        # step 5: derive the isotropic spectrum
        kk, ll = np.meshgrid(k, l)
        K = np.sqrt(kk**2 + ll**2)
        Ki = np.linspace(0, k.max(), nbins)
        Kidx = np.digitize(K.ravel(), Ki)
        area = np.bincount(Kidx)
        isotropic_spectrum = np.ma.masked_invalid(
                               np.bincount(Kidx, weights=PSD_2d.ravel()) / area )
        
        # step 6: return the results
        return nbins, Nx, Ny, k, l, PSD_2d, Ki, isotropic_spectrum[1:], area[1:]

    def structure_function(self, varname='bT', ndel=8, detre=False, windw=False):
        """Calculate a two-dimensional power spectrum of netcdf variable 'varname'
           in the box defined by lonrange and latrange.
        """        

        # load data
        T = self.mat[varname][:]
        Ny, Nx = T.shape
        Hi = np.zeros(ndel)
        Hj = np.zeros(ndel)
        sumcounti = np.zeros(ndel)
        sumcountj = np.zeros(ndel)

        Ti = T.copy()
        # detrend data
        if detre:
            print 'Detrending Data'
            d_obs = np.reshape(Ti, (Nx*Ny,1))
            G = np.ones((Ny*Nx,3))
            for i in range(Ny):
                G[Nx*i:Nx*i+Nx, 0] = i+1
                G[Nx*i:Nx*i+Nx, 1] = np.arange(1, Nx+1)
            m_est = np.dot(np.dot(lin.inv(np.dot(G.T, G)), G.T), d_obs)
            d_est = np.dot(G, m_est)
            Lin_trend = np.reshape(d_est, (Ny, Nx))
            Ti -= Lin_trend

        # window the data
        if windw:
            print 'Windowing Data'
            windowx = sig.hann(Nx)
            windowy = sig.hann(Ny)
            window = windowx*windowy[:,np.newaxis]
            Ti *= window

        # Difference with 2^m gridpoints in between
        for m in range(ndel):
            dSSTi = np.ma.masked_array((np.absolute(Ti[:,2**m:] - Ti[:,:-2**m]))**2) # .filled(0.)
            dSSTj = np.ma.masked_array((np.absolute(Ti[2**m:] - Ti[:-2**m]))**2) # .filled(0.)
            counti = (~dSSTi.mask).astype('i4')
            countj = (~dSSTj.mask).astype('i4')
            sumcounti[m] = np.sum(counti)
            sumcountj[m] = np.sum(countj)
            Hi[m] = np.sum(np.absolute(dSSTi))/sumcounti[m]
            Hj[m] = np.sum(np.absolute(dSSTj))/sumcountj[m]
    
        return ndel, Hi, Hj
        

