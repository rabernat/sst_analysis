import numpy as np
import netCDF4


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
        
    def biharmonic_tendency(self, T):
        """Caclulate tendency due to biharmonic diffusion of T."""
        d2tk = self._ahf * self.laplacian(T)
        return self._ah * self.laplacian(d2tk)
        
    def horizontal_flux_divergence(self, uflux, vflux):
        """Designed to be used with diagnostics such as DIFE_*, DIFN_*.
        Returns a pure tendency."""
        workx = 0.5 * uflux * self._dyu
        worky = 0.5 * vflux * self._dxu
        work1 = workx + np.roll(workx, 1, axis=-1)
        work1 -= np.roll(work1, 1, axis=-2)
        work2 = worky + np.roll(worky, 1, axis=-2)
        work2 -= np.roll(work2, 1, axis=-1)
        res = work1 + work2
        if self.is3d and (res.ndim>2):
            if res.shape[-3]==self.Nz:
                return np.ma.masked_array(res, self.mask3d).filled(0.)
        else:
            return np.ma.masked_array(res, self.mask).filled(0.)
    
    

