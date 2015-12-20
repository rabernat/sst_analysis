import numpy as np
import xray
import netCDF4
import warnings
import sys
from warnings import warn
from scipy import linalg as lin
from scipy import signal as sig
from scipy import fftpack as fft
from scipy import interpolate as naiso
import gfd
import jmd95

class POPFile(object):
    
    def __init__(self, fname, areaname='TAREA', maskname='KMT', hmax=None, hconst=None, pref=0., ah=-3e17, is3d=False):
        """Wrapper for POP model netCDF files"""
        #self.nc = netCDF4.Dataset(fname)
        self.nc = xray.open_dataset(fname, decode_times=False)
        #self.Ny, self.Nx = self.nc.variables[areaname].shape  
        self.Ny, self.Nx = self.nc[areaname].shape  
        self._ah = ah
        
        # mask
        self.mask = self.nc.variables[maskname][:] <= 1

        self.is3d = is3d
        if self.is3d:
            #self.z_t = nc.variables['z_t'][:]
            #self.z_w_top = nc.variables['z_w_top'][:]
            #self.z_w_bot = nc.variables['z_w_bop'][:]
            self.z_t = nc['z_t'][:]
            self.z_w_top = nc['z_w_top'][:]
            self.z_w_bot = nc['z_w_bop'][:]
            self.Nz = len(self.z_t)
            #kmt = p.nc.variables['KMT'][:]
            kmt = nc['KMT'][:]
            #self.mask3d = np.zeros((self.Nz, self.Ny, self.Nx), dtype='b')
            self.mask3d = xray.DataArray(np.zeros((self.Nz, self.Ny, self.Nx), dtype='b'), coords=kmt.coords, dims=kmt.dims)
            Nz = mask3d.shape[0]
            for k in range(Nz):
                self.mask3d[k] = (kmt<=k)
          
        self.ts_forcing = TSForcing(self, hmax=hmax, hconst=hconst)
        self._ah = ah

    def mask_field(self, maskname='KMT', varname='SST'):
        """Apply mask to tracer field T"""
        #mask = self.nc.variables[maskname][:]
        #T = self.nc.variables[varname][:]
        mask = self.nc[maskname][:]
        T = self.nc[varname][:]
        #return np.ma.masked_array(T, mask<=1)
        return T.where( mask>1 )
        
    def initialize_gradient_operator(self, areaname='TAREA'):
        """Needs to be called before calculating gradients"""
        # raw grid geometry
        #work1 = (self.nc.variables['HTN'][:] /
        #         self.nc.variables['HUW'][:])
        work1 = (self.nc['HTN'][:] / self.nc['HUW'][:]).values
        #tarea = self.nc.variables[areaname][:]
        tarea = self.nc[areaname].values
        self.tarea = tarea
        tarea_r = np.ma.masked_invalid(tarea**-1).filled(0.)
        #tarea_r = xray.DataArray( np.ma.masked_invalid(tarea**-1).filled(0.), coords=tarea.coords, dims=tarea.dims )
        dtn = work1*tarea_r
        dts = np.roll(work1,-1,axis=0)*tarea_r
        #dts = xray.DataArray.roll(work1,-1,axis=0)*tarea_r
        
        #work1 = (self.nc.variables['HTE'][:] /
        #         self.nc.variables['HUS'][:])
        work1 = (self.nc['HTE'][:] / self.nc['HUS'][:]).values
        dte = work1*tarea_r
        dtw = np.roll(work1,-1,axis=1)*tarea_r
        #dtw = xray.DataArray.roll(work1,-1,axis=1)*tarea_r
        
        # boundary conditions
        #kmt = self.nc.variables['KMT'][:] > 1
        kmt = self.nc['KMT'].values > 1
        kmtn = np.roll(kmt,-1,axis=0)
        kmts = np.roll(kmt,1,axis=0)
        kmte = np.roll(kmt,-1,axis=1)
        kmtw = np.roll(kmt,1,axis=1)
        #kmtn = kmt.roll(nlat=-1)
        #kmts = kmt.roll(nlat=1)
        #kmte = kmt.roll(nlon=-1)
        #kmtw = kmt.roll(nlon=1)
        self._cn = np.where( kmt & kmtn, dtn, 0.)
        self._cs = np.where( kmt & kmts, dts, 0.)
        self._ce = np.where( kmt & kmte, dte, 0.)
        self._cw = np.where( kmt & kmtw, dtw, 0.)
        #self._cn = xray.DataArray(np.where( kmt & kmtn, dtn, 0.))
        #self._cs = xray.DataArray(np.where( kmt & kmts, dts, 0.))
        #self._ce = xray.DataArray(np.where( kmt & kmte, dte, 0.))
        #self._cw = xray.DataArray(np.where( kmt & kmtw, dtw, 0.))
        self._cc = -(self._cn + self._cs + self._ce + self._cw)
        
        # mixing coefficients
        #self._ah = -0.2e20*(1280.0/self.Nx)
        #j_eq = np.argmin(self.nc.variables['ULAT'][:,0]**2)
        j_eq = np.argmin(self.nc['ULAT'][:,0]**2).values
        #self._ahf = (tarea / self.nc.variables['UAREA'][j_eq,0])**1.5
        self._ahf = (tarea / self.nc['UAREA'].values[j_eq,0])**1.5
        self._ahf[self.mask] = 0.   
        
        # stuff for gradient
        # reciprocal of dx and dy (in meters)
        #self._dxtr = 100.*self.nc.variables['DXT'][:]**-1
        self._dxtr = 100.*self.nc['DXT'].values**-1
        #self._dytr = 100.*self.nc.variables['DYT'][:]**-1
        self._dytr = 100.*self.nc['DYT'].values**-1
        self._kmaske = np.where(kmt & kmte, 1., 0.)
        #self._kmaske = xray.DataArray(np.where(kmt & kmte, 1., 0.))
        self._kmaskn = np.where(kmt & kmtn, 1., 0.)
        #self._kmaskn = xray.DataArray(np.where(kmt & kmtn, 1., 0.))
        
        #self._dxu = self.nc.variables['DXU'][:]
        #self._dyu = self.nc.variables['DYU'][:]
        self._dxu = self.nc['DXU'].values
        self._dyu = self.nc['DYU'].values
                
    def laplacian(self, T):
        """Returns the laplacian of T at the tracer point."""
        return (
            self._cc*T +
            self._cn*np.roll(T,-1,axis=0) +
            self._cs*np.roll(T,1,axis=0) +
            self._ce*np.roll(T,-1,axis=1) +
            self._cw*np.roll(T,1,axis=1)          
        )
        #return (
            #self._cc*T +
            #self._cn*T.roll(nlat=-1) + self._cs*T.roll(nlat=1) +
            #self._ce*T.roll(nlon=-1) + self._cw*T.roll(nlon=1)          
        #)
    
    def gradient_modulus(self, varname='SST', lonname='ULONG', latname='ULAT', maskname='KMT', dxname='DXU', dyname='DYU', lonrange=(154.9,171.7), latrange=(30,45.4), roll=-1000):
        """Return the modulus of the gradient of tracer at U points."""
        
        #tlon = np.roll(self.nc.variables[lonname][:], roll, axis=1)
        #tlat = np.roll(self.nc.variables[latname][:], roll, axis=1)
        tlon = self.nc[lonname][:].roll(nlon=roll)
        tlat = self.nc[latname][:].roll(nlon=roll)
        #tmask = np.roll(self.nc.variables[maskname][:], roll) <= 1

        # step 1: figure out the box indices
        lonmask = (tlon >= lonrange[0]) & (tlon < lonrange[1])
        latmask = (tlat >= latrange[0]) & (tlat < latrange[1])
        #boxidx = lonmask & latmask       # this won't necessarily be square
        boxidx = latmask & lonmask
        irange = np.where(boxidx.sum(axis=0))[0]
        #irange = xray.DataArray.where(boxidx.sum(dim='nlat'))[0]
        imin, imax = irange.min(), irange.max()
        jrange = np.where(boxidx.sum(axis=1))[0]
        #jrange = xray.DataArray.where(boxidx.sum(dim='nlon'))[0]
        jmin, jmax = jrange.min(), jrange.max()
        Nx = imax - imin
        Ny = jmax - jmin
        lon = tlon[jmin:jmax, imin:imax]
        lat = tlat[jmin:jmax, imin:imax]
        #mask = tmask[jmin:jmax, imin:imax]

        # step 2: load the data
        #T = np.roll(self.nc.variables[varname],-1000)[..., jmin:jmax, imin:imax]
        if varname=='SST':
            #T = np.roll(self.nc.variables[varname][:], roll)[..., jmin:jmax, imin:imax]
            T = self.nc.variables[varname][:]
        else:
            #T = 1e-2*np.roll(self.nc.variables[varname][:], roll)[..., jmin:jmax, imin:imax]
            T = 1e-2*(self.nc.variables[varname][:])  # in meters
        
        Nt, Ny, Nx = T.shape
        Ti = np.zeros((Nt,Ny,Nx))
        mask = self.nc.variables[maskname][:] <= 1
        region_mask = np.roll(mask, roll)[jmin:jmax, imin:imax]
        for n in range(Nt):
            Ti[n] = np.roll(np.ma.masked_array(T[n].copy(), mask), roll, axis=1)
        dx = 1e-2*np.roll(self.nc.variables[dxname][:], roll, axis=1)
        dy = 1e-2*np.roll(self.nc.variables[dyname][:], roll, axis=1)
        barTy = .5*(np.roll(Ti,1,axis=1)+Ti)
        barTx = .5*(np.roll(Ti,1,axis=2)+Ti)
        
        # step 3: calculate the difference
        dxdbarTy = .5 * (barTy - np.roll(barTy,1,axis=2))/dx
        dydbarTx = .5 * (barTx - np.roll(barTx,1,axis=1))/dy
        #dTx = np.roll(dTx, roll)[:, jmin:jmax, imin:imax]
        #dTy = np.roll(dTy, roll)[:, jmin:jmax, imin:imax]
        
        return lon, lat, dxdbarTy[:,jmin:jmax,imin:imax], dydbarTx[:,jmin:jmax,imin:imax]
    
    def biharmonic_tendency(self, T):
        """Caclulate tendency due to biharmonic diffusion of T."""
        d2tk = self._ahf * self.laplacian(T)
        return self._ah * self.laplacian(d2tk)
        
    def horizontal_flux_divergence(self, hfluxe, hfluxn):
        """Designed to be used with diagnostics such as DIFE_*, DIFN_*.
        Returns a pure tendency."""
        return ( hfluxe - np.roll(hfluxe, 1, -1) +
                 hfluxn - np.roll(hfluxn, 1, -2) )
        
    def mag_gradient_modulus(self, varname='SST', lonname='TLONG', latname='TLAT', maskname='KMT', detr_win=False, lonrange=(154.9,171.7), latrange=(30,45.4), roll=-1000):
        """Return the modulus of the gradient of T at the tracer point."""
        
        tlon = np.roll(self.nc.variables[lonname][:], roll)
        tlat = np.roll(self.nc.variables[latname][:], roll)
        #tmask = np.roll(self.nc.variables[maskname][:], roll) <= 1

        # step 1: figure out the box indices
        lonmask = (tlon >= lonrange[0]) & (tlon < lonrange[1])
        latmask = (tlat >= latrange[0]) & (tlat < latrange[1])
        boxidx = lonmask & latmask       # this won't necessarily be square
        irange = np.where(boxidx.sum(axis=0))[0]
        imin, imax = irange.min(), irange.max()
        jrange = np.where(boxidx.sum(axis=1))[0]
        jmin, jmax = jrange.min(), jrange.max()
        Nx = imax - imin
        Ny = jmax - jmin
        lon = tlon[jmin:jmax, imin:imax]
        lat = tlat[jmin:jmax, imin:imax]
        #mask = tmask[jmin:jmax, imin:imax]

        # step 2: load the data
        #T = np.roll(self.nc.variables[varname],-1000)[..., jmin:jmax, imin:imax]
        if varname=='SST':
            #T = np.roll(self.nc.variables[varname][:], roll)[..., jmin:jmax, imin:imax]
            #T = self.nc.variables[varname][:]
            T = self.nc[varname][:]
        else:
            #T = 1e-2*np.roll(self.nc.variables[varname][:], roll)[..., jmin:jmax, imin:imax]
            #T = 1e-2*(self.nc.variables[varname][:])
            T = 1e-2*(self.nc[varname][:])
        
        # step 6: detrend the data in two dimensions (least squares plane fit)
        if detr_win:
            for n in range(Nt):
                Ti[n] = np.ma.masked_array(T[n], region_mask)
                d_obs = np.reshape(Ti, (Nx*Ny,1))
                G = np.ones((Ny*Nx,3))
                for i in range(Ny):
                    G[Nx*i:Nx*i+Nx, 0] = i+1
                    G[Nx*i:Nx*i+Nx, 1] = np.arange(1, Nx+1)    
                m_est = np.dot(np.dot(lin.inv(np.dot(G.T, G)), G.T), d_obs)
                d_est = np.dot(G, m_est)
                Lin_trend = np.reshape(d_est, (Ny, Nx))
                Ti -= Lin_trend
    
                # step 7: window the data
                # Hanning window
                windowx = sig.hann(Nx)
                windowy = sig.hann(Ny)
                window = windowx*windowy[:,np.newaxis] 
                Ti *= window
        
        # step 3: calculate the difference
        #dTx = self._kmaske * (np.roll(Ti,-1,axis=0) - Ti)
        #dTy = self._kmaskn * (np.roll(Ti,-1,axis=1) - Ti)
        dTx = self._kmaske * xray.DataArray( (np.roll(Ti,-1,axis=0) - Ti), coords=self._kmaske.coords, dims=self._kmaske.dims )
        dTy = self._kmaskn * xray.DataArray( (np.roll(Ti,-1,axis=1) - Ti), coords=self._kmaskn.coords, dims=self._kmaskn.dims )
        #dTx = np.roll(dTx, roll)[:, jmin:jmax, imin:imax]
        #dTy = np.roll(dTy, roll)[:, jmin:jmax, imin:imax]
        
        return jmin, jmax, imin, imax, roll, lon, lat, xray.DataArray( np.sqrt( 0.5 *
                            (dTx**2 + xray.DataArray.roll(dTx,1,axis=0)**2) * self._dxtr**2
                            + .5 * (dTy**2 + xrayDataArray.roll(dTy,1,axis=1)**2) * self._dytr**2 ),  )
                   #np.sqrt( 0.5 *
                            #(dTx**2 + xray.DataArray.roll(dTx,1,axis=0)**2) * self._dxtr**2
                            #+ .5 * (dTy**2 + xrayDataArray.roll(dTy,1,axis=1)**2) * self._dytr**2 )
    
    def detrend_window_2d(self, varname='SST', lonname='TLONG', latname='TLAT', maskname='KMT', geos=False, lonrange=(154.9,171.7), latrange=(30,45.4), roll=-1000):
        """Detrend and window 2D data"""
        
        tlon = np.roll(self.nc.variables[lonname][:], roll)
        tlat = np.roll(self.nc.variables[latname][:], roll)
        #tmask = np.roll(self.nc.variables[maskname][:], roll) <= 1

        # step 1: figure out the box indices
        lonmask = (tlon >= lonrange[0]) & (tlon < lonrange[1])
        latmask = (tlat >= latrange[0]) & (tlat < latrange[1])
        boxidx = lonmask & latmask       # this won't necessarily be square
        irange = np.where(boxidx.sum(axis=0))[0]
        imin, imax = irange.min(), irange.max()
        jrange = np.where(boxidx.sum(axis=1))[0]
        jmin, jmax = jrange.min(), jrange.max()
        Nx = imax - imin
        Ny = jmax - jmin
        lon = tlon[jmin:jmax, imin:imax]
        lat = tlat[jmin:jmax, imin:imax]
        #mask = tmask[jmin:jmax, imin:imax]

        # step 2: load the data
        #T = np.roll(self.nc.variables[varname],-1000)[..., jmin:jmax, imin:imax]
        if varname=='SST':
            #T = np.roll(self.nc.variables[varname][:], roll)[..., jmin:jmax, imin:imax]
            T = np.roll(self.nc.variables[varname][:], roll, axis=2)
        else:
            #T = 1e-2*np.roll(self.nc.variables[varname][:], roll)[..., jmin:jmax, imin:imax]
            T = 1e-2*np.roll(self.nc.variables[varname][:], roll, axis=2)
       
        Nt = T.shape[0]
        Ti = np.zeros((Nt,Ny,Nx))
        mask = self.nc.variables[maskname][:] <= 1
        region_mask = np.roll(mask, roll)[jmin:jmax, imin:imax]
        for n in range(Nt):
            Ti[n] = np.ma.masked_array(T[n,jmin:jmax, imin:imax].copy(), region_mask)
        
        # detrend
        d_obs = np.reshape(Ti, (Nx*Ny,1))
        G = np.ones((Ny*Nx,3))
        for i in range(Ny):
            G[Nx*i:Nx*i+Nx, 0] = i+1
            G[Nx*i:Nx*i+Nx, 1] = np.arange(1, Nx+1)    
        m_est = np.dot(np.dot(lin.inv(np.dot(G.T, G)), G.T), d_obs)
        d_est = np.dot(G, m_est)
        Lin_trend = np.reshape(d_est, (Ny, Nx))
        Ti -= Lin_trend
        
        # window
        windowx = sig.hann(Nx)
        windowy = sig.hann(Ny)
        window = windowx*windowy[:,np.newaxis] 
        Ti *= window
        
        return Ti
    
    def variance_2d(self, varname='SST', lonname='TLONG', latname='TLAT', maskname='KMT', dxname='DXT', dyname='DYT', spac=True, filename=False, geos=False, grad=False, roll=-1000, nbins=128, MAX_LAND=0.01, xmin=0, xmax=0, ymin=0, ymax=0, ymin_bound=0, ymax_bound=0, xmin_bound=0, xmax_bound=0):
        """Calculate the variance in the spatial domain or spectral domain"""
        jmin_bound = ymin_bound
        jmax_bound = ymax_bound
        imin_bound = xmin_bound
        imax_bound = xmax_bound
        mask = self.nc.variables[maskname][:] <= 1
        tlon = np.roll(np.ma.masked_array(self.nc.variables[lonname][:],mask), roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        tlat = np.roll(np.ma.masked_array(self.nc.variables[latname][:],mask), roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        tlon[tlon<0.] += 360.
        
        imax = xmax
        imin = xmin
        jmax = ymax
        jmin = ymin
        Nx = imax - imin
        Ny = jmax - jmin
        lon = tlon[jmin:jmax, imin:imax]
        lat = tlat[jmin:jmax, imin:imax]
        #dlon_domain = np.roll(tlon,1)-np.roll(tlon, -1)
        #dlat_domain = np.roll(tlat,1)-np.roll(tlat, -1)
        dx = 1e-2*np.roll(self.nc.variables[dxname][:], roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        dy = 1e-2*np.roll(self.nc.variables[dyname][:], roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]

        # step 2: load the data
        #T = np.roll(self.nc.variables[varname],-1000)[..., jmin:jmax, imin:imax]
        if varname=='SST' or varname=='SSS':
            T = np.roll(self.nc.variables[varname][:], roll, axis=2)[:,jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
            #dx = 1e-2*np.roll(self.nc.variables['DXT'][:], roll, axis=1)
            #dy = 1e-2*np.roll(self.nc.variables['DYT'][:], roll, axis=1)
        elif varname=='SSH_2':
            T = 1e-2*np.roll(self.nc.variables[varname][:], roll, axis=2)[:,jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
            #T = gfd.g/gfd.f_coriolis(lat)*(np.roll(T,1)-np.roll(T,-1))/(gfd.A*np.cos(np.radians(dlat_domain))*np.radians(dlon_domain))
            if geos:
                barTy = .5*(np.roll(T,1,axis=1)+T)
                T = gfd.g / gfd.f_coriolis(tlat) * (np.roll(barTy,1,axis=2)-barTy) / dx
            elif grad:
                barTy = .5*(np.roll(T,1,axis=1)+T)
                T = (np.roll(barTy,1,axis=2)-np.roll(barTy,-1,axis=2)) / (dx+np.roll(dx,-1,axis=1))
        else:
            T = 1e-2*np.roll(self.nc.variables[varname][:], roll, axis=2)[:,jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]

        # step 3: figure out if there is too much land in the box
        #MAX_LAND = 0.01 # only allow up to 1% of land
        mask_domain = np.roll(mask, roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        region_mask = mask_domain[jmin:jmax, imin:imax]
        land_fraction = region_mask.sum().astype('f8') / (Ny*Nx)
        
        # Wavenumber step
        dx_domain = dx[jmin:jmax,imin:imax].copy()
        dy_domain = dy[jmin:jmax,imin:imax].copy()
        k = 2*np.pi*fft.fftshift(fft.fftfreq(Nx, dx_domain[Ny/2,Nx/2]))
        l = 2*np.pi*fft.fftshift(fft.fftfreq(Ny, dy_domain[Ny/2,Nx/2]))
        dk = np.diff(k)[0]*.5/np.pi
        dl = np.diff(l)[0]*.5/np.pi

        ###################################
        ###  Start looping through each time step  ####
        ###################################
        Nt = T.shape[0]
        tilde2_sum = np.zeros((Ny,Nx))
        Ti2_sum = np.zeros((Ny,Nx))
        var = np.zeros((Ny,Nx))
        for n in range(Nt):
            Ti = np.ma.masked_array(T[n, jmin:jmax, imin:imax].copy(), region_mask)
            
            # step 5: interpolate the missing data (only if necessary)
            if land_fraction>0. and land_fraction<MAX_LAND:
                x = np.arange(0,Nx)
                y = np.arange(0,Ny)
                X,Y = np.meshgrid(x,y)
                Zr = Ti.ravel()
                Xr = np.ma.masked_array(X.ravel(), Zr.mask)
                Yr = np.ma.masked_array(Y.ravel(), Zr.mask)
                Xm = np.ma.masked_array( Xr.data, ~Xr.mask ).compressed()
                Ym = np.ma.masked_array( Yr.data, ~Yr.mask ).compressed()
                Zm = naiso.griddata(np.array([Xr.compressed(), Yr.compressed()]).T, 
                                    Zr.compressed(), np.array([Xm,Ym]).T, method='nearest')
                Znew = Zr.data
                Znew[Zr.mask] = Zm
                Znew.shape = Ti.shape
                Ti = Znew
            elif land_fraction==0.:
            # no problem
                pass
            else:
                break
        
            # step 6: detrend the data in two dimensions (least squares plane fit)
            d_obs = np.reshape(Ti, (Nx*Ny,1))
            G = np.ones((Ny*Nx,3))
            for i in range(Ny):
                G[Nx*i:Nx*i+Nx, 0] = i+1
                G[Nx*i:Nx*i+Nx, 1] = np.arange(1, Nx+1)    
            m_est = np.dot(np.dot(lin.inv(np.dot(G.T, G)), G.T), d_obs)
            d_est = np.dot(G, m_est)
            Lin_trend = np.reshape(d_est, (Ny, Nx))
            Ti -= Lin_trend

            # step 7: window the data
            # Hanning window
            windowx = sig.hann(Nx)
            windowy = sig.hann(Ny)
            window = windowx*windowy[:,np.newaxis] 
            Ti *= window
            
            # step 8: calculate necessary variables
            Ti2_sum += Ti**2
            var += Ti**2*dx_domain*dy_domain
            Tif = fft.fftshift(fft.fft2(Ti))    # [u^2] (u: unit)
            tilde2_sum += np.real(Tif*np.conj(Tif))
        
        # step 9: check whether the Plancherel theorem is satisfied
        #tilde2_ave = tilde2_sum/Nt
        breve2_sum = tilde2_sum/((Nx*Ny)**2*dk*dl)
        #breve2_ave = tilde2_ave/((Nx*Ny)**2*dk*dl)
        #spac2_ave = Ti2_sum/Nt
        if land_fraction==0.:
            #np.testing.assert_almost_equal(breve2_ave.sum()/(dx_domain[Ny/2,Nx/2]*dy_domain[Ny/2,Nx/2]*(spac2_ave).sum()), 1., decimal=5)
            np.testing.assert_almost_equal(breve2_sum.sum()/(dx_domain[Ny/2,Nx/2]*dy_domain[Ny/2,Nx/2]*(Ti2_sum).sum()), 1., decimal=5)
        
        if spac == True:
            Variance = var.sum()
        else:
            # step 9: check whether the Plancherel theorem is satisfied
            #tilde2_ave = tilde2_sum/Nt
            #breve2_ave = tilde2_ave/((Nx*Ny)**2*dk*dl)
            #spac2_ave = Ti2_sum/Nt
            #if land_fraction==0.:
            #    np.testing.assert_almost_equal(breve2_ave.sum()/(dx_domain[Ny/2,Nx/2]*dy_domain[Ny/2,Nx/2]*(spac2_ave).sum()), 1., decimal=5)
        
            # step 10: derive the isotropic spectrum
            kk, ll = np.meshgrid(k, l)
            K = np.sqrt(kk**2 + ll**2)
            Ki = np.linspace(0, k.max(), nbins)
            #Ki = np.linspace(0, K.max(), nbins)
            deltaKi = np.diff(Ki)[0]
            Kidx = np.digitize(K.ravel(), Ki)
            area = np.bincount(Kidx)
            isotropic_PSD = np.ma.masked_invalid(
                                       np.bincount(Kidx, weights=(breve2_sum).ravel()) / area )[:-1] *Ki*2.*np.pi**2
        #isotropic_PSD = np.ma.masked_invalid(
        #                              np.bincount(Kidx, weights=(2.*dx*dy/Nx/Ny*PSD_ave*K*2.*np.pi).ravel()) / area )
        #isotropic_PSD = np.ma.masked_invalid(
        #                               np.bincount(Kidx, weights=(PSD_ave).ravel()) / area )[1:] *2.*np.pi/deltaKi
            #Variance in the spectral domain
            Variance = 2.*np.pi**2*deltaKi*(isotropic_PSD*Ki).sum()
        
        return Nt, Variance
        
    def power_spectrum_2d(self, varname='SST', lonname='TLONG', latname='TLAT', maskname='KMT', dxname='DXT', dyname='DYT', filename=False, geosx=False, geosy=False, grady=False, lonrange=(154.9,171.7), latrange=(30,45.4), roll=-1000, nbins=128, MAX_LAND=0.01, xmin=0, xmax=0, ymin=0, ymax=0, ymin_bound=0, ymax_bound=0, xmin_bound=0, xmax_bound=0):
        """Calculate a two-dimensional power spectrum of netcdf variable 'varname'
            in the box defined by lonrange and latrange.
        """
        jmin_bound = ymin_bound
        jmax_bound = ymax_bound
        imin_bound = xmin_bound
        imax_bound = xmax_bound
        #mask = self.nc.variables[maskname][:] <= 1
        #numpy_mask = self.nc[maskname].values <= 1
        mask = self.nc[maskname] <= 1
        #tlon = np.roll(np.ma.masked_array(self.nc.variables[lonname][:],mask), roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        #tlat = np.roll(np.ma.masked_array(self.nc.variables[latname][:],mask), roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        #tlon = np.roll(np.ma.masked_array(self.nc[lonname].values, mask), roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        #tlat = np.roll(np.ma.masked_array(self.nc[latname].values, mask), roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        #tlon = np.ma.masked_array(self.nc[lonname].roll( nlon=roll ).values, 
                                  #numpy_mask)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        #tlat = np.ma.masked_array(self.nc[latname].roll( nlon=roll ).values,
                                  #numpy_mask)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        #tlon[tlon<0.] += 360.
        
        #Ny, Nx = tlon.shape
        #x = np.arange(0,Nx)
        #y = np.arange(0,Ny)
        #X,Y = np.meshgrid(x,y)
        #Zr = tlon.ravel()
        #Xr = np.ma.masked_array(X.ravel(), Zr.mask)
        #Yr = np.ma.masked_array(Y.ravel(), Zr.mask)
        #Xm = np.ma.masked_array( Xr.data, ~Xr.mask ).compressed()
        #Ym = np.ma.masked_array( Yr.data, ~Yr.mask ).compressed()
        #Zm = naiso.griddata(np.array([Xr.compressed(), Yr.compressed()]).T, 
        #                            Zr.compressed(), np.array([Xm,Ym]).T, method='linear')
        #Znew = Zr.data
        #Znew[Zr.mask] = Zm
        #Znew.shape = tlon.shape
        #tlon = Znew.copy()

        #Zr = tlat.ravel()
        #Xr = np.ma.masked_array(X.ravel(), Zr.mask)
        #Yr = np.ma.masked_array(Y.ravel(), Zr.mask)
        #Xm = np.ma.masked_array( Xr.data, ~Xr.mask ).compressed()
        #Ym = np.ma.masked_array( Yr.data, ~Yr.mask ).compressed()
        #Zm = naiso.griddata(np.array([Xr.compressed(), Yr.compressed()]).T, 
        #                            Zr.compressed(), np.array([Xm,Ym]).T, method='linear')
        #Znew = Zr.data
        #Znew[Zr.mask] = Zm
        #Znew.shape = tlat.shape
        #tlat = Znew.copy()

            
        #self.mask = self.nc.variables[maskname][:] <= 1

        # step 1: figure out the box indices
        #lonmask = (tlon >= lonrange[0]) & (tlon < lonrange[1])
        #latmask = (tlat >= latrange[0]) & (tlat < latrange[1])
        #boxidx = lonmask & latmask # this won't necessarily be square
        #irange = np.where(boxidx.sum(axis=0))[0]
        #imin, imax = irange.min(), irange.max()
        #jrange = np.where(boxidx.sum(axis=1))[0]
        #jmin, jmax = jrange.min(), jrange.max()
        imax = xmax
        imin = xmin
        jmax = ymax
        jmin = ymin
        Nx = imax - imin
        Ny = jmax - jmin
        #lon = tlon[jmin:jmax, imin:imax]
        #lat = tlat[jmin:jmax, imin:imax]
        #dlon_domain = np.roll(tlon,1)-np.roll(tlon, -1)
        #dlat_domain = np.roll(tlat,1)-np.roll(tlat, -1)
        #dx = 1e-2*np.roll(self.nc.variables[dxname][:], roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        #dy = 1e-2*np.roll(self.nc.variables[dyname][:], roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        #dx = 1e-2*np.roll(self.nc[dxname].values, roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        #dy = 1e-2*np.roll(self.nc[dyname].values, roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        dx = 1e-2 * self.nc[dxname][:].roll( nlon=roll ).values[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        dy = 1e-2 * self.nc[dyname][:].roll( nlon=roll ).values[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]

        # step 2: load the data
        #T = np.roll(self.nc.variables[varname],-1000)[..., jmin:jmax, imin:imax]
        if varname=='SST' or varname=='SSS' or varname=='T_diss' or varname=='S_diss' or varname=='T_forc' or varname=='S_forc':
            #T = np.roll(self.nc.variables[varname][:], roll, axis=2)[:,jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
            #T = np.roll(self.nc[varname].values, roll, axis=2)[:,jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
            T = self.nc[varname].roll( nlon=roll ).values[:, jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
            #dx = 1e-2*np.roll(self.nc.variables['DXT'][:], roll, axis=1)
            #dy = 1e-2*np.roll(self.nc.variables['DYT'][:], roll, axis=1)
        elif varname=='SSH_2':
            #T = 1e-2*np.roll(self.nc.variables[varname][:], roll, axis=2)[:,jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
            #T = 1e-2 * np.roll(self.nc[varname].values, roll, axis=2)[:, jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
            T = 1e-2 * self.nc[varname].roll( nlon=roll ).values[:, jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
            #T = gfd.g/gfd.f_coriolis(lat)*(np.roll(T,1)-np.roll(T,-1))/(gfd.A*np.cos(np.radians(dlat_domain))*np.radians(dlon_domain))
            if geosy:
                barTy = .5*(np.roll(T,1,axis=1)+T)
                T = - gfd.g / gfd.f_coriolis(tlat) * (np.roll(barTy,1,axis=2)-barTy) / dx
                #barTy = .5 * ( T.roll( nlat=1 ) + T )
                #T = - gfd.g / gfd.f_coriolis(tlat) * ( barTy.roll( nlon=1 ) - barTy ) / dx
            elif geosx:
                barTx = .5*(np.roll(T,1,axis=2)+T)
                T = gfd.g / gfd.f_coriolis(tlat) * (np.roll(barTx,1,axis=1)-barTx) / dy
                #barTx = .5 * ( T.roll( nlon=1 ) + T )
                #T = gfd.g / gfd.f_coriolis(tlat) * ( barTx.roll( nlat=1 ) - barTx ) / dy
            elif grady:
                barTy = .5*(np.roll(T,1,axis=1)+T)
                T = (np.roll(barTy,1,axis=2)-np.roll(barTy,-1,axis=2)) / (dx+np.roll(dx,-1,axis=1))
                #barTy = .5 * ( T.roll( nlat=1 ) + T )
                #T = ( barTy.roll( nlon=1 ) - barTy.roll( nlon=-1 ) ) / ( dx + dx.roll(nlat=-1) )
        else:
            #T = 1e-2*np.roll(self.nc.variables[varname][:], roll, axis=2)[:,jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
            #T = 1e-2*np.roll(self.nc[varname].values, roll, axis=2)[:,jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
            T = 1e-2 * self.nc[varname].roll( nlon=roll ).values[:, jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]

        # step 3: figure out if there is too much land in the box
        #MAX_LAND = 0.01 # only allow up to 1% of land
        #mask_domain = ( np.roll( mask, roll, 
                                #axis=1 )[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100] + np.isnan(T)[0] )
        mask_domain = (mask.roll( nlon=roll ).values[jmin_bound:jmax_bound+100, 
                                                     imin_bound:imax_bound+100] + np.isnan(T)[0])
        region_mask = mask_domain[jmin:jmax, imin:imax]
        land_fraction = region_mask.sum().astype('f8') / (Ny*Nx)
        if land_fraction == 0.:
            # no problem
            pass
        elif land_fraction >= MAX_LAND:
            crit = 'false'
            errstr = 'The sector has too much land. land_fraction = ' + str(land_fraction)
            warnings.warn(errstr)
            #raise ValueError('The sector has too much land. land_fraction = ' + str(land_fraction))
        else:    
            # do some interpolation
            errstr = 'The sector has land (land_fraction=%g) but we are interpolating it out.' % land_fraction
            warnings.warn(errstr)
        
        # step 4: figure out FFT parameters (k, l, etc.) and set up result variable
        #dlon = lon[np.round(np.floor(lon.shape[0]*0.5)), np.round(
        #             np.floor(lon.shape[1]*0.5))+1]-lon[np.round(
        #             np.floor(lon.shape[0]*0.5)), np.round(np.floor(lon.shape[1]*0.5))]
        #dlat = lat[np.round(np.floor(lat.shape[0]*0.5))+1, np.round(
        #             np.floor(lat.shape[1]*0.5))]-lat[np.round(
        #             np.floor(lat.shape[0]*0.5)), np.round(np.floor(lat.shape[1]*0.5))]

        # Spatial step
        #dx = gfd.A*np.cos(np.radians(lat[np.round(
        #             np.floor(lat.shape[0]*0.5)),np.round(
        #             np.floor(lat.shape[1]*0.5))]))*np.radians(dlon)
        #dy = gfd.A*np.radians(dlat)
        
        # Wavenumber step
        dx_domain = dx[jmin:jmax,imin:imax].copy()
        dy_domain = dy[jmin:jmax,imin:imax].copy()
        # PREVIOUS
        #k = 2*np.pi*fft.fftshift(fft.fftfreq(Nx, dx_domain[Ny/2,Nx/2]))
        #l = 2*np.pi*fft.fftshift(fft.fftfreq(Ny, dy_domain[Ny/2,Nx/2]))
        #dk = np.diff(k)[0]*.5/np.pi
        #dl = np.diff(l)[0]*.5/np.pi
        k = fft.fftshift(fft.fftfreq(Nx, dx_domain[Ny/2,Nx/2]))
        l = fft.fftshift(fft.fftfreq(Ny, dy_domain[Ny/2,Nx/2]))
        dk = np.diff(k)[0]
        dl = np.diff(l)[0]

        ###################################
        ###  Start looping through each time step  ####
        ###################################
        Nt = T.shape[0]
        Decor_lag = 13
        tilde2_sum = np.zeros((Ny, Nx))
        Ti2_sum = np.zeros((Ny, Nx))
        Days = np.arange(0,Nt,Decor_lag)
        Neff = len(Days)
        for n in Days:
            Ti = np.ma.masked_array(T[n, jmin:jmax, imin:imax].copy(), region_mask)
            
            # step 5: interpolate the missing data (only if necessary)
            if land_fraction>0. and land_fraction<MAX_LAND:
                #x = np.arange(0,Nx)
                #y = np.arange(0,Ny)
                #X,Y = np.meshgrid(x,y)
                #Zr = Ti.ravel()
                #Xr = np.ma.masked_array(X.ravel(), Zr.mask)
                #Yr = np.ma.masked_array(Y.ravel(), Zr.mask)
                #Xm = np.ma.masked_array( Xr.data, ~Xr.mask ).compressed()
                #Ym = np.ma.masked_array( Yr.data, ~Yr.mask ).compressed()
                #Zm = naiso.griddata(np.array([Xr.compressed(), Yr.compressed()]).T, 
                #                    Zr.compressed(), np.array([Xm,Ym]).T, method='nearest')
                #Znew = Zr.data
                #Znew[Zr.mask] = Zm
                #Znew.shape = Ti.shape
                #Ti = Znew
                Ti = interpolate_2d(Ti)
            elif land_fraction==0.:
            # no problem
                pass
            else:
                break
        
            # step 6: detrend the data in two dimensions (least squares plane fit)
            #d_obs = np.reshape(Ti, (Nx*Ny,1))
            #G = np.ones((Ny*Nx,3))
            #for i in range(Ny):
            #    G[Nx*i:Nx*i+Nx, 0] = i+1
            #    G[Nx*i:Nx*i+Nx, 1] = np.arange(1, Nx+1)    
            #m_est = np.dot(np.dot(lin.inv(np.dot(G.T, G)), G.T), d_obs)
            #d_est = np.dot(G, m_est)
            #Lin_trend = np.reshape(d_est, (Ny, Nx))
            #Ti -= Lin_trend
            Ti = detrend_2d(Ti)

            # step 7: window the data
            # Hanning window
            windowx = sig.hann(Nx)
            windowy = sig.hann(Ny)
            window = windowx*windowy[:,np.newaxis] 
            Ti *= window
            
            Ti2_sum += Ti**2

            # step 8: do the FFT for each timestep and aggregate the results
            Tif = fft.fftshift(fft.fft2(Ti))    # [u^2] (u: unit)
            tilde2_sum += np.real(Tif*np.conj(Tif))

        # step 9: check whether the Plancherel theorem is satisfied
        #tilde2_ave = tilde2_sum/Nt
        breve2_sum = tilde2_sum/((Nx*Ny)**2*dk*dl)
        #breve2_ave = tilde2_ave/((Nx*Ny)**2*dk*dl)
        #spac2_ave = Ti2_sum/Nt
        if land_fraction == 0.:
            #np.testing.assert_almost_equal(breve2_ave.sum()/(dx_domain[Ny/2,Nx/2]*dy_domain[Ny/2,Nx/2]*(spac2_ave).sum()), 1., decimal=5)
            np.testing.assert_almost_equal( breve2_sum.sum() / ( dx_domain[Ny/2, Nx/2] * dy_domain[Ny/2, Nx/2] * (Ti2_sum).sum() ), 1., decimal=5)
            
        # step 10: derive the isotropic spectrum
        kk, ll = np.meshgrid(k, l)
        K = np.sqrt( kk**2 + ll**2 )
        #Ki = np.linspace(0, k.max(), nbins)
        if k.max() > l.max():
            Ki = np.linspace(0, l.max(), nbins)
        else:
            Ki = np.linspace(0, k.max(), nbins)
        #Ki = np.linspace(0, K.max(), nbins)
        deltaKi = np.diff(Ki)[0]
        Kidx = np.digitize(K.ravel(), Ki)
        invalid = Kidx[-1]
        area = np.bincount(Kidx)
        #PREVIOUS: isotropic_PSD = np.ma.masked_invalid(
        #                               np.bincount(Kidx, weights=breve2_sum.ravel()) / area )[:-1] *Ki*2.*np.pi**2
        isotropic_PSD = np.ma.masked_invalid(
                                       np.bincount(Kidx, weights=breve2_sum.ravel()) / area )[:-1] * Ki
        #isotropic_PSD = np.ma.masked_invalid(
                                       #np.bincount(Kidx, weights=(breve2_ave).ravel()) / area )[1:] *Ki*2.*np.pi**2
        #isotropic_PSD = np.ma.masked_invalid(
        #                              np.bincount(Kidx, weights=(2.*dx*dy/Nx/Ny*PSD_ave*K*2.*np.pi).ravel()) / area )
        #isotropic_PSD = np.ma.masked_invalid(
        #                               np.bincount(Kidx, weights=(PSD_ave).ravel()) / area )[1:] *2.*np.pi/deltaKi
        
        # Usage of digitize
        #>>> x = np.array([-0.2, 6.4, 3.0, 1.6, 20.])
        #>>> bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        #>>> inds = np.digitize(x, bins)
        #array([0, 4, 3, 2, 5])
        
        # Usage of bincount 
        #>>> np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
        #array([1, 3, 1, 1, 0, 0, 0, 1])
        # With the option weight
        #>>> w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6]) # weights
        #>>> x = np.array([0, 1, 1, 2, 2, 2])
        #>>> np.bincount(x,  weights=w)
        #array([ 0.3,  0.7,  1.1])  <- [0.3, 0.5+0.2, 0.7+1.-0.6]
        
        # step 10: return the results
        #PREVIOUS: return Neff, Nt, Nx, Ny, k, l, Ti2_sum, tilde2_sum, breve2_sum, Ki, isotropic_PSD, area[1:], lon, lat, land_fraction, MAX_LAND
        return Neff, Nt, Nx, Ny, k, l, Ti2_sum, tilde2_sum, breve2_sum, Ki[:], isotropic_PSD[:], area[1:-1], land_fraction, MAX_LAND
    
    def tendency_2d(self, varname='SST', lonname='TLONG', latname='TLAT', maskname='KMT', dxname='DXT', dyname='DYT', filename=False, roll=-1000, nbins=128, MAX_LAND=0.01, xmin=0, xmax=0, ymin=0, ymax=0, ymin_bound=0, ymax_bound=0, xmin_bound=0, xmax_bound=0):
        """Calculate the tendency terms in a two-dimensional power spectral variance budget 
            of netcdf variable 'varname' in a lat-lon box.
        """
        jmin_bound = ymin_bound
        jmax_bound = ymax_bound
        imin_bound = xmin_bound
        imax_bound = xmax_bound
        #mask = self.nc.variables[maskname][:] <= 1
        mask = self.nc[maskname].values <= 1
        #tlon = np.roll(np.ma.masked_array(self.nc.variables[lonname][:],mask), roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        #tlat = np.roll(np.ma.masked_array(self.nc.variables[latname][:],mask), roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        tlon = np.roll(np.ma.masked_array(self.nc[lonname].values, mask), roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        tlat = np.roll(np.ma.masked_array(self.nc[latname].values, mask), roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        #tlon = xray.DataArray(np.ma.masked_array(self.nc[lonname][:],
                                                 #mask)).roll( nlon=roll )[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        #tlat = xray.DataArray(np.ma.masked_array(self.nc[latname][:],
                                                 #mask)).roll( nlon=roll )[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        tlon[tlon<0.] += 360.
        
        #Ny, Nx = tlon.shape
        #x = np.arange(0,Nx)
        #y = np.arange(0,Ny)
        #X,Y = np.meshgrid(x,y)
        #Zr = tlon.ravel()
        #Xr = np.ma.masked_array(X.ravel(), Zr.mask)
        #Yr = np.ma.masked_array(Y.ravel(), Zr.mask)
        #Xm = np.ma.masked_array( Xr.data, ~Xr.mask ).compressed()
        #Ym = np.ma.masked_array( Yr.data, ~Yr.mask ).compressed()
        #Zm = naiso.griddata(np.array([Xr.compressed(), Yr.compressed()]).T, 
        #                            Zr.compressed(), np.array([Xm,Ym]).T, method='linear')
        #Znew = Zr.data
        #Znew[Zr.mask] = Zm
        #Znew.shape = tlon.shape
        #tlon = Znew.copy()

        #Zr = tlat.ravel()
        #Xr = np.ma.masked_array(X.ravel(), Zr.mask)
        #Yr = np.ma.masked_array(Y.ravel(), Zr.mask)
        #Xm = np.ma.masked_array( Xr.data, ~Xr.mask ).compressed()
        #Ym = np.ma.masked_array( Yr.data, ~Yr.mask ).compressed()
        #Zm = naiso.griddata(np.array([Xr.compressed(), Yr.compressed()]).T, 
        #                            Zr.compressed(), np.array([Xm,Ym]).T, method='linear')
        #Znew = Zr.data
        #Znew[Zr.mask] = Zm
        #Znew.shape = tlat.shape
        #tlat = Znew.copy()

            
        #self.mask = self.nc.variables[maskname][:] <= 1

        # step 1: figure out the box indices
        #lonmask = (tlon >= lonrange[0]) & (tlon < lonrange[1])
        #latmask = (tlat >= latrange[0]) & (tlat < latrange[1])
        #boxidx = lonmask & latmask # this won't necessarily be square
        #irange = np.where(boxidx.sum(axis=0))[0]
        #imin, imax = irange.min(), irange.max()
        #jrange = np.where(boxidx.sum(axis=1))[0]
        #jmin, jmax = jrange.min(), jrange.max()
        imax = xmax
        imin = xmin
        jmax = ymax
        jmin = ymin
        Nx = imax - imin
        Ny = jmax - jmin
        lon = tlon[jmin:jmax, imin:imax]
        lat = tlat[jmin:jmax, imin:imax]
        #dlon_domain = np.roll(tlon,1)-np.roll(tlon, -1)
        #dlat_domain = np.roll(tlat,1)-np.roll(tlat, -1)
        #dx = 1e-2*np.roll(self.nc.variables[dxname][:], roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        #dy = 1e-2*np.roll(self.nc.variables[dyname][:], roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        #dx = 1e-2*np.roll(self.nc[dxname].values, roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        #dy = 1e-2*np.roll(self.nc[dyname].values, roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        dx = 1e-2 * self.nc[dxname][:].roll( nlon=roll ).values[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        dy = 1e-2 * self.nc[dyname][:].roll( nlon=roll ).values[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]

        # step 2: load the data
        #T = np.roll(self.nc.variables[varname],-1000)[..., jmin:jmax, imin:imax]
        if varname=='T_diss'  or varname=='T_forc':
            #T = np.roll(self.nc.variables[varname][:], roll, axis=2)[:,jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
            T = np.roll(self.nc[varname].values, roll, axis=2)[:,jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
            P = np.roll(self.nc['SST'].values, roll, axis=2)[:,jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        elif varname=='S_diss' or varname=='S_forc':
            T = np.roll(self.nc[varname].values, roll, axis=2)[:,jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
            P = np.roll(self.nc['SSS'].values, roll, axis=2)[:,jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        else:
            errstr = 'You have input an incompatible variable.'
            raise ValueError(errstr)
            #warnings.warn(errstr)
            #sys.exit(0)

        # step 3: figure out if there is too much land in the box
        #MAX_LAND = 0.01 # only allow up to 1% of land
        mask_domain = ( np.roll( mask, roll, 
                                axis=1 )[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100] + np.isnan(T)[0] )
        #mask_domain = mask.roll( nlon=roll )[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        region_mask = mask_domain[jmin:jmax, imin:imax]
        land_fraction = region_mask.sum().astype('f8') / (Ny*Nx)
        if land_fraction == 0.:
            # no problem
            pass
        elif land_fraction >= MAX_LAND:
            crit = 'false'
            errstr = 'The sector has too much land. land_fraction = ' + str(land_fraction)
            warnings.warn(errstr)
            #raise ValueError('The sector has too much land. land_fraction = ' + str(land_fraction))
        else:    
            # do some interpolation
            errstr = 'The sector has land (land_fraction=%g) but we are interpolating it out.' % land_fraction
            warnings.warn(errstr)
        
        # step 4: figure out FFT parameters (k, l, etc.) and set up result variable
        #dlon = lon[np.round(np.floor(lon.shape[0]*0.5)), np.round(
        #             np.floor(lon.shape[1]*0.5))+1]-lon[np.round(
        #             np.floor(lon.shape[0]*0.5)), np.round(np.floor(lon.shape[1]*0.5))]
        #dlat = lat[np.round(np.floor(lat.shape[0]*0.5))+1, np.round(
        #             np.floor(lat.shape[1]*0.5))]-lat[np.round(
        #             np.floor(lat.shape[0]*0.5)), np.round(np.floor(lat.shape[1]*0.5))]

        # Spatial step
        #dx = gfd.A*np.cos(np.radians(lat[np.round(
        #             np.floor(lat.shape[0]*0.5)),np.round(
        #             np.floor(lat.shape[1]*0.5))]))*np.radians(dlon)
        #dy = gfd.A*np.radians(dlat)
        
        # Wavenumber step
        dx_domain = dx[jmin:jmax,imin:imax].copy()
        dy_domain = dy[jmin:jmax,imin:imax].copy()
        # PREVIOUS
        #k = 2*np.pi*fft.fftshift(fft.fftfreq(Nx, dx_domain[Ny/2,Nx/2]))
        #l = 2*np.pi*fft.fftshift(fft.fftfreq(Ny, dy_domain[Ny/2,Nx/2]))
        #dk = np.diff(k)[0]*.5/np.pi
        #dl = np.diff(l)[0]*.5/np.pi
        k = fft.fftshift(fft.fftfreq(Nx, dx_domain[Ny/2,Nx/2]))
        l = fft.fftshift(fft.fftfreq(Ny, dy_domain[Ny/2,Nx/2]))
        dk = np.diff(k)[0]
        dl = np.diff(l)[0]

        ###################################
        ###  Start looping through each time step  ####
        ###################################
        Nt = T.shape[0]
        Decor_lag = 13
        tilde2_sum = np.zeros((Ny, Nx))
        Days = np.arange(0,Nt,Decor_lag)
        Neff = len(Days)
        for n in Days:
            Ti = np.ma.masked_array(T[n, jmin:jmax, imin:imax].copy(), region_mask)
            Pi = np.ma.masked_array(P[n, jmin:jmax, imin:imax].copy(), region_mask)
            
            # step 5: interpolate the missing data (only if necessary)
            if land_fraction>0. and land_fraction<MAX_LAND:
                x = np.arange(0,Nx)
                y = np.arange(0,Ny)
                X,Y = np.meshgrid(x,y)
                # for the tendency terms
                Zr = Ti.ravel()
                Xr = np.ma.masked_array(X.ravel(), Zr.mask)
                Yr = np.ma.masked_array(Y.ravel(), Zr.mask)
                Xm = np.ma.masked_array( Xr.data, ~Xr.mask ).compressed()
                Ym = np.ma.masked_array( Yr.data, ~Yr.mask ).compressed()
                Zm = naiso.griddata(np.array([Xr.compressed(), Yr.compressed()]).T, 
                                    Zr.compressed(), np.array([Xm,Ym]).T, method='nearest')
                Znew = Zr.data
                Znew[Zr.mask] = Zm
                Znew.shape = Ti.shape
                Ti = Znew
                # for the tracer field
                Zr = Pi.ravel()
                Xr = np.ma.masked_array(X.ravel(), Zr.mask)
                Yr = np.ma.masked_array(Y.ravel(), Zr.mask)
                Xm = np.ma.masked_array( Xr.data, ~Xr.mask ).compressed()
                Ym = np.ma.masked_array( Yr.data, ~Yr.mask ).compressed()
                Zm = naiso.griddata(np.array([Xr.compressed(), Yr.compressed()]).T, 
                                    Zr.compressed(), np.array([Xm,Ym]).T, method='nearest')
                Znew = Zr.data
                Znew[Zr.mask] = Zm
                Znew.shape = Pi.shape
                Pi = Znew
            elif land_fraction==0.:
            # no problem
                pass
            else:
                break
        
            # step 6: detrend the data in two dimensions (least squares plane fit)
            # for the tendency term
            d_obs = np.reshape(Ti, (Nx*Ny,1))
            G = np.ones((Ny*Nx,3))
            for i in range(Ny):
                G[Nx*i:Nx*i+Nx, 0] = i+1
                G[Nx*i:Nx*i+Nx, 1] = np.arange(1, Nx+1)    
            m_est = np.dot(np.dot(lin.inv(np.dot(G.T, G)), G.T), d_obs)
            d_est = np.dot(G, m_est)
            Lin_trend = np.reshape(d_est, (Ny, Nx))
            Ti -= Lin_trend
            # for the tracer field
            d_obs = np.reshape(Pi, (Nx*Ny,1))
            G = np.ones((Ny*Nx,3))
            for i in range(Ny):
                G[Nx*i:Nx*i+Nx, 0] = i+1
                G[Nx*i:Nx*i+Nx, 1] = np.arange(1, Nx+1)    
            m_est = np.dot(np.dot(lin.inv(np.dot(G.T, G)), G.T), d_obs)
            d_est = np.dot(G, m_est)
            Lin_trend = np.reshape(d_est, (Ny, Nx))
            Pi -= Lin_trend

            # step 7: window the data
            # Hanning window
            windowx = sig.hann(Nx)
            windowy = sig.hann(Ny)
            window = windowx*windowy[:,np.newaxis] 
            Ti *= window
            Pi *= window

            # step 8: do the FFT for each timestep and aggregate the results
            Tif = fft.fftshift(fft.fft2(Ti))    # [u^2] (u: unit)
            Pif = fft.fftshift(fft.fft2(Pi)) 
            tilde2_sum += np.real(Tif*np.conj(Pif))

        # step 9: check whether the Plancherel theorem is satisfied
        #tilde2_ave = tilde2_sum/Nt
        breve2_sum = tilde2_sum / ((Nx*Ny)**2*dk*dl)
        #breve2_ave = tilde2_ave/((Nx*Ny)**2*dk*dl)
        #spac2_ave = Ti2_sum/Nt
        #if land_fraction == 0.:
            #np.testing.assert_almost_equal(breve2_ave.sum()/(dx_domain[Ny/2,Nx/2]*dy_domain[Ny/2,Nx/2]*(spac2_ave).sum()), 1., decimal=5)
            #np.testing.assert_almost_equal( breve2_sum.sum() / ( dx_domain[Ny/2, Nx/2] * dy_domain[Ny/2, Nx/2] * (Ti2_sum).sum() ), 1., decimal=5)
            
        # step 10: derive the isotropic spectrum
        kk, ll = np.meshgrid(k, l)
        K = np.sqrt( kk**2 + ll**2 )
        #Ki = np.linspace(0, k.max(), nbins)
        if k.max() > l.max():
            Ki = np.linspace(0, l.max(), nbins)
        else:
            Ki = np.linspace(0, k.max(), nbins)
        #Ki = np.linspace(0, K.max(), nbins)
        deltaKi = np.diff(Ki)[0]
        Kidx = np.digitize(K.ravel(), Ki)
        invalid = Kidx[-1]
        area = np.bincount(Kidx)
        #PREVIOUS: isotropic_PSD = np.ma.masked_invalid(
        #                               np.bincount(Kidx, weights=breve2_sum.ravel()) / area )[:-1] *Ki*2.*np.pi**2
        isotropic_PSD = np.ma.masked_invalid(
                                       np.bincount(Kidx, weights=breve2_sum.ravel()) / area )[:-1] * Ki
        #isotropic_PSD = np.ma.masked_invalid(
                                       #np.bincount(Kidx, weights=(breve2_ave).ravel()) / area )[1:] *Ki*2.*np.pi**2
        #isotropic_PSD = np.ma.masked_invalid(
        #                              np.bincount(Kidx, weights=(2.*dx*dy/Nx/Ny*PSD_ave*K*2.*np.pi).ravel()) / area )
        #isotropic_PSD = np.ma.masked_invalid(
        #                               np.bincount(Kidx, weights=(PSD_ave).ravel()) / area )[1:] *2.*np.pi/deltaKi
        
        # Usage of digitize
        #>>> x = np.array([-0.2, 6.4, 3.0, 1.6, 20.])
        #>>> bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        #>>> inds = np.digitize(x, bins)
        #array([0, 4, 3, 2, 5])
        
        # Usage of bincount 
        #>>> np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
        #array([1, 3, 1, 1, 0, 0, 0, 1])
        # With the option weight
        #>>> w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6]) # weights
        #>>> x = np.array([0, 1, 1, 2, 2, 2])
        #>>> np.bincount(x,  weights=w)
        #array([ 0.3,  0.7,  1.1])  <- [0.3, 0.5+0.2, 0.7+1.-0.6]
        
        # step 10: return the results
        #PREVIOUS: return Neff, Nt, Nx, Ny, k, l, Ti2_sum, tilde2_sum, breve2_sum, Ki, isotropic_PSD, area[1:], lon, lat, land_fraction, MAX_LAND
        return Neff, Nt, Nx, Ny, k, l, tilde2_sum, breve2_sum, Ki[:], isotropic_PSD[:], area[1:-1], lon, lat, land_fraction, MAX_LAND
        
    def structure_function(self, varname='SST', lonname='TLONG', latname='TLAT', maskname='KMT', dxname='DXT', dyname='DYT', lonrange=(154.9,171.7), latrange=(30,45.4), roll=-1000, q=2, MAX_LAND=0.01, xmin=0, xmax=0, ymin=0, ymax=0, ymin_bound=0, ymax_bound=0, xmin_bound=0, xmax_bound=0, detre=True, windw=True, iso=False, roll_param=True):
        """Calculate a structure function of Matlab variable 'varname'
           in the box defined by lonrange and latrange.
        """

        jmin_bound = ymin_bound
        jmax_bound = ymax_bound
        imin_bound = xmin_bound
        imax_bound = xmax_bound
        mask = self.nc.variables[maskname][:] <= 1
        tlon = np.roll(np.ma.masked_array(self.nc.variables[lonname][:],mask), roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        tlat = np.roll(np.ma.masked_array(self.nc.variables[latname][:],mask), roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        tlon[tlon<0.] += 360.

        # step 1: figure out the box indices
        #lonmask = (tlon >= lonrange[0]) & (tlon < lonrange[1])
        #latmask = (tlat >= latrange[0]) & (tlat < latrange[1])
        #boxidx = lonmask & latmask # this won't necessarily be square
        #irange = np.where(boxidx.sum(axis=0))[0]
        #imin, imax = irange.min(), irange.max()
        #jrange = np.where(boxidx.sum(axis=1))[0]
        #jmin, jmax = jrange.min(), jrange.max()
        #Nx = imax - imin
        #Ny = jmax - jmin
        imax = xmax
        imin = xmin
        jmax = ymax
        jmin = ymin
        Nx = imax - imin
        Ny = jmax - jmin
        lon = tlon[jmin:jmax, imin:imax]
        lat = tlat[jmin:jmax, imin:imax]
        
        # Difference of latitude and longitude
        #dlon = lon[np.round(np.floor(lon.shape[0]*0.5)), np.round(
        #             np.floor(lon.shape[1]*0.5))+1]-lon[np.round(
        #             np.floor(lon.shape[0]*0.5)), np.round(np.floor(lon.shape[1]*0.5))]
        #dlat = lat[np.round(np.floor(lat.shape[0]*0.5))+1, np.round(
        #             np.floor(lat.shape[1]*0.5))]-lat[np.round(
        #             np.floor(lat.shape[0]*0.5)), np.round(np.floor(lat.shape[1]*0.5))]

        # Spatial step
        #dx = gfd.A*np.cos(np.pi/180*lat[np.round(
        #             np.floor(lat.shape[0]*0.5)),np.round(
        #             np.floor(lat.shape[1]*0.5))])*np.radians(dlon)
        #dy = gfd.A*np.radians(dlat)
        dx = 1e-2*np.roll(self.nc.variables[dxname][:], roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        dy = 1e-2*np.roll(self.nc.variables[dyname][:], roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]

        # load data
        if varname=='SST':
            T = np.roll(self.nc.variables[varname][:], roll, axis=2)[:,jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        else:
            T = 1e-2*np.roll(self.nc.variables[varname][:], roll, axis=2)[:,jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        
        # define variables
        Nt = T.shape[0]
        n = np.arange(0,np.log2(Nx/2), dtype='i4')
        ndel = len(n)
        dx_domain = dx[jmin:jmax,imin:imax].copy()
        dy_domain = dy[jmin:jmax,imin:imax].copy()
        L = 2**n*dx_domain[dx_domain.shape[0]/2,dx_domain.shape[1]/2]
        Hi = np.zeros(ndel)
        Hj = np.zeros(ndel)
        sumcounti = np.zeros(ndel)
        sumcountj = np.zeros(ndel)

        # Figure out if there is too much land in the box
        #MAX_LAND = 0.01 # only allow up to 1% of land
        mask_domain = np.roll(mask, roll, axis=1)[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        region_mask = mask_domain[jmin:jmax, imin:imax]
        land_fraction = region_mask.sum().astype('f8') / (Ny*Nx)
        if land_fraction >= MAX_LAND:
            #raise ValueError('The sector has too much land. land_fraction = ' + str(land_fraction))
            errstr = 'The sector has land (land_fraction=%g).' % land_fraction
            warn(errstr)
        else:
            # no problem
            pass
        #else:
            # do some interpolation
            #errstr = 'The sector has land (land_fraction=%g) but we are interpolating it out.' % land_fraction
            #warn(errstr)
            # have to figure out how to actually do it
            # Ti = ...

        ###################################
        ###  Start looping through each time step  ####
        ###################################
        for n in range(Nt):
            Ti = np.ma.masked_array(T[n, jmin:jmax, imin:imax].copy(), region_mask)
            
            if land_fraction<MAX_LAND:
                pass
            else:
                break
           
            # Interpolate the missing data (only if necessary)
            #if land_fraction>0. and land_fraction<MAX_LAND:
                #x = np.arange(0,Nx)
                #y = np.arange(0,Ny)
                #X,Y = np.meshgrid(x,y)
                #Zr = Ti.ravel()
                #Xr = np.ma.masked_array(X.ravel(), Zr.mask)
                #Yr = np.ma.masked_array(Y.ravel(), Zr.mask)
                #Xm = np.ma.masked_array( Xr.data, ~Xr.mask ).compressed()
                #Ym = np.ma.masked_array( Yr.data, ~Yr.mask ).compressed()
                #Zm = griddata(np.array([Xr.compressed(), Yr.compressed()]).T, 
                #                        Zr.compressed(), np.array([Xm,Ym]).T, method='nearest')
                #Znew = Zr.data
                #Znew[Zr.mask] = Zm
                #Znew.shape = Z.shape
                #Ti = Znew

            # step 5: interpolate the missing data (only if necessary)
            if land_fraction>0. and land_fraction<MAX_LAND:
                x = np.arange(0,Nx)
                y = np.arange(0,Ny)
                X,Y = np.meshgrid(x,y)
                Zr = Ti.ravel()
                Xr = np.ma.masked_array(X.ravel(), Zr.mask)
                Yr = np.ma.masked_array(Y.ravel(), Zr.mask)
                Xm = np.ma.masked_array( Xr.data, ~Xr.mask ).compressed()
                Ym = np.ma.masked_array( Yr.data, ~Yr.mask ).compressed()
                Zm = naiso.griddata(np.array([Xr.compressed(), Yr.compressed()]).T, 
                                    Zr.compressed(), np.array([Xm,Ym]).T, method='nearest')
                Znew = Zr.data
                Znew[Zr.mask] = Zm
                Znew.shape = Ti.shape
                Ti = Znew
            elif land_fraction==0.:
            # no problem
                pass
            else:
                break
        
            # step 6: detrend the data in two dimensions (least squares plane fit)
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
            # Hanning window
            windowx = sig.hann(Nx)
            windowy = sig.hann(Ny)
            window = windowx*windowy[:,np.newaxis]
            Ti *= window
            
            Ti = np.ma.masked_array(Ti.copy(), region_mask)

            if iso:
            # Calculate structure functions isotropically
                #print 'Isotropic Structure Function'
                angle = np.arange(0,2.*np.pi,np.pi/180.)
                radi = np.arange(0,Nx/2,1)
                ang_index = len(angle)
                rad_index = len(radi)
                polar_coodx = np.zeros((ang_index,rad_index))
                polar_coody = np.zeros_like(polar_coodx)
                for j in range(ang_index):
                    for i in range(rad_index):
                        polar_coodx[j,i] = radi[j,i]*np.cos(angle[j,i])
                        polar_coody[j,i] = radi[j,i]*np.sin(angle[j,i])
                Sq = np.zeros((ang_index/2,Nx))
                    # Unfinished script (This option does not work)
            else:
            # Calculate structure functions along each x-y axis
                #print 'Anisotropic Structure Function'
                for m in range(ndel):
                    dx_cen = dx_domain[dx_domain.shape[0]/2,dx_domain.shape[1]/2]
                    dy_cen = dy_domain[dy_domain.shape[0]/2,dy_domain.shape[1]/2]
                    if roll_param:
                    # Use roll function
                        if dx_cen<dy_cen:
                            dSSTi = np.abs(Ti - np.roll(Ti,2**m*int(round(dy_cen/dx_cen)),axis=1))**q
                        else:
                            dSSTi = np.abs(Ti - np.roll(Ti,2**m,axis=1))**q
                        dSSTj = np.abs(Ti - np.roll(Ti,2**m,axis=0))**q
                        Hi[m] += dSSTi.mean()
                        Hj[m] += dSSTj.mean()
                    else:
                        # Take the difference by displacing the matrices along each axis 
                        #for i in range(Nx):
                        if dx_cen<dy_cen:
                            dSSTi = np.ma.masked_array((np.absolute(Ti[:,2**m*int(round(dy_cen/dx_cen)):] - Ti[:,:-2**m*int(round(dy_cen/dx_cen))]))**2) 
                        else:
                            dSSTi = np.ma.masked_array((np.absolute(Ti[:,2**m:] - Ti[:,:-2**m]))**2) 
                        dSSTj = np.ma.masked_array((np.absolute(Ti[2**m:] - Ti[:-2**m]))**2) # .filled(0.)
                        # Take the difference by specifying the grid spacing
                        #dSSTi = np.empty((Ny-1,Nx))*np.nan
                        #dSSTj = np.empty((Ny,Nx-1))*np.nan
                        #for i in range(Nx):
                        #    if i+2**m<Nx:
                        #        if dx_cen<dy_cen:
                        #            dSSTi[:,:-2**m*int(round(dy_cen/dx_cen))] = np.ma.masked_array(
                        #                               (np.absolute(Ti[:,2**m*int(round(dy_cen/dx_cen)):] - Ti[:,:-2**m*int(round(dy_cen/dx_cen))]))**q)
                        #        else:
                        #            dSSTi[:,:-2**m] = np.ma.masked_array(
                        #                               (np.absolute(Ti[:,2**m:] - Ti[:,:-2**m]))**q)
                            #else:
                            #    dSSTi[:,i] = np.ma.masked_array(
                            #                           (np.absolute(Ti[:,i+2**m-Nx] - Ti[:,i]))**q)
                        #    if j+2**m<Ny:
                        #        dSSTj[:-2**m,:] = np.ma.masked_array((np.absolute(Ti[2**m:] - Ti[:-2**m]))**2)
                        counti = (~dSSTi.mask).astype('i4')
                        countj = (~dSSTj.mask).astype('i4')
                        sumcounti[m] = np.sum(counti)
                        sumcountj[m] = np.sum(countj)
                        Hi[m] = np.sum(np.absolute(dSSTi))/sumcounti[m]
                        Hj[m] = np.sum(np.absolute(dSSTj))/sumcountj[m]
                        Hi[m] = np.sum(dSSTi)/(Ny-1)
                        Hj[m] = np.sum(dSSTj)/(Nx-1)                    

        return Nt, dx_cen, dy_cen, L, Hi, Hj, lon, lat, land_fraction, MAX_LAND
    
def interpolate_2d(Ti):
    """Interpolate a 2D field
    """
    Ny, Nx = Ti.shape
    x = np.arange(0,Nx)
    y = np.arange(0,Ny)
    X,Y = np.meshgrid(x,y)
    Zr = Ti.ravel()
    Xr = np.ma.masked_array(X.ravel(), Zr.mask)
    Yr = np.ma.masked_array(Y.ravel(), Zr.mask)
    Xm = np.ma.masked_array( Xr.data, ~Xr.mask ).compressed()
    Ym = np.ma.masked_array( Yr.data, ~Yr.mask ).compressed()
    Zm = naiso.griddata(np.array([Xr.compressed(), Yr.compressed()]).T, 
                                    Zr.compressed(), np.array([Xm,Ym]).T, method='nearest')
    Znew = Zr.data
    Znew[Zr.mask] = Zm
    Znew.shape = Ti.shape
    
    return Znew
 
def detrend_2d(Ti):
    """Detrend a 2D field.
        Linear plane fit.
    """
    Ny, Nx = Ti.shape
    d_obs = np.reshape(Ti, (Nx*Ny,1))
    G = np.ones((Ny*Nx,3))
    for i in range(Ny):
        G[Nx*i:Nx*i+Nx, 0] = i+1
        G[Nx*i:Nx*i+Nx, 1] = np.arange(1, Nx+1)    
    m_est = np.dot(np.dot(lin.inv(np.dot(G.T, G)), G.T), d_obs)
    d_est = np.dot(G, m_est)
    Lin_trend = np.reshape(d_est, (Ny, Nx))
    Ti -= Lin_trend
    
    return Ti


    
    
#cp_sw = 3.996e7
#rho_sw = 4.1/3.996
#hflux_factor = 1000.0/(rho_sw*cp_sw) / 100.

class EOSCalculator(object):
    def __init__(self, parent, p=0., hmax=None, hconst=None):
        self.p = p
        self.hmax = hmax
        self.hconst = hconst
        self.nc = parent.nc
        self.parent = parent
        #if self.nc.variables.has_key('SHF'):
        if 'SHF' in self.nc.keys():
            self.hfname = 'SHF'
            self.fwfname = 'SFWF'
            self.mlname = 'HMXL'
        else:
            self.hfname = 'SHF_2'        #"watt/m^2"
            self.fwfname = 'SFWF_2'     #"kg/m^2/s
            self.mlname = 'HMXL_2'      #"cm"




# from forcing.F90
# !-----------------------------------------------------------------------
# !
# !  convert fresh water flux (kg/m^2/s) to virtual salt flux (msu*cm/s):
# !  --------------------------------------------------------------------
# !    ocean reference salinity in (o/oo=psu)
# !    density of freshwater rho_fw = 1.0 (g/cm^3)
# !    h2o_flux in (kg/m^2/s) = 0.1 (g/cm^2/s)
# !
# !    salt_flux            = - h2o_flux * ocn_ref_salinity / rho_fw
# !    salt_flux (msu*cm/s) = - h2o_flux (kg/m^2/s)
# !                           * ocn_ref_salinity (psu)
# !                           * 1.e-3 (msu/psu)
# !                           * 0.1 (g/cm^2/s)/(kg/m^2/s)
# !                           / 1.0 (g/cm^3)
# !                         = - h2o_flux (kg/m^2/s)
# !                           * ocn_ref_salinity (psu)
# !                           * fwflux_factor (cm/s)(msu/psu)/(kg/m^2/s)
# !
# !    ==>  fwflux_factor = 1.e-4
# !
# !    salt_flux(msu*cm/s) = h2oflux(kg/m^2/s) * salinity_factor[(msu cm/s) / (kg/m^2/s)]
# !
# !    ==> salinity_factor = - ocn_ref_salinity(psu) * fwflux_factor
# !
# !    ==> salt_flux [psu*cm/s] = h2oflux * salinity_factor * 1e3 
# !
# !-----------------------------------------------------------------------
#
ocn_ref_salinity = 34.7
# using PSU, kg, m as units
#fwflux_factor = 1e-3
rho_fw = 1e3
fwflux_factor = 1e-4
#fwflux_factor = 1.  
salinity_factor = - ocn_ref_salinity * fwflux_factor * 1e3 * 1e-2   # [(psu*m/s) / (kg/m^2/s)]

# !-----------------------------------------------------------------------
# !
# !  convert heat, solar flux (W/m^2) to temperature flux (C*cm/s):
# !  --------------------------------------------------------------
# !    heat_flux in (W/m^2) = (J/s/m^2) = 1000(g/s^3)
# !    density of seawater rho_sw in (g/cm^3)
# !    specific heat of seawater cp_sw in (erg/g/C) = (cm^2/s^2/C)
# !
# !    temp_flux             = heat_flux / (rho_sw*cp_sw)
# !    temp_flux (C*cm/s) = heat_flux (W/m^2)
# !                         * 1000 (g/s^3)/(W/m^2)
# !                         / [(rho_sw*cp_sw) (g/cm/s^2/C)]
# !
# !                             = heat_flux (W/m^2)
# !                         * hflux_factor (C*cm/s)/(W/m^2)
# !
# !    ==>  hflux_factor    = 1000 / (rho_sw*cp_sw)  [(C cm/s) / (W/m^2)]
# !
# !    ==>  temp_flux      = heat_flux * hflux_factor / 100  [C m/s]
# !
# !-----------------------------------------------------------------------

cp_sw = 3.996e7
rho_sw = 4.1/3.996
hflux_factor = 1e3 / (rho_sw*cp_sw) / 1e2

def get_surface_ts(nc, i):
    try:
        #S0 = nc.variables['SSS'].__getitem__(i)
        #T0 = nc.variables['SST'].__getitem__(i)
        S0 = nc['SSS'].values.__getitem__(i)
        T0 = nc['SST'].values.__getitem__(i)
    except KeyError:
        #S0 = nc.variables['SALT'][:,0,:,:].__getitem__(i)
        #T0 = nc.variables['TEMP'][:,0,:,:].__getitem__(i) 
        S0 = nc['SALT'][:, 0, :, :].values.__getitem__(i)
        T0 = nc['TEMP'][:, 0, :, :].values.__getitem__(i) 
    return T0, S0

class TSForcing(EOSCalculator):
    def __getitem__(self, i):
        T0, S0 = get_surface_ts(self.nc, i)
        #Ffw = self.nc.variables[self.fwfname].__getitem__(i)
        #Qhf = self.nc.variables[self.hfname].__getitem__(i)
        Ffw = self.nc[self.fwfname].values.__getitem__(i)     #Fresh Water flux "kg/m^2/s"
        Qhf = self.nc[self.hfname].values.__getitem__(i)       #Surface Heat flux "watt/m^2" 

        if self.hconst is not None:
            H_ml = self.hconst
        else:
            #H_ml = self.nc.variables[self.mlname].__getitem__(i)/100.
            H_ml = self.nc[self.mlname].__getitem__(i) / 100.
            if self.hmax is not None:
                H_ml = np.ma.masked_greater(H_ml, self.hmax).filled(self.hmax)
        
        FT_forc = hflux_factor * Qhf
        FS_forc = salinity_factor * Ffw
        FT_mix = H_ml * self.parent.biharmonic_tendency(T0)
        FS_mix = H_ml * self.parent.biharmonic_tendency(S0)
        
        return [ np.ma.masked_array(F, self.parent.mask) 
                 for F in [T0, S0, FT_forc, FT_mix, FS_forc, FS_mix] ]  
        
        
        

