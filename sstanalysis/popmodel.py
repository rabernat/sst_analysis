import numpy as np
import netCDF4
import warnings
from warnings import warn
from scipy import linalg as lin
from scipy import signal as sig
from scipy import fftpack as fft
from scipy import interpolate as naiso
import gfd

class POPFile(object):
    
    def __init__(self, fname, areaname='TAREA', maskname='KMT', ah=-3e17, is3d=False):
        """Wrapper for POP model netCDF files"""
        self.nc = netCDF4.Dataset(fname)
        self.Ny, self.Nx = self.nc.variables[areaname].shape     
        self._ah = ah
        
        # mask
        self.mask = self.nc.variables[maskname][:] <= 1

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

    def mask_field(self, maskname='KMT', varname='SST'):
        """Apply mask to tracer field T"""
        mask = self.nc.variables[maskname][:]
        T = self.nc.variables[varname][:]
        return np.ma.masked_array(T, mask<=1)
        
    def initialize_gradient_operator(self, areaname='TAREA'):
        """Needs to be called before calculating gradients"""
        # raw grid geometry
        work1 = (self.nc.variables['HTN'][:] /
                 self.nc.variables['HUW'][:])
        tarea = self.nc.variables[areaname][:]
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
    
    def gradient_modulus(self, varname='SST', lonname='ULONG', latname='ULAT', maskname='KMT', dxname='DXU', dyname='DYU', lonrange=(154.9,171.7), latrange=(30,45.4), roll=-1000):
        """Return the modulus of the gradient of tracer at U points."""
        
        tlon = np.roll(self.nc.variables[lonname][:], roll, axis=1)
        tlat = np.roll(self.nc.variables[latname][:], roll, axis=1)
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
        dxdbarTy = (barTy - np.roll(barTy,1,axis=2))/dx
        dydbarTx = (barTx - np.roll(barTx,1,axis=1))/dy
        #dTx = np.roll(dTx, roll)[:, jmin:jmax, imin:imax]
        #dTy = np.roll(dTy, roll)[:, jmin:jmax, imin:imax]
        
        return lon, lat, dxdbarTy[:,jmin:jmax,imin:imax], dydbarTx[:,jmin:jmax,imin:imax]
        
    def mag_gradient_modulus(self, varname='SST', lonname='TLONG', latname='TLAT', maskname='KMT', lonrange=(154.9,171.7), latrange=(30,45.4), roll=-1000):
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
            T = self.nc.variables[varname][:]
        else:
            #T = 1e-2*np.roll(self.nc.variables[varname][:], roll)[..., jmin:jmax, imin:imax]
            T = 1e-2*(self.nc.variables[varname][:])
        
        # step 3: calculate the difference
        dTx = self._kmaske * (np.roll(T,-1,axis=0) - T)
        dTy = self._kmaskn * (np.roll(T,-1,axis=1) - T)
        #dTx = np.roll(dTx, roll)[:, jmin:jmax, imin:imax]
        #dTy = np.roll(dTy, roll)[:, jmin:jmax, imin:imax]
        
        return jmin, jmax, imin, imax, roll, lon, lat, np.sqrt( 0.5 *
                    (dTx**2 + np.roll(dTx,1,axis=0)**2) * self._dxtr**2
                    + .5 * (dTy**2 + np.roll(dTy,1,axis=1)**2) * self._dytr**2 )   
    
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
        
    def power_spectrum_2d(self, varname='SST', lonname='TLONG', latname='TLAT', maskname='KMT', geos=False, lonrange=(154.9,171.7), latrange=(30,45.4), roll=-1000, nbins=256, MAX_LAND=0.01):
        """Calculate a two-dimensional power spectrum of netcdf variable 'varname'
           in the box defined by lonrange and latrange.
        """
    
        tlon = np.roll(self.nc.variables[lonname][:], roll)
        tlat = np.roll(self.nc.variables[latname][:], roll)
        #self.mask = self.nc.variables[maskname][:] <= 1

        # step 1: figure out the box indices
        lonmask = (tlon >= lonrange[0]) & (tlon < lonrange[1])
        latmask = (tlat >= latrange[0]) & (tlat < latrange[1])
        boxidx = lonmask & latmask # this won't necessarily be square
        irange = np.where(boxidx.sum(axis=0))[0]
        imin, imax = irange.min(), irange.max()
        jrange = np.where(boxidx.sum(axis=1))[0]
        jmin, jmax = jrange.min(), jrange.max()
        Nx = imax - imin
        Ny = jmax - jmin
        lon = tlon[jmin:jmax, imin:imax]
        lat = tlat[jmin:jmax, imin:imax]
        dlon_domain = np.roll(tlon,1)-np.roll(tlon, -1)
        dlat_domain = np.roll(tlat,1)-np.roll(tlat, -1)

        # step 2: load the data
        #T = np.roll(self.nc.variables[varname],-1000)[..., jmin:jmax, imin:imax]
        if varname=='SST':
            T = np.roll(self.nc.variables[varname][:], roll)[..., jmin:jmax, imin:imax]
        elif varname=='SSH' and geos:
            T = 1e-2*np.roll(self.nc.variables[varname][:], roll)[..., jmin:jmax, imin:imax]
            T = gfd.g/gfd.f_coriolis(lat)*(np.roll(T,1)-np.roll(T,-1))/(gfd.A*np.cos(np.radians(dlat_domain))*np.radians(dlon_domain))
        else:
            T = 1e-2*np.roll(self.nc.variables[varname][:], roll)[..., jmin:jmax, imin:imax]

        # step 3: figure out if there is too much land in the box
        #MAX_LAND = 0.01 # only allow up to 1% of land
        mask = self.nc.variables[maskname][:] <= 1
        region_mask = np.roll(mask, roll)[jmin:jmax, imin:imax]
        land_fraction = region_mask.sum().astype('f8') / (Ny*Nx)
        if land_fraction==0.:
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
        dlon = lon[np.round(np.floor(lon.shape[0]*0.5)), np.round(
                     np.floor(lon.shape[1]*0.5))+1]-lon[np.round(
                     np.floor(lon.shape[0]*0.5)), np.round(np.floor(lon.shape[1]*0.5))]
        dlat = lat[np.round(np.floor(lat.shape[0]*0.5))+1, np.round(
                     np.floor(lat.shape[1]*0.5))]-lat[np.round(
                     np.floor(lat.shape[0]*0.5)), np.round(np.floor(lat.shape[1]*0.5))]

        # Spatial step
        dx = gfd.A*np.cos(np.radians(lat[np.round(
                     np.floor(lat.shape[0]*0.5)),np.round(
                     np.floor(lat.shape[1]*0.5))]))*np.radians(dlon)
        dy = gfd.A*np.radians(dlat)
        
        # Wavenumber step
        k = fft.fftshift(fft.fftfreq(Nx, dx))
        l = fft.fftshift(fft.fftfreq(Ny, dy))
        dk = np.diff(k)[0]
        dl = np.diff(l)[0]

        ###################################
        ###  Start looping through each time step  ####
        ###################################
        Nt = T.shape[0]
        tilde2_sum = np.zeros((Ny,Nx))
        Ti2_sum = np.zeros((Ny,Nx))
        for n in range(Nt):
            Ti = np.ma.masked_array(T[n], region_mask)
            
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
            
            Ti2_sum += Ti**2

            # step 8: do the FFT for each timestep and aggregate the results
            Tif = fft.fftshift(fft.fft2(Ti))    # [u^2] (u: unit)
            tilde2_sum += np.real(Tif*np.conj(Tif))

        # step 9: check whether the Plancherel theorem is satisfied
        tilde2_ave = tilde2_sum/Nt
        breve2_ave = tilde2_ave/((Nx*Ny)**2*dk*dl)
        spac2_ave = Ti2_sum/Nt
        np.testing.assert_almost_equal(breve2_ave.sum()/(dx*dy*(spac2_ave).sum()), 1., decimal=5)
        
        # step 10: derive the isotropic spectrum
        kk, ll = np.meshgrid(k, l)
        K = np.sqrt(kk**2 + ll**2)
        #Ki = np.linspace(0, k.max(), nbins)
        Ki = np.linspace(0, K.max(), nbins)
        deltaKi = np.diff(Ki)[0]
        Kidx = np.digitize(K.ravel(), Ki)
        area = np.bincount(Kidx)
        isotropic_PSD = np.ma.masked_invalid(
                                       np.bincount(Kidx, weights=(breve2_ave).ravel()) / area )[1:] *Ki*2.*np.pi**2
        #isotropic_PSD = np.ma.masked_invalid(
        #                              np.bincount(Kidx, weights=(2.*dx*dy/Nx/Ny*PSD_ave*K*2.*np.pi).ravel()) / area )
        #isotropic_PSD = np.ma.masked_invalid(
        #                               np.bincount(Kidx, weights=(PSD_ave).ravel()) / area )[1:] *2.*np.pi/deltaKi
        
        # step 10: return the results
        return Nt, Nx, Ny, k, l, spac2_ave, tilde2_ave, breve2_ave, Ki, isotropic_PSD, area[1:], lon, lat, land_fraction, MAX_LAND
        
    def structure_function(self, varname='SST', lonname='TLONG', latname='TLAT', maskname='KMT', lonrange=(154.9,171.7), latrange=(30,45.4), roll=-1000, q=2, detre=True, windw=True, iso=False):
        """Calculate a structure function of Matlab variable 'varname'
           in the box defined by lonrange and latrange.
        """

        tlon = np.roll(self.nc.variables[lonname][:], roll)
        tlat = np.roll(self.nc.variables[latname][:], roll)

        # step 1: figure out the box indices
        lonmask = (tlon >= lonrange[0]) & (tlon < lonrange[1])
        latmask = (tlat >= latrange[0]) & (tlat < latrange[1])
        boxidx = lonmask & latmask # this won't necessarily be square
        irange = np.where(boxidx.sum(axis=0))[0]
        imin, imax = irange.min(), irange.max()
        jrange = np.where(boxidx.sum(axis=1))[0]
        jmin, jmax = jrange.min(), jrange.max()
        #Nx = imax - imin
        #Ny = jmax - jmin
        lon = tlon[jmin:jmax, imin:imax]
        lat = tlat[jmin:jmax, imin:imax]
        
        # Difference of latitude and longitude
        dlon = lon[np.round(np.floor(lon.shape[0]*0.5)), np.round(
                     np.floor(lon.shape[1]*0.5))+1]-lon[np.round(
                     np.floor(lon.shape[0]*0.5)), np.round(np.floor(lon.shape[1]*0.5))]
        dlat = lat[np.round(np.floor(lat.shape[0]*0.5))+1, np.round(
                     np.floor(lat.shape[1]*0.5))]-lat[np.round(
                     np.floor(lat.shape[0]*0.5)), np.round(np.floor(lat.shape[1]*0.5))]

        # Spatial step
        dx = gfd.A*np.cos(np.pi/180*lat[np.round(
                     np.floor(lat.shape[0]*0.5)),np.round(
                     np.floor(lat.shape[1]*0.5))])*np.radians(dlon)
        dy = gfd.A*np.radians(dlat)

        # load data
        if varname=='SST':
            T = np.roll(self.nc.variables[varname][:], roll)[..., jmin:jmax, imin:imax]
        else:
            T = 1e-2*np.roll(self.nc.variables[varname][:], roll)[..., jmin:jmax, imin:imax]

        # define variables
        Nt, Ny, Nx = T.shape
        n = np.arange(0,np.log2(Nx/2), dtype='i4')
        ndel = len(n)
        L = 2**n
        Hi = np.zeros(ndel)
        Hj = np.zeros(ndel)
        sumcounti = np.zeros(ndel)
        sumcountj = np.zeros(ndel)

        # Figure out if there is too much land in the box
        MAX_LAND = 0.01 # only allow up to 1% of land
        mask = self.nc.variables[maskname][:] <= 1
        region_mask = np.roll(mask, roll)[jmin:jmax, imin:imax]
        land_fraction = region_mask.sum().astype('f8') / (Ny*Nx)
        if land_fraction==0.:
            # no problem
            pass
        elif land_fraction >= MAX_LAND:
            #raise ValueError('The sector has too much land. land_fraction = ' + str(land_fraction))
            errstr = 'The sector has land (land_fraction=%g).' % land_fraction
            warn(errstr)
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
            Ti = np.ma.masked_array(T[n], region_mask)
           
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

            # Detrend the data in two dimensions (least squares plane fit)
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

            # Difference with 2^m gridpoints in betweend
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
                    #dSSTi = np.zeros((Ny,Nx))
                    #dSSTj = np.zeros((Ny,Nx))
                    # Take the difference by displacing the matrices along each axis 
                    #dSSTi = np.ma.masked_array((np.absolute(Ti[:,2**m:] - Ti[:,:-2**m]))**2) # .filled(0.)
                    #dSSTj = np.ma.masked_array((np.absolute(Ti[2**m:] - Ti[:-2**m]))**2) # .filled(0.)
                    # Take the difference by specifying the grid spacing
                    #for i in range(Nx):
                    #    if i+2**m<Nx:
                    #        dSSTi[:,:Nx-2**m] = np.ma.masked_array(
                    #                               (np.absolute(Ti[:,2**m:] - Ti[:,:-2**m]))**q)
                    #    else:
                    #        dSSTi[:,i] = np.ma.masked_array(
                    #                               (np.absolute(Ti[:,i+2**m-Nx] - Ti[:,i]))**q)
                    # Use roll function
                    dSSTi = np.abs(Ti - np.roll(Ti,2**m,axis=1))**q
                    dSSTj = np.abs(Ti - np.roll(Ti,2**m,axis=0))**q
                    #counti = (~dSSTi.mask).astype('i4')
                    #countj = (~dSSTj.mask).astype('i4'
                    #sumcounti[m] = np.sum(counti)
                    #sumcountj[m] = np.sum(countj)
                    #Hi[m] = np.sum(np.absolute(dSSTi))/sumcounti[m]
                    #Hj[m] = np.sum(np.absolute(dSSTj))/sumcountj[m]
                    #Hi[m] = np.sum(dSSTi)/Ny
                    #Hj[m] = np.sum(dSSTj)/Nx
                    Hi[m] += dSSTi.mean()
                    Hj[m] += dSSTj.mean()

            return Nt, dx, dy, L, Hi, Hj, lon, lat

        
        
        
        

