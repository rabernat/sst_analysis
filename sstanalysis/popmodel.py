import numpy as np
import xarray
import netCDF4
import warnings
import sys
from warnings import warn
from scipy import linalg as lin
from scipy import signal as sig
from scipy import fftpack as fft
from scipy import interpolate as naiso
import gsw
# import jmd95

class POPFile(object):
    
    def __init__(self, fname, areaname='TAREA', maskname='KMT', hmax=None, 
                 hconst=None, pref=0., ah=-3e9, am=-2.7e10, is3d=False):
        """Wrapper for POP model netCDF files"""
        #self.nc = netCDF4.Dataset(fname)
        self.nc = xarray.open_dataset(fname, decode_times=False)
        #self.Ny, self.Nx = self.nc.variables[areaname].shape  
        self.Ny, self.Nx = self.nc[areaname].shape  
        self._ah = ah
        self._am = am
        
        # mask
        self.mask = self.nc[maskname] > 1

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
        self.uv_dissipation = UVDissipation(self, hmax=hmax, hconst=hconst)
        self.hconst = hconst
        self.hmax = hmax

    def mask_field(self, maskname='KMT', varname='SST'):
        """Apply mask to tracer field T"""
        #mask = self.nc.variables[maskname][:]
        #T = self.nc.variables[varname][:]
        mask = self.nc[maskname][:]
        T = self.nc[varname][:]
        #return np.ma.masked_array(T, mask<=1)
        return T.where( mask>1 )
        
    def initialize_gradient_operator(self, field='tracer'):
        """Needs to be called before calculating gradients"""
        tarea = self.nc['TAREA'].values
        self.tarea = tarea
        tarea_r = np.ma.masked_invalid(tarea**-1).filled(0.)
        self.tarea_r = tarea_r
        
        dxtr = 1e2 * self.nc['DXT'].values**-1
        self._dxtr = dxtr
        dytr = 1e2 * self.nc['DYT'].values**-1
        self._dytr = dytr
        dxur = 1e2 * self.nc['DXU'].values**-1
        self._dxur = dxur
        dyur = 1e2 * self.nc['DYU'].values**-1
        self._dyur = dyur
        
        ############
        # Tracer
        ############
        if field == 'tracer':
            # raw grid geometry
            #work1 = (self.nc.variables['HTN'][:] /
            #         self.nc.variables['HUW'][:])
            work1 = (self.nc['HTN'][:] / self.nc['HUW'][:]).values
            #tarea = self.nc.variables[areaname][:]
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
            j_eq = np.argmin(self.nc['ULAT'][:,0].values**2)
            #self._ahf = (tarea / self.nc.variables['UAREA'][j_eq,0])**1.5
            self._ahf = (tarea / self.nc['UAREA'].values[j_eq,0])**1.5
            self._ahf[self.mask] = 0.   

            # stuff for gradient
            # reciprocal of dx and dy (in meters)
            #self._dxtr = 100.*self.nc.variables['DXT'][:]**-1
            #self._dytr = 100.*self.nc.variables['DYT'][:]**-1
            self._kmaske = np.where(kmt & kmte, 1., 0.)
            #self._kmaske = xray.DataArray(np.where(kmt & kmte, 1., 0.))
            self._kmaskn = np.where(kmt & kmtn, 1., 0.)
            #self._kmaskn = xray.DataArray(np.where(kmt & kmtn, 1., 0.))

            #self._dxu = self.nc.variables['DXU'][:]
            #self._dyu = self.nc.variables['DYU'][:]
            
        
        ############
        # Momentum
        ############
        elif field == 'momentum':
            p5 = .5
            c2 = 2.
            hus = self.nc['HUS'][:].values
            hte = self.nc['HTE'][:].values
            huw = self.nc['HUW'][:].values
            htn = self.nc['HTN'][:].values
            kmu = self.nc['KMU'].values > 1
            self.kmu = kmu
            uarea = self.nc['UAREA'].values
            self.uarea = uarea
            uarea_r = np.ma.masked_invalid(uarea**-1).filled(0.)
            self.uarea_r = uarea_r
            # coefficients for \nabla**2(U) (without metric terms)
            work1 = hus * hte**-1
            dus = work1 * uarea_r
            self._dus = dus
            dun = np.roll(work1, 1, axis=0) * uarea_r
            self._dun = dun
            work1 = huw * htn**-1
            duw = work1 * uarea_r
            self._duw = duw
            due = np.roll(work1, 1, axis=1) * uarea_r
            self._due = due
            
            # coefficients for metric terms in \nabla**2(U)
            # North-South
            work1 = (htn - np.roll(htn, -1, axis=0))
            work2 = np.roll(work1, 1, axis=0) - work1
            dyky = p5 * (work2 + np.roll(work2, 1, axis=1)) * dyur
            work2 = np.roll(work1, 1, axis=1) - work1
            dxky = p5 * (work2 + np.roll(work2, 1, axis=0)) * dxur
            #East-West
            work1 = (hte - np.roll(hte, -1, axis=0)) * tarea_r
            work2 = np.roll(work1, 1, axis=0) - work1
            dxkx = p5 * (work2 + np.roll(work2, 1, axis=0)) * dxur
            work2 = np.roll(work1, 1, axis=0) - work1
            dykx = p5 * (work2 + np.roll(work2, 1, axis=1)) * dyur
            
            kxu = (np.roll(huw, 1, axis=1) - huw) * uarea_r
            kyu = (np.roll(hus, 1, axis=0) - hus) * uarea_r
            
            dum = - (dxkx + dyky + c2*(kxu**2 + kyu**2))
            self._dum = dum
            dmc = dxky - dykx
            self._dmc = dmc
            duc = - (dun + dus + due + duw)
            self._duc = duc
            
            # coefficients for metric mixing terms which mix U,V.
            cc = duc + dum
            self._cc = cc
            dme = c2*kyu * (htn + np.roll(htn, 1, axis=1))**-1
            self._dme = dme
            dmn = -c2*kxu * (hte + np.roll(hte, 1, axis=0))**-1
            self._dmn = dmn
            self._dmw = -dme
            self._dms = -dmn
            
            j_eq = np.argmin(self.nc['ULAT'][:,0].values**2)
            self._amf = (uarea / self.nc['UAREA'].values[j_eq, 0])**1.5
            self._amf[~kmu] = 0. 
                
    def laplacian(self, T, field='tracer', V=0.):
        """Returns the laplacian"""
        #######
        # \nabla**2(T) at tracer points
        #######
        if field == 'tracer':
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
        #######
        # \nabla**2(U or V) at momentum points
        #######
        elif field == 'momentum':
            U = T.copy()
            u = (self._cc * U +
                self._dun * np.roll(U, -1, axis=0) + 
                self._dus * np.roll(U, 1, axis=0) + 
                self._due * np.roll(U, -1, axis=1) + 
                self._duw * np.roll(U, 1, axis=1) +
                self._dmc * V + 
                self._dmn * np.roll(V, -1, axis=0) + 
                self._dms * np.roll(V, 1, axis=0) + 
                self._dme * np.roll(V, -1, axis=1) + 
                self._dmw * np.roll(V, 1, axis=1)
                )
            v = (self._cc * V +
                self._dun * np.roll(V, -1, axis=0) + 
                self._dus * np.roll(V, 1, axis=0) + 
                self._due * np.roll(V, -1, axis=1) + 
                self._duw * np.roll(V, 1, axis=1) +
                self._dmc * U + 
                self._dmn * np.roll(U, -1, axis=0) + 
                self._dms * np.roll(U, 1, axis=0) + 
                self._dme * np.roll(U, -1, axis=1) + 
                self._dmw * np.roll(U, 1, axis=1)
                )
            #if u.ndim == 3:
                #for k in range(u.shape[0]):
                    #u[k][~self.kmu] = 0.
                    #v[k][~self.kmu] = 0.
            #else:
                #u[~self.kmu] = 0.
                #v[~self.kmu] = 0.
            
            return u, v
        
    def biharmonic_tendency(self, T, field='tracer'):
        """Calculate tendency due to biharmonic diffusion."""
        #########
        # tracer
        #########
        if field == 'tracer':
            d2tk = self._ahf * self.laplacian(T, field=field)
            return self._ah * self.laplacian(d2tk, field=field)
        #########
        # momentum
        #########
        elif field == 'momentum':
            d2uk, d2vk = self._amf * self.laplacian(T, field=field, V=self.nc['V1_1'].values)
            hduk, hdvk = self.laplacian(d2uk, field=field, V=d2vk)
            if hduk.ndim == 3:
                for k in range(hduk.shape[0]):
                    hduk[k][~self.kmu] = 0.
                    hdvk[k][~self.kmu] = 0.
            else:
                hduk[~self.kmu] = 0.
                hdvk[~self.kmu] = 0.
            return float(self._am) * np.asarray(hduk), float(self._am) * np.asarray(hdvk)
    
    def gradient_2d(self, varname='SST', lonname='ULONG', latname='ULAT', maskname='KMT', roll=0):
        """Return the gradient of tracer at velocity points
            for the global field
        """
        # step 1: load the data
        T = self.nc[varname]
        mask = self.nc[maskname] > 1
        Ti = T.where(mask)
        
        # shift T to points between
        barTy = .5 * (Ti.roll(nlat=-1) + Ti)
        barTx = .5 * (Ti.roll(nlon=-1) + Ti)
        
        # step 3: calculate the difference at U points
        dxdbarTy = ( (barTy.roll(nlon=-1).values - barTy.values) * self._dxur )
        dydbarTx = ( (barTx.roll(nlat=-1).values - barTx.values) * self._dyur )
        
        return dxdbarTy, dydbarTx
    
    def advection(self, varname='SST'):
        """Calculates the tracer advection term at T points
            as defined in POP Reference Manual
        """
        dyuu = self.nc['U1_1'] * self.nc['DYU']
        dxuv = self.nc['V1_1'] * self.nc['DXU']
        T = self.nc[varname]
        adv = .5**2 * ( (dyuu.values + dyuu.roll(nlat=-1).values) * (T.roll(nlon=1).values + T.values) 
                          - (dyuu.roll(nlon=-1).values + dyuu.roll(nlon=-1, nlat=-1).values) * (T.values + T.roll(nlon=-1).values)
                         + (dxuv.values + dxuv.roll(nlon=-1).values) * (T.roll(nlat=1).values + T.values)
                          - (dxuv.roll(nlat=-1).values + dxuv.roll(nlon=-1, nlat=-1).values) * (T.values + T.roll(nlat=-1).values)
                         ) * self.tarea**-1
        return adv
    
    def geostrophy(self, varname='SSH_2'):
        """Calculates the geostrophic velocity at U points from SSH
            as defined in POP Reference Manual"""
        mask = self.nc['KMT'] > 1
        SSH = 1e-2 * ( self.nc[varname].where(mask).values )
        dx = 1e-2 * self.nc['DXT'].where(mask).values
        dy = 1e-2 * self.nc['DYT'].where(mask).values
        uarea = 1e-4 * ( self.nc['UAREA'].values )
        lat = self.nc['ULAT'].values 
        g = gsw.earth.grav( lat.copy() )
        f = gsw.earth.f( lat.copy() )
        dyth = SSH.copy() * dy.copy()
        dxth = SSH.copy() * dx.copy()
        # derive geostrophic velocity
        geou = - g / f * .5 * ( 
                          (np.roll(np.roll(dxth, -1, axis=1), -1, axis=2) + np.roll(dxth, -1, axis=1))
                          - (np.roll(dxth, -1, axis=2) + dxth)
                          ) * uarea.copy()**-1
        geov = g / f * .5 * ( 
                          (np.roll(np.roll(dyth, -1, axis=2), -1, axis=1) + np.roll(dyth, -1, axis=2))
                          - (np.roll(dyth, -1, axis=1) + dyth)
                          ) * uarea.copy()**-1
        return geou, geov
        
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
        
    def spectrum_2d(self, varname='SST', tendname='Tdiss', advname='Tadv', lonname='TLONG', latname='TLAT', maskname='KMT', dxname='DXT', dyname='DYT', advt=False, tend=False, detrend=True, demean=False, filename=False, geosx=False, geosy=False, grady=False, lonrange=(154.9,171.7), latrange=(30,45.4), roll=-1000, nbins=128, MAX_LAND=0.01, xmin=0, xmax=0, ymin=0, ymax=0, ymin_bound=0, ymax_bound=0, xmin_bound=0, xmax_bound=0, daylag=13, daystart=0):
        """Calculate a two-dimensional power spectrum of netcdf variable 'varname'
            in the box defined by lonrange and latrange.
        """
        jmin_bound = ymin_bound
        jmax_bound = ymax_bound
        imin_bound = xmin_bound
        imax_bound = xmax_bound

        mask = self.nc[maskname] <= 1
        if advt:
            maskU = self.nc['KMU'] <= 1

        # step 1: figure out the box indices
        imax = xmax
        imin = xmin
        jmax = ymax
        jmin = ymin
        Nx = imax - imin
        Ny = jmax - jmin

        dx = 1e-2 * self.nc[dxname][:].roll( nlon=roll ).values[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        dy = 1e-2 * self.nc[dyname][:].roll( nlon=roll ).values[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        if advt:
            dxu = 1e-2 * self.nc['DXU'][:].roll( nlon=roll ).values[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
            dyu = 1e-2 * self.nc['DYU'][:].roll( nlon=roll ).values[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]

        ##############
        # step 2: load the data
        ##############
        if varname=='SST' or varname=='SSS':
            T = ( self.nc[varname].roll( nlon=roll ).values[:, 
                                                              jmin_bound:jmax_bound+100, imin_bound:imax_bound+100] )
            
            if advt:
                geoU = self.nc['geou'][:].roll( nlon=roll ).values[:, jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
                geoV = self.nc['geov'][:].roll( nlon=roll ).values[:, jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
                U = 1e-2 * ( self.nc['U1_1'].where(~maskU).roll( nlon=roll ).values[:, 
                                                            jmin_bound:jmax_bound+100, imin_bound:imax_bound+100] )
                V = 1e-2 * ( self.nc['V1_1'].where(~maskU).roll( nlon=roll ).values[:,
                                                            jmin_bound:jmax_bound+100, imin_bound:imax_bound+100] )
                tarea = 1e-4 * ( self.nc['TAREA'].where(~mask).roll( nlon=roll ).values[
                                                            jmin_bound:jmax_bound+100, imin_bound:imax_bound+100] )
                if self.hconst is not None:
                    Hmli = self.hconst
                else:
                    Hml = 1e-2 * ( self.nc['HMXL_2'].roll( nlon=roll ).values[:,
                                                            jmin_bound:jmax_bound+100, imin_bound:imax_bound+100] )

                # terms in advection
                #U = 1e-2 * self.nc[uname].roll( nlon=roll )
                #V = 1e-2 * self.nc[vname].roll( nlon=roll )
                #dTdx = self.nc[gradxname].roll( nlon=roll )
                #dTdy = self.nc[gradyname].roll( nlon=roll )
                #adv = U * dTdx + V * dTdy
                # give advection term at tracer points
                #P = .5 * ( .5 * ((adv).roll(nlat=-1) + (adv)).roll(nlon=-1).values 
                #                  + .5 * ((adv).roll(nlat=-1) + (adv).values) 
                #          )[:, jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
                #if self.hconst is not None:
                    #H_ml = self.hconst
                #else:
                    #H_ml = self.nc.variables[self.mlname].__getitem__(i)/100.
                    #H_ml = self.nc['HMXL_2'] / 1e2
                    #if self.hmax is not None:
                        #H_ml = np.ma.masked_greater(H_ml, self.hmax).filled(self.hmax)
                #P = (self.nc[advname] * H_ml).roll( nlon=roll ).values[:, jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
                
            elif tend:
                P = ( self.nc[tendname].roll( nlon=roll ).values[:,
                                                              jmin_bound:jmax_bound+100, imin_bound:imax_bound+100] )
            
        elif varname=='SSH_2':
            T = 1e-2 * self.nc[varname].roll( nlon=roll ).values[:, jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
            if geosy:
                barTy = .5*(np.roll(T,1,axis=1)+T)
                T = - gfd.g / gfd.f_coriolis(tlat) * (np.roll(barTy,1,axis=2)-barTy) / dx
            elif geosx:
                barTx = .5*(np.roll(T,1,axis=2)+T)
                T = gfd.g / gfd.f_coriolis(tlat) * (np.roll(barTx,1,axis=1)-barTx) / dy
            elif grady:
                barTy = .5*(np.roll(T,1,axis=1)+T)
                T = (np.roll(barTy,1,axis=2)-np.roll(barTy,-1,axis=2)) / (dx+np.roll(dx,-1,axis=1))
        else:
            T = 1e-2 * self.nc[varname].roll( nlon=roll ).values[:, jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]

        #############
        # step 3: figure out if there is too much land in the box
        #############
        #if crossspec or advt:
            #mask_domain_P = ( mask.roll( nlon=roll ).values[jmin_bound:jmax_bound+100, 
            #                                         imin_bound:imax_bound+100] + np.isnan(P[0]) )
            #region_mask_P = mask_domain_P[jmin:jmax, imin:imax]
        if advt:
            mask_domain_U = ( maskU.roll( nlon=roll ).values[jmin_bound:jmax_bound+100, 
                                                imin_bound:imax_bound+100] )
            region_mask_U = mask_domain_U[jmin:jmax, imin:imax]

        mask_domain_T = ( mask.roll( nlon=roll ).values[jmin_bound:jmax_bound+100, 
                                                imin_bound:imax_bound+100] )
        region_mask_T = mask_domain_T[jmin:jmax, imin:imax]            
        land_fraction = region_mask_T.sum().astype('f8') / (Ny*Nx)
        
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
        
        #############
        # step 4: figure out FFT parameters (k, l, etc.) and set up result variable      
        #############
        # Wavenumber step
        dx_domain = dx[jmin:jmax, imin:imax].copy()
        dy_domain = dy[jmin:jmax, imin:imax].copy()
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
        spac2_sum = np.zeros((Ny, Nx))
        tilde2_sum = np.zeros((Ny, Nx))
        if advt:
            tilde2_sum_P = np.zeros((Ny, Nx))
            spac2_sum_P = np.zeros((Ny, Nx))
            tilde2_sum_Q = np.zeros((Ny, Nx))
            spac2_sum_Q = np.zeros((Ny, Nx))
        Days = np.arange(daystart, Nt, daylag)
        Neff = len(Days)
        
        #nday = 0
        for n in Days:
            Ti = np.ma.masked_invalid( np.ma.masked_array(T[n, jmin:jmax, imin:imax].copy(), region_mask_T) )
            
            if advt: 
                Ui = np.ma.masked_invalid( 
                     np.ma.masked_array(U[n, jmin:jmax, imin:imax].copy(), region_mask_U) )
                Vi = np.ma.masked_invalid( 
                     np.ma.masked_array(V[n, jmin:jmax, imin:imax].copy(), region_mask_U) )
                geoui = np.ma.masked_invalid(
                    np.ma.masked_array(geoU[n, jmin:jmax, imin:imax].copy(), region_mask_U) )
                geovi = np.ma.masked_invalid(
                    np.ma.masked_array(geoV[n, jmin:jmax, imin:imax].copy(), region_mask_U) )
                dxui = np.ma.masked_invalid( 
                    np.ma.masked_array(dxu[jmin:jmax, imin:imax].copy(), region_mask_U) )
                dyui = np.ma.masked_invalid(
                    np.ma.masked_array(dyu[jmin:jmax, imin:imax].copy(), region_mask_U) )
                tareai = np.ma.masked_invalid(
                    np.ma.masked_array(tarea[jmin:jmax, imin:imax].copy(), region_mask_T) )
                if self.hconst is None:
                    Hmli = np.ma.masked_invalid( 
                        np.ma.masked_array(Hml[n, jmin:jmax, imin:imax].copy(), region_mask_T) )
              
            elif tend:
                Pi = np.ma.masked_invalid( np.ma.masked_array(P[n, jmin:jmax, imin:imax].copy(), region_mask_T) )
                #nday += 1
                #if np.isnan(Pi).any():
                    #print 'The tendency field includes NANs'

                    #sys.exit()
            
            # step 5: interpolate the missing data (only if necessary)
            if land_fraction>0. and land_fraction<MAX_LAND:
                Ti = interpolate_2d(Ti)
                
                if advt:
                    Ui = interpolate_2d(Ui)
                    Vi = interpolate_2d(Vi)
                    geoui = interpolate_2d(geoui)
                    geovi = interpolate_2d(geovi)
                    tareai = interpolate_2d(tareai)
                    if self.hconst is None:
                        Hmli = interpolate_2d(Hmli)
                    dxui = interpolate_2d(dxui)
                    dyui = interpolate_2d(dyui)
                
                elif tend:
                    if Pi.size != 0:
                        Pi = interpolate_2d(Pi)
                    else:
                        warnings.warn('Pi is a zero-size array')
                        break
                        #if np.isnan(Pi).any():
                            #print 'The tendency field includes NANs after interpolating'

                        #sys.exit()
            elif land_fraction==0.:
            # no problem
                pass
            else:
                break
            
            ###############
            # step 6: detrend the data in two dimensions (least squares plane fit)
            ###############
            if advt:
                Tp = detrend_2d(Ti.copy())
                Tb = Ti.copy() - Tp.copy()
                Up = Ui.copy() - geoui.copy()
                #Ub = Ui.copy() - Up.copy()
                Vp = Vi.copy() - geovi.copy()
                #Vb = Vi.copy() - Vp.copy()
                dyuub = geoui.copy() * dyui.copy()
                dxuvb = geovi.copy() * dxui.copy()
                dyuup = Up.copy() * dyui.copy()
                dxuvp = Vp.copy() * dxui.copy()
                #################
                # variance production with tracer advection scheme in POP Reference
                #################
                Pi = .5**2 * ( (dyuub + np.roll(dyuub, 1, axis=0)) * (np.roll(Tp, -1, axis=1) + Tp) 
                          - (np.roll(dyuub, 1, axis=1) + np.roll(np.roll(dyuub, 1, axis=0), 1, axis=1)) * (Tp + np.roll(Tp, 1, axis=1))
                         + (dxuvb + np.roll(dxuvb, 1, axis=1)) * (np.roll(Tp, -1, axis=0) + Tp)
                          - (np.roll(dxuvb, 1, axis=0) + np.roll(np.roll(dxuvb, 1, axis=0), 1, axis=1)) * (Tp + np.roll(Tp, 1, axis=0))
                         ) * tareai**-1 * Hmli
                Qi = .5**2 * ( (dyuup + np.roll(dyuup, 1, axis=0)) * (np.roll(Tb, -1, axis=1) + Tb) 
                          - (np.roll(dyuup, 1, axis=1) + np.roll(np.roll(dyuup, 1, axis=1), 1, axis=0)) * (Tb + np.roll(Tb, 1, axis=1))
                         + (dxuvp + np.roll(dxuvp, 1, axis=1)) * (np.roll(Tb, -1, axis=0) + Tb)
                          - (np.roll(dxuvp, 1, axis=0) + np.roll(np.roll(dxuvp, 1, axis=1), 1, axis=0)) * (Tb + np.roll(Tb, 1, axis=0))
                         ) * tareai**-1 * Hmli
                
                if np.isnan(Pi).any() or np.isnan(Qi).any():
                    print 'The tendency field has NANs'
                
                if detrend:
                    Pi = detrend_2d(Pi) 
                    Qi = detrend_2d(Qi)
                    if demean:
                        Pi -= Pi.mean()
                        Qi -= Qi.mean()
                else:
                    Pi -= Pi.mean()
                    Qi -= Qi.mean()
                
                if np.isnan(np.asarray(np.ma.masked_invalid(Pi))).any() == True or np.isnan(np.asarray(np.ma.masked_invalid(Qi))).any() == True:
                    print 'Pi or Qi has invalid numbers'
                    break
                
            elif tend:
                if detrend:
                    Pi = detrend_2d(Pi)
                        #if np.isnan(Pi).any():
                        #print 'The tendency field includes NANs after detrending'
                        #sys.exit()
                    if demean:
                        Pi -= Pi.mean()
                else:
                    # subtract the spatial mean
                    Pi -= Pi.mean()
                        #if np.isnan(Pi).any():
                            #print 'The tendency field includes NANs after demeaning'
                            #sys.exit()
            
            Ti = detrend_2d(Ti)
            if demean:
                Ti -= Ti.mean()

            ############
            # step 7: window the data
            # Hanning window
            ############
            windowx = sig.hann(Nx)
            windowy = sig.hann(Ny)
            window = windowx*windowy[:,np.newaxis] 
            Ti *= window
            #if np.isnan(Ti).any():
                #print 'The tracer field includes NANs after windowing'
            if advt:
                Pi *= window
                Qi *= window
                
            elif tend:
                Pi *= window
                #if np.isnan(Pi).any():
                    #print 'The tendency field includes NANs after windowing'
                    #sys.exit()
            
            # Aggregate the spatial variance
            if advt:
                spac2_sum_P += Ti * Pi
                spac2_sum_Q += Ti * Qi
                
            elif tend:
                spac2_sum += Ti * Pi
                #if np.isnan(spac2_sum).any():
                    #print 'The spac2 includes NANs'
            else:
                spac2_sum += Ti**2

            #############
            # step 8: do the FFT for each timestep and aggregate the results
            #############
            Tif = fft.fftshift(fft.fft2(Ti))    # [u^2] (u: unit)
            #if np.isnan(Tif).any():
                #print 'The tracer field includes NANs after FFT'

            if advt:
                Pif = fft.fftshift(fft.fft2(Pi)) 
                Qif = fft.fftshift(fft.fft2(Qi)) 
                tilde2_sum_P += np.real(np.conj(Tif) * Pif)
                tilde2_sum_Q += np.real(np.conj(Tif) * Qif)
                
            elif tend:
                Pif = fft.fftshift(fft.fft2(Pi)) 
                if np.isnan(Pif).any():
                    errstr = 'The tendency field (Pif) includes NANs after FFT'
                    warnings.warn(errstr)
                    #sys.exit()
                tilde2_sum += np.real(np.conj(Tif) * Pif)
                #if np.isnan(tilde2_sum).any():
                    #print 'The tilde2 includes NANs'

            else:
                tilde2_sum += np.real(Tif*np.conj(Tif))
            

        # step 9: check whether the Plancherel theorem is satisfied
        #tilde2_ave = tilde2_sum/Nt
        if advt:
            tilde2_sum = tilde2_sum_P + tilde2_sum_Q
            spac2_sum = spac2_sum_P + spac2_sum_Q
            breve2_sum = tilde2_sum / ((Nx*Ny)**2*dk*dl)  
        else:
            breve2_sum = tilde2_sum/((Nx*Ny)**2*dk*dl)  
        #if np.isnan(breve2_sum).any():
            #print 'The breve2 includes NANs'

        #breve2_ave = tilde2_ave/((Nx*Ny)**2*dk*dl)
        #spac2_ave = Ti2_sum/Nt
        kk, ll = np.meshgrid(k, l)
        K = np.sqrt( kk**2 + ll**2 )
        #Ki = np.linspace(0, k.max(), nbins)
        if k.max() > l.max():
            Ki = np.linspace(0, l.max(), nbins)
        else:
            Ki = np.linspace(0, k.max(), nbins)
        deltaKi = np.diff(Ki)[0]
        Kidx = np.digitize(K.ravel(), Ki)
        invalid = Kidx[-1]
        area = np.bincount(Kidx)
                
        if np.isnan(tilde2_sum).any():
            nanarray = np.zeros(nbins)
            nanarray[:] = np.nan
            isotropic_PSD = nanarray
        else:
            if land_fraction == 0.:
                #np.testing.assert_almost_equal(breve2_ave.sum()/(dx_domain[Ny/2,Nx/2]*dy_domain[Ny/2,Nx/2]*(spac2_ave).sum()), 1., decimal=5)
                np.testing.assert_almost_equal( breve2_sum.sum() 
                                               / ( dx_domain[Ny/2, Nx/2] * dy_domain[Ny/2, Nx/2] * (spac2_sum).sum() ), 1., decimal=5)
            
            # step 10: derive the isotropic spectrum
            # PREVIOUS: isotropic_PSD = np.ma.masked_invalid(
            #                               np.bincount(Kidx, weights=breve2_sum.ravel()) / area )[:-1] *Ki*2.*np.pi**2
            iso_wv = np.ma.masked_invalid(
                                       np.bincount(Kidx, weights=K.ravel()) / area )
            isotropic_PSD = np.ma.masked_invalid(
                                       np.bincount(Kidx, weights=breve2_sum.ravel()) / area ) * iso_wv

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
        return Neff, Nt, Nx, Ny, k, l, spac2_sum, tilde2_sum, breve2_sum, iso_wv, isotropic_PSD[:], area[:-1], land_fraction, MAX_LAND
        
    def structure_function(self, varname='SST', lonname='TLONG', latname='TLAT', maskname='KMT', dxname='DXT', dyname='DYT', lonrange=(154.9,171.7), latrange=(30,45.4), roll=-1000, q=2, MAX_LAND=0.01, Decor_lag = 13, xmin=0, xmax=0, ymin=0, ymax=0, ymin_bound=0, ymax_bound=0, xmin_bound=0, xmax_bound=0, detre=True, windw=True, iso=False, roll_param=True):
        """Calculate a structure function of Matlab variable 'varname'
           in the box defined by lonrange and latrange.
        """

        jmin_bound = ymin_bound
        jmax_bound = ymax_bound
        imin_bound = xmin_bound
        imax_bound = xmax_bound

        mask = self.nc[maskname] <= 1
        tlon = self.nc[lonname].where(~mask).roll( nlon=roll ).values[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        tlat = self.nc[latname].where(~mask).roll( nlon=roll ).values[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
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
        dx = 1e-2 * self.nc[dxname][:].roll( nlon=roll ).values[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
        dy = 1e-2 * self.nc[dyname][:].roll( nlon=roll ).values[jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]

        # load data
        if varname=='SST' or varname=='SSS':
            T = ( self.nc[varname].roll( nlon=roll ).values[:, 
                                                              jmin_bound:jmax_bound+100, imin_bound:imax_bound+100] )
        else:
            T = 1e-2 * ( self.nc[varname].roll( nlon=roll ).values[:, 
                                                              jmin_bound:jmax_bound+100, imin_bound:imax_bound+100] )
        
        # define variables
        Nt = T.shape[0]
        n = np.arange(0,np.log2(Nx/2), dtype='i4')
        ndel = len(n)
        dx_domain = dx[jmin:jmax,imin:imax].copy()
        dy_domain = dy[jmin:jmax,imin:imax].copy()
        L = 2**n*dy_domain[dy_domain.shape[0]/2, dy_domain.shape[1]/2]
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
        Nt = T.shape[0]
        #Decor_lag = 13
        Days = np.arange(0,Nt,Decor_lag)
        Neff = len(Days)
        for n in Days:
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
                            dSSTi = np.abs( Ti - np.roll(Ti, 2**m*int(round(dy_cen/dx_cen)),axis=1) )**q
                        else:
                            dSSTi = np.abs( Ti - np.roll(Ti, 2**m,axis=1) )**q
                        dSSTj = np.abs( Ti - np.roll(Ti, 2**m,axis=0) )**q
                        Hi[m] += dSSTi.mean()
                        Hj[m] += dSSTj.mean()
                    else:
                        # Take the difference by displacing the matrices along each axis 
                        #for i in range(Nx):
                        if dx_cen<dy_cen:
                            dSSTi = np.ma.masked_array((np.absolute(Ti[:,2**m*int(round(dy_cen/dx_cen)):] - Ti[:,:-2**m*int(round(dy_cen/dx_cen))]))**q) 
                        else:
                            dSSTi = np.ma.masked_array((np.absolute(Ti[:,2**m:] - Ti[:,:-2**m]))**q) 
                        dSSTj = np.ma.masked_array((np.absolute(Ti[2**m:] - Ti[:-2**m]))**q) # .filled(0.)
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

        return Neff, Nt, dx_cen, dy_cen, L, Hi, Hj, lon, lat, land_fraction, MAX_LAND
    
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

            
###########################################
# tracer advection
###########################################
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
        #Ffw = self.nc.variables[self.fwfname].__getitem__(i)
        #Qhf = self.nc.variables[self.hfname].__getitem__(i)
        Ffw = self.nc[self.fwfname].values.__getitem__(i)      #Fresh Water flux "kg/m^2/s"
        Qhf = self.nc[self.hfname].values.__getitem__(i)       #Surface Heat flux "watt/m^2" 
        
        FT_forc = hflux_factor * Qhf
        FS_forc = salinity_factor * Ffw
        
        return [ np.ma.masked_array(F, self.parent.mask) 
                 for F in [FT_forc, FS_forc] ]  

class TSDissipation(EOSCalculator):
    def __getitem__(self, i):
        T0, S0 = get_surface_ts(self.nc, i) 
    
        ###########
        # Necessary for dissipation term
        ###########
        if self.hconst is not None:
            H_ml = self.hconst
        else:
            #H_ml = self.nc.variables[self.mlname].__getitem__(i)/100.
            H_ml = self.nc[self.mlname].__getitem__(i) / 100.
            if self.hmax is not None:
                H_ml = np.ma.masked_greater(H_ml, self.hmax).filled(self.hmax)
        
        FT_mix = H_ml * self.parent.biharmonic_tendency(T0)
        FS_mix = H_ml * self.parent.biharmonic_tendency(S0)
        
        return [ np.ma.masked_array(F, self.parent.mask) 
                 for F in [FT_mix, FS_mix] ]  


#############################################
# momentum dissipation
#############################################
def get_surface_uv(nc, i):
    try:
        #S0 = nc.variables['SSS'].__getitem__(i)
        #T0 = nc.variables['SST'].__getitem__(i)
        U0 = nc['U1_1'].values.__getitem__(i)
        V0 = nc['V1_1'].values.__getitem__(i)
    except KeyError:
        #S0 = nc.variables['SALT'][:,0,:,:].__getitem__(i)
        #T0 = nc.variables['TEMP'][:,0,:,:].__getitem__(i) 
        U0 = nc['UVEL'][:, 0, :, :].values.__getitem__(i)
        V0 = nc['VVEL'][:, 0, :, :].values.__getitem__(i) 
    return U0, V0

class UVDissipation(EOSCalculator):
    def __getitem__(self, i):
        U0, V0 = get_surface_uv(self.nc, i) 
    
        ###########
        # Necessary for dissipation term
        ###########
        if self.hconst is not None:
            H_ml = self.hconst
        else:
            #H_ml = self.nc.variables[self.mlname].__getitem__(i)/100.
            H_ml = self.nc[self.mlname].__getitem__(i) / 100.
            if self.hmax is not None:
                H_ml = np.ma.masked_greater(H_ml, self.hmax).filled(self.hmax)
        
        FU_mix, FV_mix = H_ml * self.parent.biharmonic_tendency(U0, field='momentum')
        
        return [ np.ma.masked_array(F, self.parent.mask) 
                 for F in [FU_mix, FV_mix] ]  

        
        
        

