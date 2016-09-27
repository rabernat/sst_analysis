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
                 hconst=None, pref=0., ah=-3e17, am=-2.7e18, is3d=False):
        """Wrapper for POP model netCDF files. 
            The units of diffusivity and viscosity are in [cm^4/s]
        """
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
        """Needs to be called before calculating gradients
        """
        tarea = self.nc.TAREA.values
        self.tarea = tarea
        tarea_r = np.ma.masked_invalid(tarea**-1).filled(0.)
        self.tarea_r = tarea_r

        dxt_r = self.nc.DXT.values**-1
        self.dxt_r = dxt_r
        dyt_r = self.nc.DYT.values**-1
        self.dyt_r = dyt_r
        dxu_r = self.nc.DXU.values**-1
        self.dxu_r = dxu_r
        dyu_r = self.nc.DYU.values**-1
        self.dyu_r = dyu_r

        
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
            hus = self.nc.HUS.values
            hte = self.nc.HTE.values
            huw = self.nc.HUW.values
            htn = self.nc.HTN.values

            uarea = self.nc.UAREA.values
            self.uarea = uarea
            uarea_r = np.ma.masked_invalid(uarea**-1).filled(0.)
            self.uarea_r = uarea_r


            ###########
            # coefficients for \nabla**2(U) (without metric terms)
            ###########
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

            ###########
            # coefficients for metric terms in \nabla**2(U, V)
            ###########
            kxu = (np.roll(huw, 1, axis=1) - huw) * uarea_r
            kyu = (np.roll(hus, 1, axis=0) - hus) * uarea_r

            #East-West
            work1 = (hte - np.roll(hte, -1, axis=1)) * tarea_r
            work2 = np.roll(work1, 1, axis=1) - work1
            dxkx = p5 * (work2 + np.roll(work2, 1, axis=0)) * dxu_r
            self._dxkx = dxkx
            work2 = np.roll(work1, 1, axis=0) - work1
            dykx = p5 * (work2 + np.roll(work2, 1, axis=1)) * dyu_r
            self._dykx = dykx

            # North-South
            work1 = (htn - np.roll(htn, -1, axis=0)) * tarea_r
            work2 = np.roll(work1, 1, axis=0) - work1
            dyky = p5 * (work2 + np.roll(work2, 1, axis=1)) * dyu_r
            self._dyky = dyky
            work2 = np.roll(work1, 1, axis=1) - work1
            dxky = p5 * (work2 + np.roll(work2, 1, axis=0)) * dxu_r
            self._dxky = dxky


            dum = - (dxkx + dyky + c2*(kxu**2 + kyu**2))
            self._dum = dum
            dmc = dxky - dykx
            self._dmc = dmc


            ###########      
            # coefficients for metric mixing terms which mix U,V.
            ###########
            dme = c2*kyu * (htn + np.roll(htn, 1, axis=1))**-1
            self._dme = dme
            dmn = -c2*kxu * (hte + np.roll(hte, 1, axis=0))**-1
            self._dmn = dmn

            duc = - (dun + dus + due + duw)
            self._duc = duc
            dmw = -dme
            self._dmw = dmw
            dms = -dmn
            self._dms = dms
            
            
            j_eq = np.argmin(self.nc['ULAT'][:,0].values**2)
            self._amf = np.ma.masked_array((uarea 
                                   / self.nc['UAREA'].values[j_eq, 0])**1.5, ~self.mask).filled(0.)
                
    def _scalar_laplacian(self, T):
        """ \nabla**2(T) at tracer points
        """
        return (
            self._cc*T +
            self._cn*np.roll(T,-1,axis=0) +
            self._cs*np.roll(T,1,axis=0) +
            self._ce*np.roll(T,-1,axis=1) +
            self._cw*np.roll(T,1,axis=1)          
        )
            
    def _vector_laplacian(self, U, V):
        """ \nabla**2(U or V) at momentum points
        """
        cc = self._duc + self._dum
        
        if U.ndim == 2:
            lap_u = (cc * U +
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
            lap_v = (cc * V +
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
            lap_u = np.ma.masked_array(lap_u, ~self.mask).filled(0.)
            lap_v = np.ma.masked_array(lap_v, ~self.mask).filled(0.)
        elif U.ndim == 3:
            lap_u = (cc * U +
                self._dun * np.roll(U, -1, axis=1) + 
                self._dus * np.roll(U, 1, axis=1) + 
                self._due * np.roll(U, -1, axis=2) + 
                self._duw * np.roll(U, 1, axis=2) +
                self._dmc * V + 
                self._dmn * np.roll(V, -1, axis=1) + 
                self._dms * np.roll(V, 1, axis=1) + 
                self._dme * np.roll(V, -1, axis=2) + 
                self._dmw * np.roll(V, 1, axis=2)
                )
            lap_v = (cc * V +
                self._dun * np.roll(V, -1, axis=1) + 
                self._dus * np.roll(V, 1, axis=1) + 
                self._due * np.roll(V, -1, axis=2) + 
                self._duw * np.roll(V, 1, axis=2) +
                self._dmc * U + 
                self._dmn * np.roll(U, -1, axis=1) + 
                self._dms * np.roll(U, 1, axis=1) + 
                self._dme * np.roll(U, -1, axis=2) + 
                self._dmw * np.roll(U, 1, axis=2)
                )
            for t in range(lap_u.shape[0]):
                lap_u[t] = np.ma.masked_array(lap_u[t], ~self.mask).filled(0.)
                lap_v[t] = np.ma.masked_array(lap_v[t], ~self.mask).filled(0.)

            
        return lap_u, lap_v
    
    def laplacian(self, field, *args):
        """Returns the laplacian
        """
        if field == 'tracer':
            assert len(args)==1
            T = args[0]
            return self._scalar_laplacian(T)
        elif field == 'momentum':
            assert len(args)==2
            U, V = args
            return self._vector_laplacian(U, V)
        else:
            raise ValueError('field should be either tracer or momentum')
        
    def biharmonic_tendency(self, field, *args):
        """Calculate tendency due to biharmonic diffusion.
            Return the values in units [cm^2/s^2]
        """
        if field == 'tracer':
            assert len(args)==1
            T = args[0]
            d2tk = self._ahf * self.laplacian(field, T)
            return self._ah * self.laplacian(field, d2tk)
        elif field == 'momentum':
            assert len(args)==2
            U, V = args
            d2uk, d2vk = [self._amf * t for t in self.laplacian(field, U, V)]
            hduk, hdvk = [self._am * t for t in self.laplacian(field, d2uk, d2vk)]
            if hduk.ndim == 3:
                for t in range(hduk.shape[0]):
                    hduk[t] = np.ma.masked_array(hduk[t], ~self.mask).filled(0.)
                    hdvk[t] = np.ma.masked_array(hduk[t], ~self.mask).filled(0.)
            else:
                hduk = np.ma.masked_array(hduk, ~self.mask).filled(0.)
                hdvk = np.ma.masked_array(hduk, ~self.mask).filled(0.)
            return [hduk, hdvk]
        else:
            raise ValueError('field should be either tracer or momentum')
                
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
    
    def _spectra(T, days, k, l, mask, MAX_LAND=0.01, nbins=64, demean=False):
        """Calculate a isotropic wavenumber spectrum of T
        """
        #############
        # step 1: figure out if there is too much land in the box
        #############
        Ny, Nx = mask.shape
        land_fraction = mask.sum().astype('f8') / (Ny*Nx)

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
        # step 2: figure out FFT parameters (k, l, etc.) and set up result variable      
        #############
        # Wavenumber step
        dk = np.diff(k)[0]
        dl = np.diff(l)[0]

        #########################
        ###  Start actual calculation  ###
        #########################
        spac2_sum = np.zeros((Ny, Nx))
        tilde2_sum = np.zeros((Ny, Nx))

        for n in days:
    #         print n
            Ti = np.ma.masked_invalid( np.ma.masked_array(T[n].copy(), mask) )

            ##############
            # step 5: interpolate the missing data (only if necessary)
            ##############
            if land_fraction > 0. and land_fraction < MAX_LAND:
                Ti = _interpolate_2d(Ti)
            elif land_fraction==0.:
                # no problem
                pass
            else:
                break

            ###############
            # step 6: detrend the data in two dimensions (least squares plane fit)
            ###############
            Ti = _detrend_2d(Ti)
            if demean:
                Ti -= Ti.mean()

            ############
            # step 7: window the data
            ############
            # Hanning window
            windowx = sig.hann(Nx)
            windowy = sig.hann(Ny)
            window = windowx*windowy[:,np.newaxis] 
            Ti *= window

            spac2_sum += Ti**2

            #############
            # step 8: do the FFT for each timestep and aggregate the results
            #############
            Tif = fft.fftshift(fft.fft2(Ti))    # [u^2] (u: unit) 

            tilde2_sum += np.real(Tif*np.conj(Tif))

            #############
            # step 9: check whether the Plancherel theorem is satisfied
            #############
            #tilde2_ave = tilde2_sum/Nt
        breve2_sum = tilde2_sum/((Nx*Ny)**2*dk*dl)  

        kk, ll = np.meshgrid(k, l)
        K = np.sqrt( kk**2 + ll**2 )
        #Ki = np.linspace(0, k.max(), nbins)
        if k.max() > l.max():
    #         Ki = np.logspace(-8, np.log10(l.max()), nbins)
            Ki = np.linspace(0., l.max(), nbins)
        else:
            Ki = np.linspace(0., k.max(), nbins)
    #         Ki = np.logspace(-8, np.log10(k.max()), nbins)
        deltaKi = np.diff(Ki)[0]
        Kidx = np.digitize(K.ravel(), Ki)
        invalid = Kidx[-1]
        area = np.bincount(Kidx)

        if land_fraction == 0.:
                    #np.testing.assert_almost_equal(breve2_ave.sum()
    #                 /(dx_domain[Ny/2,Nx/2]*dy_domain[Ny/2,Nx/2]*(spac2_ave).sum()), 1., decimal=5)
            np.testing.assert_almost_equal( breve2_sum.sum() 
                                               / ( dx[Ny/2, Nx/2] * dy[Ny/2, Nx/2] 
                                                  * (spac2_sum).sum() ), 1., decimal=5)

        ###############    
        # step 10: derive the isotropic spectrum
        ###############
                #PREVIOUS: isotropic_PSD = np.ma.masked_invalid(
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

        ############
        # step 11: return the results
        ############
            #PREVIOUS: return Neff, Nt, Nx, Ny, k, l, Ti2_sum, tilde2_sum, breve2_sum, Ki, isotropic_PSD, area[1:], lon, lat, land_fraction, MAX_LAND
        return k, l, iso_wv, isotropic_PSD[:], area[:-1]
    
    def _cross_spectra(T, P, days, k, l, mask, MAX_LAND=0.01, nbins=64, demean=False):
        """Calculates a isotropic wavenumber cross spectrum of T and P.
        """
        #############
        # step 1: figure out if there is too much land in the box
        #############
        Ny, Nx = mask.shape
        land_fraction = mask.sum().astype('f8') / (Ny*Nx)

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
        # step 2: figure out FFT parameters (k, l, etc.) and set up result variable      
        #############
        # Wavenumber step
        dk = np.diff(k)[0]
        dl = np.diff(l)[0]

        #########################
        ###  Start actual calculation  ###
        #########################
        spac2_sum = np.zeros((Ny, Nx))
        tilde2_sum = np.zeros((Ny, Nx))

        for n in days:
    #         print n
            Ti = np.ma.masked_invalid( np.ma.masked_array(T[n].copy(), mask) )
            Pi = np.ma.masked_invalid( np.ma.masked_array(P[n].copy(), mask) )

            ##############
            # step 5: interpolate the missing data (only if necessary)
            ##############
            if land_fraction > 0. and land_fraction < MAX_LAND:
                Ti = _interpolate_2d(Ti)
                Pi = _interpolate_2d(Pi)
            elif land_fraction==0.:
                # no problem
                pass
            else:
                break

            ###############
            # step 6: detrend the data in two dimensions (least squares plane fit)
            ###############
            Ti = _detrend_2d(Ti)
            Pi = _detrend_2d(Pi)
            if demean:
                Ti -= Ti.mean()
                Pi -= Pi.mean()

            ############
            # step 7: window the data
            ############
            # Hanning window
            windowx = sig.hann(Nx)
            windowy = sig.hann(Ny)
            window = windowx*windowy[:,np.newaxis] 
            Ti *= window
            Pi *= window

            spac2_sum += Ti*Pi

            #############
            # step 8: do the FFT for each timestep and aggregate the results
            #############
            Tif = fft.fftshift(fft.fft2(Ti))    # [u^2] (u: unit)
            Pif = fft.fftshift(fft.fft2(Pi)) 

            tilde2_sum += np.real(Tif*np.conj(Pif))

            #############
            # step 9: check whether the Plancherel theorem is satisfied
            #############
            #tilde2_ave = tilde2_sum/Nt
        breve2_sum = tilde2_sum/((Nx*Ny)**2*dk*dl)  

        kk, ll = np.meshgrid(k, l)
        K = np.sqrt( kk**2 + ll**2 )
        #Ki = np.linspace(0, k.max(), nbins)
        if k.max() > l.max():
    #         Ki = np.logspace(-8, np.log10(l.max()), nbins)
            Ki = np.linspace(0., l.max(), nbins)
        else:
            Ki = np.linspace(0., k.max(), nbins)
    #         Ki = np.logspace(-8, np.log10(k.max()), nbins)
        deltaKi = np.diff(Ki)[0]
        Kidx = np.digitize(K.ravel(), Ki)
        invalid = Kidx[-1]
        area = np.bincount(Kidx)

        if land_fraction == 0.:
                    #np.testing.assert_almost_equal(breve2_ave.sum()
    #                 /(dx_domain[Ny/2,Nx/2]*dy_domain[Ny/2,Nx/2]*(spac2_ave).sum()), 1., decimal=5)
            np.testing.assert_almost_equal( breve2_sum.sum() 
                                               / ( dx[Ny/2, Nx/2] * dy[Ny/2, Nx/2] 
                                                  * (spac2_sum).sum() ), 1., decimal=5)

        ###############    
        # step 10: derive the isotropic spectrum
        ###############
                #PREVIOUS: isotropic_PSD = np.ma.masked_invalid(
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

        ############
        # step 11: return the results
        ############
            #PREVIOUS: return Neff, Nt, Nx, Ny, k, l, Ti2_sum, tilde2_sum, breve2_sum, Ki, isotropic_PSD, area[1:], lon, lat, land_fraction, MAX_LAND
        return k, l, iso_wv, isotropic_PSD[:], area[:-1]
    
    def spectrum_2d(self, maskname, dxname, dyname, tend, detrend, demean, roll, nbins, MAX_LAND, 
                    xmin, xmax, ymin, ymax, ymin_bound, ymax_bound, xmin_bound, xmax_bound, daylag, daystart, *args):
        """Calculates a isotropic wavenumber power spectrum of the variables prescribed in *args
            for the box defined (xmin_bound, xmax_bound) and (ymin_bound, ymax_bound).
        """
        jmin_bound = ymin_bound
        jmax_bound = ymax_bound
        imin_bound = xmin_bound
        imax_bound = xmax_bound

        mask = self.nc[maskname] <= 1

        # step 1: figure out the box indices
        imax = xmax
        imin = xmin
        jmax = ymax
        jmin = ymin
        Nx = imax - imin
        Ny = jmax - jmin

        dx = 1e-2 * (self.nc[dxname][:].roll( nlon=roll ).values[
                jmin_bound:jmax_bound+100, imin_bound:imax_bound+100])
        dy = 1e-2 * (self.nc[dyname][:].roll( nlon=roll ).values[
                jmin_bound:jmax_bound+100, imin_bound:imax_bound+100])

        ##############
        # step 2: load the data
        ##############
        if len(args) == 1:
            assert len(args) == 1
            Tname = args[0]
        elif len(args) == 2:
            assert len(args) == 2
            Tname, Pname = args
            P = ( self.nc[Pname].roll( nlon=roll ).values[:,
                                         jmin_bound:jmax_bound+100, imin_bound:imax_bound+100] )
        else:
            raise ValueError('The length of args should be either one or two')
            
        if Tname=='SST' or Tname=='SSS':
            T = ( self.nc[varname].roll( nlon=roll ).values[:, 
                                         jmin_bound:jmax_bound+100, imin_bound:imax_bound+100] )
            
            #if advt:
                #geoU = self.nc['geou'][:].roll( nlon=roll ).values[:, jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
                #geoV = self.nc['geov'][:].roll( nlon=roll ).values[:, jmin_bound:jmax_bound+100, imin_bound:imax_bound+100]
                #U = 1e-2 * ( self.nc['U1_1'].where(~maskU).roll( nlon=roll ).values[:, 
                #                                            jmin_bound:jmax_bound+100, imin_bound:imax_bound+100] )
                #V = 1e-2 * ( self.nc['V1_1'].where(~maskU).roll( nlon=roll ).values[:,
                #                                            jmin_bound:jmax_bound+100, imin_bound:imax_bound+100] )
                #tarea = 1e-4 * ( self.nc['TAREA'].where(~mask).roll( nlon=roll ).values[
                #                                            jmin_bound:jmax_bound+100, imin_bound:imax_bound+100] )
                #if self.hconst is not None:
                #    Hmli = self.hconst
                #else:
                #    Hml = 1e-2 * ( self.nc['HMXL_2'].roll( nlon=roll ).values[:,
                #                                            jmin_bound:jmax_bound+100, imin_bound:imax_bound+100] )

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
            
        elif Tname=='SSH_2' or Tname=='U1_1' or Tname=='V1_1':
            T = 1e-2 * (self.nc[varname].roll( nlon=roll ).values[:, 
                                                  jmin_bound:jmax_bound+100, imin_bound:imax_bound+100])
        else:
            raise ValueError('The field you prescribed does not exist')


        #############
        # step 3: figure out if there is too much land in the box
        #############
        mask_domain = ( mask.roll( nlon=roll ).values[jmin_bound:jmax_bound+100, 
                                                imin_bound:imax_bound+100] )
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

        ###################################
        ###  Start looping through each time step  ####
        ###################################
        Nt = T.shape[0]
        Days = np.arange(daystart, Nt, daylag)
        Neff = len(Days)
        
        if len(args) == 1:
            return _spectra(T, Days, k, l, mask, MAX_LAND, nbins, demean)
        elif len(args) == 1:
            return _cross_spectra(T, P, Days, k, l, mask, MAX_LAND, nbins, demean)
        
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
    
def _interpolate_2d(Ti):
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
 
def _detrend_2d(Ti):
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
            H_ml = self.nc[self.mlname].__getitem__(i)
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
    """Get the surface velocities: units in [cm/s]
    """
    try:
        U0 = nc['U1_1'].values.__getitem__(i)
        V0 = nc['V1_1'].values.__getitem__(i)
    except KeyError:
        U0 = nc['UVEL'][:, 0, :, :].values.__getitem__(i)
        V0 = nc['VVEL'][:, 0, :, :].values.__getitem__(i) 
    return U0, V0

class UVDissipation(EOSCalculator):
    def __getitem__(self, i):
        """Calculates the momentum dissipation term
        """
        U0, V0 = get_surface_uv(self.nc, i) 
    
        ###########
        # Necessary for dissipation term
        ###########
        if self.hconst is not None:
            H_ml = self.hconst
        else:
            #H_ml = self.nc.variables[self.mlname].__getitem__(i)/100.
            H_ml = self.nc[self.mlname].__getitem__(i)
            if self.hmax is not None:
                H_ml = np.ma.masked_greater(H_ml, self.hmax).filled(self.hmax)
        
        FU_mix, FV_mix = [H_ml * t for t in self.parent.biharmonic_tendency('momentum', U0, V0)]
        
        return [ np.ma.masked_array(F, self.parent.mask) 
                 for F in [FU_mix, FV_mix] ]  

        
        
        

