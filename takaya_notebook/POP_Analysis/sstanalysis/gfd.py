import numpy as np

# Earth radius
A = 6.371e6  # [m]
# rotation rate
Om = 7.292e-5  # s^-1
# gravity
g = 9.81  # m s^-2

def f_coriolis(lat):
    """Calculate the coriolis parameter.
    Latitude is assumed to be in degrees"""
    return 2*Om*np.sin(np.radians(lat))

def beta(lat):
    """Calculate the beta parameter.
    Latitude is assumed to be in degrees"""
    return 2*Om/A*np.cos(np.radians(lat))

#def dy_sph(lat):
#    """Calculate the spherical distance
#    between two longitudes"""
#    return A*np.radians(lat)

#def dx_sph(lat,lon):
#    """Calculate the spherical distance
#    between two latitudes"""
#    return A*np.sin(np.radians(lat)*np.radians(lon)
