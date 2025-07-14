#! /usr/bin/python3
# -*- coding: utf-8 -*-
""" Utility functions for flow analysis.
"""
import numpy as np
import logging
import scipy.stats as sc
from math import e
from scipy.optimize import curve_fit
import windtunnel as wt

logger = logging.getLogger()
__all__ = [
    'get_lux_referencedata',
    'get_new_lux_referencedata',
    'find_nearest',
    'get_reference_spectra',
    'transit_time_weighted_mean',
    'transit_time_weighted_var',
    'transit_time_weighted_flux',
    'calc_theo_arrival_law',
    'calc_arrival_law',
    'calc_transit_time_distribution',
    'get_turb_reference_values',
    'get_new_turb_reference_values'
    
]

def get_turb_reference_values(component, ref_path=None):
    if ref_path == None:
        ref_path = '//cifs-isi03.cen.uni-hamburg.de/ewtl/work/_EWTL Software/Python/Reference data/'
    if component=='I_u':
        slightly_rough = np.genfromtxt(ref_path + 'Iu_data.dat', unpack=True,
                        skip_header=11, skip_footer=367, usecols=(0,1))
        moderatly_rough = np.genfromtxt(ref_path + 'Iu_data.dat', unpack=True,
                        skip_header=41, skip_footer=337, usecols=(0,1))
        rough = np.genfromtxt(ref_path + 'Iu_data.dat', unpack=True,
                        skip_header=69, skip_footer=310, usecols=(0,1))
        very_rough = np.genfromtxt(ref_path + 'Iu_data.dat', unpack=True,
                        skip_header=103, skip_footer=269, usecols=(0,1))
    if component=='I_v':
        slightly_rough = np.genfromtxt(ref_path + 'Iv_data.dat', unpack=True,
                        skip_header=7, skip_footer=40, usecols=(0,1))
        moderatly_rough = np.genfromtxt(ref_path + 'Iv_data.dat', unpack=True,
                        skip_header=20, skip_footer=29, usecols=(0,1))
        rough = np.genfromtxt(ref_path + 'Iv_data.dat', unpack=True,
                        skip_header=31, skip_footer=15, usecols=(0,1))
        very_rough = np.genfromtxt(ref_path + 'Iv_data.dat', unpack=True,
                        skip_header=45, skip_footer=0, usecols=(0,1))
    
    if component=='I_w':
        slightly_rough = np.genfromtxt(ref_path + 'Iw_data.dat', unpack=True,
                        skip_header=11, skip_footer=347, usecols=(0,1))
        moderatly_rough = np.genfromtxt(ref_path + 'Iw_data.dat', unpack=True,
                        skip_header=37, skip_footer=321, usecols=(0,1))
        rough = np.genfromtxt(ref_path + 'Iw_data.dat', unpack=True,
                        skip_header=63, skip_footer=295, usecols=(0,1))
        very_rough = np.genfromtxt(ref_path + 'Iw_data.dat', unpack=True,
                        skip_header=89, skip_footer=269, usecols=(0,1))
        
    return slightly_rough, moderatly_rough, rough, very_rough

def get_new_turb_reference_values(manuel_VDI=False,values_VDI={}):
    """Calculates and returns the new VDI reference data for the turbulence intensity of
    component.

    Returns
    ----------
    

    z: np.array
    I_u: np.array
    I_v: np.array
    I_w: np.array

    """

    Kappa = 0.4
    zref = 10
    d0 = 0
    Uref = 5
    z0 = np.array([0.00001, 0.005, 0.1, 0.5, 2])

    if(manuel_VDI==False):
        tm = 3600
        Ave = 0.18
        Av = 1
        fu = 2.4
        fv = 2
        fw = 1.3
    
    else:
        for key, value in values_VDI.items():
             globals()[key] = value

    
    Ustar = ((Uref * Kappa)/np.log((zref-d0)/z0))

    #tm = 3600
    #Ave = 0.18
    #Av = 1
    #fu = 2.4
    #fv = 2
    #fw = 1.3
    z = np.arange(10, 305, 5) #np.arange(20, 305, 5)
    Iu = []
    Iv = []
    Iw = []
    for i in range(0, len(z0)-1):
       
        
        Sigmau=Av*fu*Ustar[i]
    
        Sigmav=Av*fv*Ustar[i]
    
        Sigmaw=Av*fw*Ustar[i]
        
        U = (Ustar[i]/Kappa)*np.log((z-d0)/z0[i])
        
        Iu.append(Sigmau/U)
        Iv.append(Sigmav/U)
        Iw.append(Sigmaw/U)
        
        
        
    return z, Iu, Iv, Iw

def get_lux_referencedata(ref_path=None):
    """Reads and returns reference data for the integral length scale (Lux).
    This function takes no parameters. 
    
    Returns
    ----------
    
    
    Lux_10: array-like
    Lux_1: array-like
    Lux_01: array-like
    Lux_001:array-like
    Lux_obs_smooth: array-like
    Lux_obs_rough:array-like
    
    """
    if ref_path == None:
        ref_path = '//cifs-isi03.cen.uni-hamburg.de/ewtl/work/_EWTL Software/Python/Reference data/'
    Lux_10 = np.genfromtxt(ref_path + 'Lux_data.dat', skip_header=7, skip_footer=421,
                           usecols=(0, 1), unpack=True)
    Lux_1 = np.genfromtxt(ref_path + 'Lux_data.dat', skip_header=32, skip_footer=402,
                          usecols=(0, 1), unpack=True)
    Lux_01 = np.genfromtxt(ref_path + 'Lux_data.dat', skip_header=51,
                           skip_footer=388, usecols=(0, 1), unpack=True)
    Lux_001 = np.genfromtxt(ref_path + 'Lux_data.dat', skip_header=65,
                            skip_footer=375, usecols=(0, 1), unpack=True)
    Lux_obs_smooth = np.genfromtxt(ref_path + 'Lux_data.dat', skip_header=78,
                                   skip_footer=317, usecols=(0, 1), unpack=True)
    Lux_obs_rough = np.genfromtxt(ref_path + 'Lux_data.dat', skip_header=136,
                                  skip_footer=276, usecols=(0, 1), unpack=True)

    return Lux_10, Lux_1, Lux_01, Lux_001, Lux_obs_smooth, Lux_obs_rough



def get_new_lux_referencedata(manuel_VDI=False,values_VDI={}):
    
    z0 = np.array([10.0, 1.0, 0.1, 0.01])
    z = np.arange(10, 305, 5)# np.arange(20, 305, 5)

    if(manuel_VDI==False):
        Gamma =1/(1.55-0.01/z0+0.64/(z0**0.5))
        B=3.12 + 19.92/(z0**0.5)-0.51/z0
    else:
        print("gets here")
        for key,value in values_VDI.items():
            vars()[key] = value
            print(key)
            print(value)
            #print(B)
    print(Gamma)
    print(B)        
        
    
    Lux_10=np.vstack([z,B[0]*z**Gamma[0]])
    Lux_1=np.vstack([z,B[1]*z**Gamma[1]])
    Lux_01=np.vstack([z,B[2]*z**Gamma[2]])
    Lux_001=np.vstack([z,B[3]*z**Gamma[3]])
    
    return Lux_10, Lux_1, Lux_01, Lux_001




def find_nearest(array, value):
    """ Finds nearest element of array to value.

    Parameters
    ----------
    
    
    array: np.array
    value: int or float

    Returns
    ----------
    
    
    array[idx]: float
    
     """
    idx = (np.abs(array - value)).argmin()

    return array[idx]

def get_reference_spectra(height, ref_path=None):
    """ Get reference spectra from pre-defined location.
    
    Parameters
    ----------
    
    
    height: int or float

    Returns
    ----------
    

    ref_specs: array-like
    
    """
    #  REFERENCE SPAECTRA RANGE FIT
    if ref_path == None:
        ref_path = '//cifs-isi03.cen.uni-hamburg.de/ewtl/work/_EWTL Software/Python/Reference data/'
    ref_heights = np.array([7.00, 10.50, 14.00, 17.50, 22.75, 42.00, 70.00, 105.00])
    value = find_nearest(ref_heights, height)
    value = '{:03.2f}'.format(value)
    ref_specs = np.genfromtxt(ref_path + 'ref_spectra_S_ii_z_' + value + 'm.txt')

    return ref_specs

def transit_time_weighted_mean(transit_time, component):
    """ Weigh the flow component with its transit time through the
    measurement volume. This is analoguous to the processing of the raw
    data in the BSA software. Transit time weighting removes a possible
    bias towards higher wind velocities. Returns the weighted component mean.
    
    Parameters
    ----------
    
    
    transit_time: np.arrray
    component: np.arrray
    
    Returns
    ----------
    

    weighted_mean: float
    
    """
    #edit 05/27/2020: removed nan values from transit time array    

    transit_time_sum = np.sum(transit_time[~np.isnan(transit_time)])

    weighted_mean = np.sum((component[~np.isnan(transit_time)] 
                        * transit_time[~np.isnan(transit_time)]) / transit_time_sum)

    return float(weighted_mean)

def transit_time_weighted_var(transit_time, component):
    """ Weigh the u and v component with its transit time through the
    measurement volume. This is analoguous to the processing of the raw
    data in the BSA software. Transit time weighting removes a possible
    bias towards higher wind velocities. Returns the weighted u and v
    component variance.

    Parameters
    ----------
    
    transit_time: np.arrray
    component: np.arrray
    
    Returns
    ----------
    
    weighted_var: float

    """
    #edit 05/27/2020: removed nan values from transit time array    

    transit_time_sum = np.sum(transit_time[~np.isnan(transit_time)])

    tmp = (((component[~np.isnan(component)] - np.mean(component[~np.isnan(component)])) ** 2) * \
          (transit_time[~np.isnan(transit_time)])) / transit_time_sum

    weighted_var = np.sum(tmp)

    return float(weighted_var)

def transit_time_weighted_flux(transit_time, component_1, component_2):
    """ Calculate mean flux using transit time weighted statistics. Transit
    time weighting removes a possible bias towards higher wind velocities.
    Returns a mean weighted flux.

    Parameters
    ----------
    
    
    transit_time: np.arrray
    component_1: np.arrray
    component_2: np.arrray
    
    Returns
    ----------
    
    weighted_flux: float

    """

    transit_time_sum = np.sum(transit_time[~np.isnan(transit_time)])
    weighted_flux = np.sum((component_1 - np.mean(component_1)) 
                            * (component_2 - np.mean(component_2)) 
                            * transit_time[~np.isnan(transit_time)]) / \
                            transit_time_sum

    return float(weighted_flux)

def calc_theo_arrival_law(t_arr, data_rate):
    """ 
    calculate theoretical particle arrival law. 
    if exponential, there is temporally uniform seeding.
    Input parameters are the arrival times for each burst and the data rate of the measurement.

    Parameters
    ----------
    

    t_arr:list or np.array
    data_rate:float

    Returns
    ----------
    

    delta_t_arr: array  
    particle_arrival_law: array
    
    """

    # allocate
    delta_t_arr = []
    # calculate inter arrival times for each burst
    delta_t_arr = [ t_arr[i+1] - t_arr[i] for i in range(len(t_arr)-1) ]
    delta_t_arr = np.asarray(delta_t_arr)
    # calculate particle arrival law: p(t_arr) = dN/dt * exp(- dN/dt * t_arr)
    particle_arrival_law = data_rate * np.exp(-data_rate * delta_t_arr)

    return delta_t_arr, particle_arrival_law

def calc_arrival_law(t_arr, data_rate):
    """ 
    calculate particle arrival law and fit the distribution. 
    if exponential, there is temporally uniform seeding.

    Parameters
    ----------
    
    
    t_arr: list or np.array
    data_rate: float

    Returns
    ----------
    

    binscenters: list or array
    data_entries: numpy object
    popt: array

    """

    # allocate
    delta_t_arr = []
    # calculate inter arrival times for each burst
    delta_t_arr = [ t_arr[i+1] - t_arr[i] for i in range(len(t_arr)-1) ]
    delta_t_arr = np.asarray(delta_t_arr)

    data_entries, bins = np.histogram(delta_t_arr, bins='auto',density=True)
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])

    def fit_function(x, A):         #No documentation available
        return (A * np.exp(-x * A) )

    popt, pcov = curve_fit(fit_function, xdata=binscenters, ydata=data_entries)
    print('     fitted data rate = {}'.format(popt))
    print('     expected data rate = {}'.format(data_rate))

    return binscenters, data_entries, popt

def calc_transit_time_distribution(transit_time):
    """ 
    calculate particle arrival law. 
    if exponential, there is temporally uniform seeding.

    Parameters
    ----------
    
    transit_time: list or np.array
    
    Returns
    ----------
    
    
    """

    return sc.skew(transit_time, nan_policy='omit')
