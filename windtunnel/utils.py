#! /usr/bin/python3
# -*- coding: utf-8 -*-
""" General utility functions for the windtunnel package.
"""
import numpy as np
from scipy.interpolate import interp1d
#from skimage.measure import label
import pandas as pd
import fnmatch
import logging
import os
import scipy.stats as sc
import windtunnel as wt
from math import e

logger = logging.getLogger()
__all__ = [
    'find_block',
    'equ_dist_ts',
    'trunc_at',
    'get_files',
    'get_pdf_max',
    'check_directory',
    'get_percentiles'
]

def find_block(indata, length, tolerance):
    """ Finds block of size length in indata. Tolerance allows some leeway.
    Returns array.

    Parameters
    ----------
    

    indata: np.array (1D)
    length: int
    tolerance: int 

    Returns
    ----------

    block: int or float
    
    """

    for i in range(0, np.size(indata) - length):
        block = indata[i:i + length]
        if np.sum(np.isnan(block)) <= tolerance:
            return block

    raise Exception('Interval of given length and quality not found.')

def equ_dist_ts(arrival_time, eq_dist_array, data):
    """ Create a time series with constant time steps. The nearest point of the
   original time series is used for the corresponding time of the equi-distant
   time series.

    Parameters
    ----------
    
   
    arrival_time: np.array
    eq_dist_array: np.array
    data: np.array
   
    Returns
    ----------

    eq_dist_array: array
     
    """

    valid = ~np.isnan(data)

    f = interp1d(arrival_time[valid], data[valid],
                 kind='nearest',
                 fill_value='extrapolate')

    return f(eq_dist_array)

def trunc_at(string, delimiter, n=3):
    """ Returns string truncated at the n'th (3rd by default) occurrence of the
    delimiter.
    
    Parameters
    ----------
    
    string: str
    delimiter: str
    n: int

    Returns
    ----------

    """

    return delimiter.join(string.split(delimiter, n)[:n])
    
def get_files(path, filename):
    """Finds files with filename in path as specified. Filename supports the
    Unix shell-style wildcards.

    Parameters
    ----------

    path: str
    filename: str 
    
    Returns
    ----------

    return_files: list

    """

    all_files = os.listdir(path)
    return_files = []
    for file in all_files:
        if fnmatch.fnmatch(file, filename + '*'):
            return_files.append(file)

    return_files.sort()

    return return_files

def get_pdf_max(data):
    """Finds maximum of the probability distribution of data.
    
    Parameters
    ----------

    data: np.array
    
    Returns
    ----------

    result: float

    """

    df = pd.DataFrame(data, columns=['data'])
    nparam_density = sc.kde.gaussian_kde(df.values.ravel())
    heights, bins = np.histogram(data[~np.isnan(data)], bins='auto')
    nparam_density = nparam_density(bins)
    result = bins[np.argsort(nparam_density)[-1]]

    return result

def check_directory(directory):
    """ Checks if directory exists. If directory doesn't exist, it is created.

    Parameters
    ----------
    
    directory: str 
    
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Desired directory created.")

def get_percentiles(data_dict, percentile_list):
    """ Get percentiles from each entry in data_dict specified in
    percentile_list.
    
    Parameters
    ----------
    
    data_dict: dictionary
    percentile_list: list 
    
    Returns
    ----------
    

    percentile_dict: dictionary

    """

    # Generate namelist from dict keys
    namelist = list(data_dict.keys())

    percentile_dict = {}
    percentile_dict.fromkeys(namelist)

    for name in namelist:
        percentile_dict[name] = {}
        for percentile in percentile_list:
            percentile_dict[name][percentile] = np.percentile(
                data_dict[name],
                percentile)

    return percentile_dict
	
