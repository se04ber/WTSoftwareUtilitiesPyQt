#! /usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import logging
import os
import pandas as pd
import scipy as sc
from scipy.stats import norm
import matplotlib.pyplot as plt


# Create logger
logger = logging.getLogger()
__all__ = ['PuffConcentration']

class PuffConcentration(pd.DataFrame):
    """ PuffConcentration is a class that holds data collected during 
    a puff release point concentration measurement. The class can hold
    the raw time series, the corresponding wtref and all other quantities
    necessary to analyse the time series. The PuffConcentration class
    inherits from pandas.DataFrame, thus offers all of the functionality
    offered by pandas (e.g. DataFrame.plot.hist(), DataFrame.to_excel(),
    or DataFrame.rolling().mean()) All the information in a
    PuffConcentration object can be saved to a txt file, as well as all
    file type offered by pandas.


    Parameters
    ----------
    


    time: pd.Series
    wtref: np.array
    fast_FID: pd.Series
    slow_FID: pd.Series
    signal: np.array
    open_rate: np.array
    
    """

    def __init__(self, time, wtref, slow_FID, fast_FID, signal, open_rate):
        """ Initialise PuffConcentration object. """
		#edit 07/26/2019: added attributes 
		#edit 07/29/2019: added attribute mask for proper plotting
        #edit 08/01/2019: added attribute mean_puff
        #edit 08/02/2019: added attributes signal_array, puffs_array, mean_array	, pct10_puff, pct10_signal, pct90_puff, pct90_signal
        #edit 09/19/2019: added attibutes wtref_mean, c_star, calibration_curve, calibration_factor, full_scale_concentration, full_scale_flow_rate,
        #full_scale_ref_length, full_scale_time, full_scale_wtref, gas_factor, gas_name, mol_weight, temperature, temperature_K, mass_flow_controller, 
        #mass_flow_rate, pressure, ref_height, ref_length, scaling_factor, scaling_factor, standard_pressure, R (universal Gas constant), and __check_sum.  
        #edit 09/26/2019: added attribute begin_release_index_masked
        #edit 09/27/2019: added attributes dosage_unmasked, mask_full, and dt
        #edit 10/14/2019: added attributes arrival_index and leaving_index	
        #edit 02/21/2020: added attributes x_source, y_source, z_source, x_measure, y_measure, z_measure, and distance.
		
        super().__init__()

        self['slow_FID'] = pd.Series(data=slow_FID)
        self['fast_FID'] = pd.Series(data=fast_FID)

        self.signal = signal
        self.time = time
        self.dt = None
        self.wtref = wtref
        self.net_concentration = None			
        self.open_rate = open_rate  # [%]
        self.x = None
        self.y = None
        self.z = None
        self.x_source = None
        self.y_source = None
        self.z_source = None 
        self.x_measure = None
        self.y_measure = None
        self.z_measure = None         
        self.distance = None        
        self.scale = None
        self.wtref_mean = None	
        self.c_star = None	
        self.calibration_curve = None
        self.calibration_factor = None	
        self.full_scale_concentration = None
        self.full_scale_flow_rate = None
        self.full_scale_ref_length = None	
        self.full_scale_time = None	
        self.full_scale_wtref = None		
        self.begin_release_period = None	
        self.end_release_period = None		
        self.begin_release_index = None	
        self.arrival_index = None			
        self.begin_release_index_unmasked = None	        
        self.end_release_index = None
        self.leaving_index = None			
        self.gas_factor = None
        self.gas_name = None
        self.mol_weight = None	
        self.temperature = None	
        self.temperature_K = None
        self.mass_flow_controller = None
        self.mass_flow_rate = None
        self.ref_height = None
        self.ref_length = None	
        self.scaling_factor = None	
        self.standard_temp_K = None		
        self.pressure = None		
        self.mask = None
        self.mask_full = None		
        self.release_length = None
        self.residence_time = None
        self.arrival_time = None		
        self.leaving_time = None
        self.ascent_time = None
        self.descent_time = None
        self.peak_time = None
        self.peak_concentration = None
        self.dosage = None
        self.dosage_unmasked = None        
        self.puffs_array = None
        self.signal_array = None				
        self.mean_puff = None
        self.pct90_puff = None	
        self.pct10_puff = None			
        self.mean_signal = None	
        self.pct90_signal = None	
        self.pct10_signal = None				
        self.min_puff_length = None		
        self.puff_deviations = None
        self.threshold_concentration = None
        self.number = None
        self.standard_temp = 20  # [°C]	
        self.standard_pressure = 101325  # [Pa]
        self.R = 8.3144621  # universal gas constant [kJ/kgK]	
        self.__check_sum = 0

        #New quantites
        self.peak_to_mean_ratio = None
        self.dosage_fluctuation = None
        self.frequency = None
        self.power_spectral_density = None
        self.fluctuation_intensity = None
    
        self.integral_time_scale = None
        self.autocorrelation = None
        		

    def __repr__(self):
        """ Return the x, y and z coordinate of the PointConcentration 
        object. """
        return 'PuffConcentration (x={x}, y={y}, z={z})'.format(x=self.x,
                                                                y=self.y,
                                                                z=self.z)

    def __eq__(self, other):
        """ Two PuffConcentration objects are considered equal, if their x, y
        and z coordinates are the same. """
        return self.x == other.x and self.y == other.y and self.z == other.z

    @classmethod
    def from_file(cls, filename):
        """ Create PuffConcentration object from file. open_rate is converted
        to %.
        
        Parameters
        ----------
        
        
        filename: str
        cls:str
        
        """
        #time, wtref, slow_FID, fast_FID, open_rate, signal # for trying with point data debugging
        #time, wtref, slow_FID, fast_FID, signal, open_rate #default before
        time, wtref, slow_FID, fast_FID, signal, open_rate = np.genfromtxt(
            filename, usecols=(0, 1, 2, 3, 4, 5), unpack=True)

        # Turn signal vector into a binary vector, using 4.5V as the
        # threshold to detect an active release signal.
        threshold_indices = signal < 4.5
        signal[threshold_indices] = 0
        signal[~threshold_indices] = 1
	

        # Apply median filter to signal, to guarantee a clean array
        signal = sc.signal.medfilt(signal, kernel_size=9)
		
        return cls(time, wtref, slow_FID, fast_FID, signal, open_rate * 10)
		
    def to_full_scale(self):
        """Converts all quantities to full scale while preserving model scale time"""
        # Store model scale time before converting
        self.model_scale_time = self.time.copy()
        
        if self.__check_sum >= 8:

           #quantities = ['x', 'y', 'z', 'time', 'concentration', 'flow rate']
           #your_measurement = {}
           #your_measurement.fromkeys(quantities)

           self.x = self.x * self.scale / 1000  # [m]
           self.y = self.y * self.scale / 1000  # [m]
           self.z = self.z * self.scale / 1000  # [m]
           self.x_source = self.x_source * self.scale / 1000  # [m]
           self.y_source = self.y_source * self.scale / 1000  # [m]
           self.z_source = self.z_source * self.scale / 1000  # [m]   
           self.x_measure = self.x_measure * self.scale / 1000  # [m]
           self.y_measure = self.y_measure * self.scale / 1000  # [m]
           self.z_measure = self.z_measure * self.scale / 1000  # [m]               

           self.calc_full_scale_time()
           self.calc_full_scale_concentration()	
           self.calc_full_scale_flow_rate()		   

           self.time = self.full_scale_time		   
           self.net_concentration = self.full_scale_concentration

        else: 
           raise Exception('Please enter or calculate all full scale data '
                            'necessary!')
        

    def to_non_dimensional(self):
        """Converts all quantities to non-dimensional while preserving model scale time"""
        # Store model scale time before converting
        self.model_scale_time = self.time.copy()
        
        if self.__check_sum >= 8:

           #quantities = ['x', 'y', 'z', 'time', 'concentration', 'flow rate']
           #your_measurement = {}
           #your_measurement.fromkeys(quantities)

           self.x = self.x / self.ref_length  # [-]
           self.y = self.y / self.ref_length  # [-]
           self.z = self.z / self.ref_length  # [-]
           self.x_source = self.x_source / self.ref_length  # [-]
           self.y_source = self.y_source / self.ref_length  # [-]
           self.z_source = self.z_source / self.ref_length  # [-]    
           self.x_measure = self.x_measure / self.ref_length  # [-]
           self.y_measure = self.y_measure / self.ref_length  # [-]
           self.z_measure = self.z_measure / self.ref_length  # [-]                

           self.calc_non_dimensional_time()
           self.calc_c_star()	
           self.calc_non_dimensional_flow_rate()		   

           self.time = self.non_dimensional_time
           self.net_concentration = self.c_star

        else: 
            raise Exception('Please enter or calculate all full scale data '
                            'necessary!')             

    def get_ambient_conditions(path=None,name=None,input_file=None):
        """Read ambient conditions from csv file. If no such file exists, function
		does nothing and instead ambient conditions are read from values in
		example_puff_measurement.py.
        
        Returns
        ----------
        
        
        ambient_conditions: str
        """	
		#edit 10/18/2019: new function, which reads ambient conditions during measurement 
		#from seperate csv file. Assumes that the csv data is located in the same folder
        #as the measurement data, and that each column represents an input variable,
        #and each row represents a dataset. If no such file exists in the data directory,
        #function does nothing and instead ambient conditions are read from input variables
		#in example_puff_measurement.py.fcon	
        #edit 02/21/2020: added variables for source and measurement locations      
        if input_file==None:
           print('Warning: Input csv filename (for ambient conditions) not specified. Resorting to input data in example_puff_measurement.py')
           return		   
        elif name==None:
           print('Warning: Name of dataset not specified. Cannot attempt to locate csv file containing ambient conditions data. Resorting\
to input data in example_puff_measurement.py')
        elif path==None:
           print('Warning: Path of input csv file (for ambient conditions) not specified. Resorting to input data in example_puff_measurement.py')			   
           return
        elif not os.path.exists(input_file):        
           print('Error: Cannot find csv file containing ambient conditions in specified directory. Check name and/or location of ambient \
conditions file. Resorting to input data in example_puff_measurement.py')	
           return		   
        else:	
           ambient_conditions=pd.read_csv(input_file,sep=',',index_col=0) 
        
        if name not in ambient_conditions.keys():
           print('Error: Dataset not found in csv file. Check to make sure that csv file to make sure that the csv file contains all necessary \
data and is properly formatted. Resorting to input data in example_puff_measurement.py')
           return	

        #list of all variables output by read_ambient_conditions fuction.  
        necessary_keys={'x_source','y_source','z_source','x_measure','y_measure','z_measure','pressure','temperature','wdir','calibration_curve','mass_flow_controller','calibration_factor', \
        'scaling_factor','scale','ref_length','ref_height','gas_name','mol_weight','gas_factor','full_scale_wtref','full_scale_flow_rate' }
        if not all(name2 in ambient_conditions[name] for name2 in necessary_keys):
           print('Error: csv file does not contain all necessary ambient conditions data. Check to make sure that csv file to make sure that \
the csv file contains all necessary data and is properly formatted. Resorting to input data in example_puff_measurement.py')
           return			   
       		

        return ambient_conditions	

    def read_ambient_conditions(ambient_conditions,name):
        """Populate individual variables representing ambient conditions based on data
		in ambient_conditions array. 
        
        Parameters
        ----------
        

        ambient_conditions: str
        name: str
        
        Returns
        ----------
        
        
        
        x_source: float
        y_source: float
        z_source: float
        x_measure: float
        y_measure: float
        z_measure: float
        pressure: float
        temperature: float
        wdir: float
        calibration_curve: float
        mass_flow_controller: str
        calibration_factor: float
        scaling_factor: float
        scale: float
        ref_length: float
        ref_height: float
        gas_name: str
        mol_weight: float
        gas_factor: float
        full_scale_wtref: float
        full_scale_flow_rate: float
        """	           
 	   
        x_source=None if ambient_conditions[name]['x_source'] =='None' else np.float64(ambient_conditions[name]['x_source'])
        y_source=None if ambient_conditions[name]['y_source'] =='None' else np.float64(ambient_conditions[name]['y_source'])
        z_source=None if ambient_conditions[name]['z_source'] =='None' else np.float64(ambient_conditions[name]['z_source'])  
        x_measure=None if ambient_conditions[name]['x_measure'] =='None' else np.float64(ambient_conditions[name]['x_measure'])
        y_measure=None if ambient_conditions[name]['y_measure'] =='None' else np.float64(ambient_conditions[name]['y_measure'])
        z_measure=None if ambient_conditions[name]['z_measure'] =='None' else np.float64(ambient_conditions[name]['z_measure'])          
        pressure=None if ambient_conditions[name]['pressure'] =='None' else np.float64(ambient_conditions[name]['pressure'])		
        temperature=None if ambient_conditions[name]['temperature'] =='None' else np.float64(ambient_conditions[name]['temperature'])
        wdir=None if ambient_conditions[name]['wdir'] =='None' else np.float64(ambient_conditions[name]['wdir'])	
        calibration_curve=None if ambient_conditions[name]['calibration_curve'] =='None' else np.float64(ambient_conditions[name]['calibration_curve'])
        mass_flow_controller=None if ambient_conditions[name]['mass_flow_controller'] =='None' else ambient_conditions[name]['mass_flow_controller']
        calibration_factor=None if ambient_conditions[name]['calibration_factor'] =='None' else np.float64(ambient_conditions[name]['calibration_factor'])
        scaling_factor=None if ambient_conditions[name]['scaling_factor'] =='None' else np.float64(ambient_conditions[name]['scaling_factor'])	
        scale=None if ambient_conditions[name]['scale'] =='None' else np.float64(ambient_conditions[name]['scale'])
        ref_length=None if ambient_conditions[name]['ref_length'] =='None' else np.float64(eval(ambient_conditions[name]['ref_length']))
        ref_height=None if ambient_conditions[name]['ref_height'] =='None' else np.float64(ambient_conditions[name]['ref_height'])	
        gas_name=None if ambient_conditions[name]['gas_name'] =='None' else ambient_conditions[name]['gas_name']
        mol_weight=None if ambient_conditions[name]['mol_weight'] =='None' else np.float64(ambient_conditions[name]['mol_weight'])
        gas_factor=None if ambient_conditions[name]['gas_factor'] =='None' else np.float64(ambient_conditions[name]['gas_factor'])
        full_scale_wtref=None if ambient_conditions[name]['full_scale_wtref'] =='None' else np.float64(ambient_conditions[name]['full_scale_wtref'])
        full_scale_flow_rate=None if ambient_conditions[name]['full_scale_flow_rate'] =='None' else np.float64(ambient_conditions[name]['full_scale_flow_rate'])	
		
        return x_source,y_source,z_source,x_measure,y_measure,z_measure,pressure,temperature,wdir,calibration_curve,mass_flow_controller,\
        calibration_factor, scaling_factor,scale,ref_length,ref_height,gas_name,mol_weight,\
        gas_factor,full_scale_wtref,full_scale_flow_rate
           		
    def ambient_conditions(self, x_source, y_source, z_source, x_measure, y_measure, z_measure, pressure, temperature, calibration_curve,
                           mass_flow_controller, calibration_factor=0):
        """ Collect ambient conditions during measurement. pressure in [Pa],
        temperature in [°C]. 
        
        Parameters
        ----------
        


        x_source: float
        y_source: float 
        z_source: float 
        x_measure: float 
        y_measure: float 
        z_measure: float 
        pressure: float 
        temperature: float 
        calibration_curve: float
        mass_flow_controller: float 
        calibration_factor: float
        
        """
        #edit 09/19/2019: new function,based on ambient_conditions in PointConcentration.py,
        #which collects ambient conditions during measurement. Pressure in [Pa]!	
        #edit 02/21/2020: added handling of variables for source and measurement locations, added calculation of distance variable            
        self.__check_sum = self.__check_sum + 1
        
        self.x_source = x_source
        self.y_source = y_source
        self.z_source = z_source
        self.x_measure = x_measure
        self.y_measure = y_measure
        self.z_measure = z_measure
        x = x_measure-y_source
        y = y_measure-y_source
        z = z_measure-z_source        
        self.x = x
        self.y = y
        self.z = z
        self.distance = np.sqrt(x**2 + y**2 + z**2)        
        self.pressure = pressure
        self.temperature = temperature
        self.calibration_curve = calibration_curve
        self.calibration_factor = calibration_factor
        self.mass_flow_controller = mass_flow_controller

    def scaling_information(self, scaling_factor, scale, ref_length, ref_height):
        """ Collect data necessary to scale the results. unit: [m], where
        applicable.
        
        Parameters
        ----------
        


        scaling_factor: float  
        scale: float  
        ref_length: float  
        ref_height: float 
        
        """
        #edit 09/19/2019: new function, based on scaling_information in PointConcentration.py,
        #which collects scaling data. Units (where applicable) is [m]	
        self.__check_sum = self.__check_sum + 1

        self.scaling_factor = scaling_factor
        self.scale = scale
        self.ref_length = ref_length
        self.ref_height = ref_height
        self.full_scale_ref_length = self.scale * self.ref_length	

    def tracer_information(self, gas_name, mol_weight, gas_factor):
        """ Collect information on tracer gas used during measurement.
        Molecular weight in [kg/mol]. 
        
        Parameters
        ----------
        
        

        gas_name: str
        mol_weight: float
        gas_factor: float
        
        """
        #edit 09/19/2019: new function, based on tracer_information in PointConcentration.py,
        #which collects tracer information. Units (where applicable) is [m].		
        self.__check_sum = self.__check_sum + 1

        self.gas_name = gas_name
        self.mol_weight = mol_weight
        self.gas_factor = gas_factor	

    def full_scale_information(self, full_scale_wtref, full_scale_flow_rate):
        """ Collect information on desired full scale information.
        full_scale_wtref in [m/s]. full_scale_flow_rate is automatically
        adjusted to standard atmosphere conditions.
        input in [kg/s], output in [m^3/s]. 
        
        Parameters
        ----------
        

        full_scale_wtref: float
        full_scale_flow_rate: float
        
        """
        #edit 09/19/2019: new function, based on full_scale_information in PointConcentration.py,
        #which collects information on full scale information. full_scale_wtref is in [m/s], 
        #full_scale_flow_rate takes input in [kg/s], and outputs the flow rate in m^3/s, adjusted
        #to ambient conditions
        self.__check_sum = self.__check_sum + 1

        self.full_scale_wtref = full_scale_wtref
        self.full_scale_flow_rate = full_scale_flow_rate	

    def convert_temperature(self):
        """ Convert ambient temperature to °K. """
        #edit 09/19/2019: new function, based on convert_temperature in PointConcentration.py,
        #which converts temperature, Also edited code to account for removal 
        #of variable kelvin_temperature. 
        self.temperature_K = self.temperature + 273.15
        self.standard_temp_K = self.standard_temp + 273.15



    def calc_model_mass_flow_rate(self):
        """ Calculate the model scale flow rate in [kg/s]. 
        
        Returns
        ----------
        
        
        self.mass_flow_rate: float 
        
        """
        #edit 09/19/2019: new function, based on calc_model_mass_flow_rate in PointConcentration.py,
        #which calculates the model scale mass flow rate in [kg/s]		
        self.__check_sum = self.__check_sum + 1

        self.mass_flow_rate = self.gas_factor * (np.mean(self.open_rate) *
                                                 self.calibration_curve +
                                                 self.calibration_factor) * \
                              self.temperature_K * self.standard_pressure / \
                              (self.pressure * self.standard_temp_K)

        return self.mass_flow_rate	

    def calc_full_scale_flow_rate(self):
        """ Convert flow rate to full scale flow rate in [m^3/s]. 
        
        Returns
        ----------
        
        self.full_scale_flow_rate: float 
        
        """
        #edit 09/19/2019: new function, based on calc_full_scale_mass_flow_rate in PointConcentration.py,
        #which calculates the full scale mass flow rate in [m^3/s]			
        self.full_scale_flow_rate = (self.full_scale_flow_rate * self.R *
                                     self.standard_temp_K) / \
                                    (self.standard_pressure * self.mol_weight)

        return self.full_scale_flow_rate
        
    def calc_non_dimensional_flow_rate(self):
        """ Convert flow rate to non-dimensional flow rate in [m^3/s]. 
        
        Returns
        ----------
        
        
        self.full_scale_flow_rate: float 
        
        """
        #edit 01/14/2020: new function, based on calc_full_scale_mass_flow_rate in PointConcentration.py,
        #which calculates the non-dimensional mass flow rate in [-]		
        #TODO: fix function!!        
        self.full_scale_flow_rate = (self.full_scale_flow_rate * self.R *
                                     self.standard_temp_K) / \
                                    (self.standard_pressure * self.mol_weight)

        return self.full_scale_flow_rate        

    def calc_c_star(self):
        """ Calculate dimensionless concentration. [-] 
        
        Returns
        ----------
        
        
        self.c_star: float 
        
        """
        #edit 09/19/2019: new function, based on calc_c_star in PointConcentration.py,
        #which calculates the dimensionless concentration [unitless]			
        self.__check_sum = self.__check_sum + 1
        # TODO: calc_mass_flow_rate (for Point, Line and Area)
        self.c_star = self.net_concentration * self.wtref_mean * \
                      self.ref_length ** 2 / self.mass_flow_rate * 1000 * 3600

        return self.c_star
		
    def calc_full_scale_concentration(self):
        """ Calculate full scale concentration in [ppmV].
        
        Returns
        ----------
        
        
        self.full_scale_concentration: float 
        
        """
        #edit 09/19/2019: new function, based on calc_full_scale_concentration in
        #PointConcentration.py, which calculates the full scale concentration [ppmV]		
        self.full_scale_concentration = self.c_star * \
                                        self.full_scale_flow_rate / \
                                        (self.full_scale_ref_length ** 2 *
                                         self.full_scale_wtref)	
		
        return self.full_scale_concentration
		
    def calc_wtref_mean(self):
        """ Calculate scaled wtref mean in [m/s].
        
        Returns
        ----------
        
        
        self.wtref_mean: float 

        """
       
        self.__check_sum = self.__check_sum + 1

        self.wtref_mean = self.scaling_factor * np.mean(self.wtref)

        return self.wtref_mean		
		
		
    def calc_full_scale_time(self):
        """ Calculate full scale timesteps in [s]. 
        
        Returns
        ----------
        self.full_scale_time: float 
        """
       		
        if self.wtref_mean is None:
            self.wtref_mean = PointConcentration.calc_wtref_mean()

        # Store model scale time before converting
        self.model_scale_time = self.time.copy()

        self.full_scale_time = self.full_scale_ref_length / self.ref_length * \
                               self.wtref_mean / self.full_scale_wtref * \
                               self.time

        return self.full_scale_time	

    def calc_non_dimensional_time(self):
        """ Calculate non-dimensional time step [-]. 
        
        Returns
        ----------
        
        
        self.non_dimensional_time: float 
        
        """        
        if self.wtref_mean is None:
            self.wtref_mean = PointConcentration.calc_wtref_mean()

        self.non_dimensional_time = self.wtref_mean / self.ref_length * \
                               self.time

        return self.non_dimensional_time		        
		

    def calc_net_concentration(self):
        """ Calculate net concentration in [ppmV]. """
		
        self.__check_sum = self.__check_sum + 1		
		
        self.net_concentration = self.fast_FID - self.slow_FID

		

    def detect_end_release_index(self):
        """ Detects the indices of the end of each release period. Returns a
        list containing the index of the last timestamp of each release 
        period. 
        
        Returns
        ----------
        
        
        self.end_release_index: list 
        
        """
      
        self.end_release_index =(np.argwhere(np.diff(self.signal) < 0).flatten()).tolist()
        if len(self.begin_release_index)> len(self.end_release_index):
            print('Time series terminates during puff release. '
                  'Ignoring last puff which continues beyond the end of the dataset!')
            self.begin_release_index = self.begin_release_index[:-1]
        return self.end_release_index

    def detect_end_release_period(self):
        """ Detects the end of each release period. Returns an np.array 
        containing the last timestamp of each release period. 
        
        Returns
        ----------
       

        self.end_release_period: np.arrray 
        
        """			
        indices = self.detect_end_release_index()	
        self.end_release_period = self.time[indices]
		

        return self.end_release_period

    def detect_begin_release_index(self):
        """ Detects the indices of the beginning of each release period. Returns a
        list containing the index of the first timestamp of each release 
        period.

        Returns
        ----------
        
        
        self.begin_release_index: list

        """

        self.begin_release_index = (np.argwhere(np.diff(self.signal) > 0).flatten() + 1).tolist()

        return self.begin_release_index

    def detect_begin_release_period(self):
        """ Detects the beginning of each release period. Returns an np.array 
        containing the first timestamp of each release period.
        
        Returns
        ----------
        
        
        self.begin_release_period: np.arrray
        
        """	
        indices = self.detect_begin_release_index()		
        self.begin_release_period = self.time[indices]

		
        return self.begin_release_period

    def calc_release_length(self):
        """ Calculate the length of each release period. Returns an np.array
        containing the duration of each release period. 
        
        Returns
        ----------
        
        
        self.release_length: np.arrray
        
        """

        beginning = self.detect_begin_release_period()
        end = self.detect_end_release_period()
        self.release_length = (end-beginning).tolist()

        return self.release_length

    def get_dosage(self):
        """ Calculates the dosage of each puff between two release times. """
			
        beginnings = self.begin_release_index
        self.dosage = []

      
        #for i in range(np.shape(beginnings)[0]):  
            #print(i)
            #if i<np.shape(beginnings)[0]-1:
               #self.dosage.append(self.net_concentration[beginnings[i]:beginnings[i+1]].sum())
            #else:
               #self.dosage.append(self.net_concentration[beginnings[i]:].sum())
		
		#original algorithm
        for i, begin in enumerate(beginnings):

            if i == np.size(beginnings) - 1:
                self.dosage.append(self.net_concentration[begin:].sum())				

            if i < np.size(beginnings) - 1:
                end = beginnings[i + 1]
                self.dosage.append(self.net_concentration[begin:end].sum())

      
        self.dt = np.mean(np.asarray(self.time[1:])-np.asarray(self.time[:-1]))
        self.dosage=[self.dt*l for l in self.dosage] 
 
                   
    def get_mean_puff(self):
        """Calcuate mean puff """
       
        puffs_start = np.asarray(self.begin_release_index)
        self.mean_puff = []
        self.mean_signal = []
        self.pct10_puff = []
        self.pct90_puff = []	
        self.pct10_signal = []
        self.pct90_signal = []		     	
        puffs_length=puffs_start[1:]-puffs_start[:-1]
        puffs_length_mask=np.transpose(self.mask)
        puffs_length_mask=puffs_length_mask[(np.asarray(puffs_length_mask)<(np.max(puffs_length_mask)))]                 
        puffs_length=puffs_length[puffs_length_mask]
        self.min_puff_length=int(np.nanmin(puffs_length))        
        puffs_start=puffs_start[self.mask]
        puffs_start=np.asarray(puffs_start,dtype=np.int64)
       
        self.puffs_array=np.zeros((np.shape(puffs_start)[0],self.min_puff_length))
        self.signal_array=np.zeros((np.shape(puffs_start)[0],self.min_puff_length))	  
		#Create array of all puffs, truncated to minimum puff length
        for i in range(np.shape(puffs_start)[0]-1):
            self.puffs_array[i,:]=self.net_concentration[puffs_start[i]:puffs_start[i]+self.min_puff_length]
            self.signal_array[i,:]=self.signal[puffs_start[i]:puffs_start[i]+self.min_puff_length]			

        self.mean_puff = np.nanmean(self.puffs_array,axis=0)
        self.mean_signal = np.nanmean(self.signal_array,axis=0)
        self.pct10_puff = np.percentile(self.puffs_array, 10, axis=0)
        self.pct90_puff = np.percentile(self.puffs_array, 90, axis=0)
        self.pct10_signal= np.percentile(self.signal_array, 10, axis=0)
        self.pct90_signal = np.percentile(self.signal_array, 90, axis=0)


    def detect_leaving_time(self,time_threshold=0.05):
        """ Detects the end of each puff. Returns an np.array 
        containing the last timestamp of each puff. 
        
        Parameters
        ----------
        

        time_threshold: float

        Returns
        ----------
        

        self.arrival_time: np.array
        self.arrival_index: list
        
        """
       		
        self.leaving_time = []
        self.leaving_index = []			

        if time_threshold != 0.05:
            print('Warning: threshold dosage used to compute characteristic start and end times set to '+str(100*time_threshold)+'%, which does not equal the default value of 5%. Consider using default value!')

        for i in range(len(self.begin_release_index)):

            begin = self.begin_release_index[i]
            if (i + 1) < len(self.begin_release_index):
                end = self.begin_release_index[i+1]
            else:
                end = len(self.net_concentration)

            index = begin + np.argwhere(np.nancumsum(
                self.net_concentration[begin:end])*self.dt > (1-time_threshold) * self.dosage[i])[0][0]

            self.leaving_index.append(index)
            self.leaving_time.append(self.time[index] - self.time[begin])

    def detect_arrival_time(self,time_threshold=0.05):
        """ Detects the beginning of each puff. Returns an np.array 
        containing the first timestamp of each puff. 
        
        Parameters
        ----------
        

        time_threshold: float

        Returns
        ----------
        

        self.arrival_time: np.array
        self.arrival_index: list
    
        """
          
        self.arrival_time = []
        self.arrival_index = []
		
        if time_threshold != 0.05:
            print('Warning: threshold dosage used to compute characteristic start and end times set to '+str(100*time_threshold)+'%, which does not equal the default value of 5%. Consider using default value!')
        for i in range(len(self.begin_release_index)):

            begin = self.begin_release_index[i]
            if (i + 1) < len(self.begin_release_index):
                end = self.begin_release_index[i+1]
            else:
                end = len(self.net_concentration)

            index = begin + np.argwhere(np.nancumsum(
                self.net_concentration[begin:end])*self.dt > time_threshold * self.dosage[i])[0][0]

            self.arrival_index.append(index)
            self.arrival_time.append(self.time[index] - self.time[begin])

    def get_residence_time(self):
        """ Calculate the residence time of each puff. Returns an np.array. """
        self.residence_time = [i - j for i, j in zip(self.leaving_time,
                                                     self.arrival_time)]

    def get_peak_concentration(self):
        """ Acquire peak concentration of each puff. Returns a list. """
      	
        self.peak_concentration = []
        for i, begin in enumerate(self.arrival_index):
            end = self.leaving_index[i]
  
            self.peak_concentration.append(
                self.net_concentration[begin:end].max())               

    def get_peak_time(self):
        """ Acquire peak time of each puff. Returns a list. """
        		
        self.peak_time = []
        log_peak=np.zeros(np.shape(self.arrival_index))		
        for i, begin in enumerate(self.arrival_index):				
            begin_release= self.begin_release_index[i]	
            if i == np.shape(log_peak)[0]-1:
               begin_next_release=-1
            else:			
               begin_next_release= self.begin_release_index[i+1]			
            end = self.leaving_index[i]	
            time = self.time[begin:end]           
            if self.net_concentration[begin_release:begin_next_release].max() > self.net_concentration[begin:end].max() or begin==end:            
               log_peak[i]=1             
         			
            if np.isnan(self.peak_concentration[i]) == 1:			 
               self.peak_time.append(np.nan)                      
            else:		   
               self.peak_time.append(float(time[np.where(
                self.net_concentration[begin:end] ==
                self.net_concentration[begin:end].max())]) -
                                  self.time[begin_release])            

        puff_size=np.size(self.begin_release_index)	              
        for i in sorted(np.where(log_peak==1)[0],reverse=True):
           self.arrival_index[i] = np.nan
           self.leaving_index[i] = np.nan 
           self.arrival_time[i] = np.nan
           self.leaving_time[i] = np.nan                
           self.peak_time[i] = np.nan
           self.peak_concentration[i] = np.nan           
           self.begin_release_index[i] = np.nan          
           #del self.end_release_index[i]  
           self.dosage[i] = np.nan          
           #self.begin_release_period[i]=np.nan           
           #self.end_release_period[i]=np.nan         
								  
  
        logger.info('Puffs with concentration before arrival time or after leaving time greater than peak concentration:\
 {} or {:.4f}%. Note that peak concentration is calculated only between arrival and leaving time.'.format(
            np.int64(log_peak.sum()),
            log_peak.sum() / puff_size * 100))            

    def get_ascent_time(self):
        """ Calculate the ascent time between arrrival time and peak time. 
        Returns an np.array. """
        self.ascent_time = [i - j for i, j in zip(self.peak_time,
                                                  self.arrival_time)]

    def get_descent_time(self):
        """ Calculate the ascent time between arrrival time and peak time. 
        Returns an np.array. """
        self.descent_time = [i - j for i, j in zip(self.leaving_time,
                                                   self.peak_time)]

    def offset_correction(self,method='mean'):
        """ Correct a non-zero offset in the concentration measured. """
   	
		
        avg_release = []
        beginnings = self.detect_begin_release_index()
        endings = self.detect_end_release_index()
        net_concentration_offset=self.net_concentration
        #begin = begin[:200]
        #end = end[:200]      
        for i, release_index in enumerate(zip(beginnings, endings)):
            begin=release_index[0]
            end_release=release_index[1]
            #print(i)
            #Caluclate and subtract offset individually for each puff			
            avg_release.append(np.nanmean(self.net_concentration[begin:end_release]))
            if method=='ind':			   
               if i == np.size(beginnings) - 1:		   
                  net_concentration_offset[begin:]=net_concentration_offset[begin:]-avg_release[i]				
               elif i < np.size(beginnings) - 1:		
                  end = beginnings[i + 1]				
                  net_concentration_offset[begin:end]=net_concentration_offset[begin:end]-avg_release[i]				  
        if method=='ind':
           []
        elif method=='mean':
           #subtract mean of individual offsets from entire time series. Necesarry to do this
           #outside the above loop because all individual offsets must be calculated first
           #before taking mean. 	   
           net_concentration_offset=net_concentration_offset-np.mean(avg_release)
           net_concentration_offset=net_concentration_offset-np.mean(avg_release)
        else:
           print('Error: Invalid offset calculation method. Currently only supported methods for calculating\
		    #offset are mean (mean of all individual offsets) and ind (subtract all offset invididually).')		
          		  
        #avg_release_concentration = np.nanmean(avg_release)		
        self.net_concentration = net_concentration_offset
        return self.net_concentration 
                
    def check_against_avg_puff(self):
        """ Check each puff against the average puff of the time series. 
        
        Returns
        ----------
        
        a
        self.puff_deviations: list
        
        """
        numbers = np.arange(np.size(self.dosage))
        puffs = {}
        puffs.fromkeys(numbers)

        for i, puff in enumerate(numbers):
            arrival = []
            leaving = []
            peak_t = []
            peak_c = []

            arrival.append([self.arrival_time[i] -
                            self.avg_arrival_time,
                            (self.arrival_time[i] - self.avg_arrival_time) /
                            self.avg_arrival_time * 100])

            leaving.append([self.leaving_time[i] -
                            self.avg_leaving_time,
                            (self.leaving_time[i] - self.avg_leaving_time) /
                            self.avg_leaving_time * 100])

            peak_t.append([self.peak_time[i] -
                           self.avg_peak_time,
                           (self.peak_time[i] - self.avg_peak_time) /
                           self.avg_peak_time * 100])

            peak_c.append([self.peak_concentration[i] -
                           self.avg_peak_concentration,
                           (self.peak_concentration[i] -
                            self.avg_peak_concentration) /
                           self.avg_peak_concentration * 100])

            quantities = ['arrival_time', 'leaving_time', 'peak_concentration',
                          'peak_time']
            puffs[puff] = {}
            puffs[puff].fromkeys(quantities)
            for quantity in quantities:
                if quantity == 'arrival_time':
                    puffs[puff]['arrival_time'] = arrival
                if quantity == 'leaving_time':
                    puffs[puff]['leaving_time'] = leaving
                if quantity == 'peak_time':
                    puffs[puff]['peak_time'] = peak_t
                if quantity == 'peak_concentration':
                    puffs[puff]['peak_concentration'] = peak_c

        self.puff_deviations = puffs

        return self.puff_deviations
		
    def get_mask(self, threshold_concentration=0., threshold_dosage=0.,n_exclude=None):	
        """ Return array that containts locations of unmaksed datapoints. """
        

        if n_exclude==None:
           n_exclude=np.int64(np.shape(self.peak_concentration)[0]/100)
           print("Using default setting of removing top "+str(n_exclude)+" values, representing 1% of concentration and dosage measurements, as these are considered to be outliers")
        else: 
           print(n_exclude)
           print("Using specified setting of removing top "+str(n_exclude)+" values, representing "+str(n_exclude/np.shape(self.peak_concentration)[0])+"% of concentration and dosage measurements, as these are considered to be outliers")
        norm_dist_peak_concentration=self.calc_norm_distribution('peak_concentration')
        self.max_peak_concentration=norm_dist_peak_concentration.ppf((np.shape(self.peak_concentration)[0]-n_exclude)/np.shape(self.peak_concentration)[0])
        norm_dist_dosage=self.calc_norm_distribution('dosage')
        self.max_dosage=norm_dist_dosage.ppf((np.shape(self.dosage)[0]-n_exclude)/np.shape(self.dosage)[0])

        self.mask=np.asarray(((np.asarray(self.peak_concentration) >
                        threshold_concentration) & (np.asarray(self.peak_concentration) <
                        self.max_peak_concentration)) & ((np.asarray(self.max_dosage) >
                        threshold_dosage) & (np.asarray(self.dosage) <
                        self.max_dosage))).nonzero() 
        #self.mask=np.asarray((np.asarray(self.peak_concentration) >
                        #threshold_concentration) & (np.asarray(self.dosage) >
                        #threshold_dosage)).nonzero() 

        beginnings = self.begin_release_index    
        peak_concentration_ts=np.zeros(np.shape(self.net_concentration))
        dosage_ts=np.zeros(np.shape(self.net_concentration))
        
        for i, value in enumerate(beginnings):
            begin = value
            if np.isnan(begin) == 1:
               continue             
            if i == np.size(beginnings) - 1:
                peak_concentration_ts[begin:]=self.peak_concentration[i]	
                dosage_ts[begin:]=self.dosage[i]                

            if i < np.size(beginnings) - 1:
                #end=self.end_release_index[i]            
                end = beginnings[i + 1]
                if np.isnan(beginnings[i + 1]) == 1:
                   continue                
                peak_concentration_ts[begin:end]=self.peak_concentration[i]  
                dosage_ts[begin:end]=self.dosage[i]    

        #edit 09/27/2019: set data before first puff to nan. This ensures it is removed by the threshold dosage and threshold dosage.         
        peak_concentration_ts[:self.begin_release_index_unmasked[0]]=np.nan          
        dosage_ts[:self.begin_release_index_unmasked[0]]=np.nan                
         
        self.mask_full=np.asarray((np.asarray(peak_concentration_ts) >
                        threshold_concentration) & (np.asarray(dosage_ts) >
                        threshold_dosage)).nonzero()
        
        concentration_size = np.size(self.net_concentration)        
        puff_size=np.size(self.begin_release_index)	   
        logger.info('Total values removed by applying threshold concentration and dosage: {} or {:.4f}% \n Total puffs removed by\
 applying threshold concentration and theshold dosage: {} or {:.4f}%'.format(
            concentration_size-np.shape(self.mask_full)[1],
            (concentration_size-np.shape(self.mask_full)[1]) / concentration_size * 100,
            puff_size-np.shape(self.mask)[1],
            (puff_size-np.shape(self.mask)[1]) / puff_size * 100             
        ))	               
                        
         
	
    def calc_norm_distribution(self,var):
        "calculate normal distribution using mean and standard deviation of a given variable"
       

        norm_dist=norm(np.nanmean(getattr(self,var)),np.nanstd(getattr(self,var)))

        return norm_dist

    def apply_threshold_concentration(self, threshold_concentration=0.):
        """ Apply a given threshold concentration to peak_concentration to 
        remove weak puffs, and maximum concentration based on statistical 
        analysis to remove outliers. The default value for threshold_concentration 
        is 0. (float). 
        
        Parameters
        ----------
        

        threshold_concentration: float
        
        """
	

        
        #norm_dist_peak_concentration=self.calc_norm_distribution('peak_concentration')
        #self.max_peak_concentration=norm_dist_peak_concentration.ppf((np.shape(self.peak_concentration)[0]-5)/np.shape(self.peak_concentration)[0])
        #print(max_peak_concentration)
        #print(max_dosage)
        #pause()

        self.threshold_concentration = threshold_concentration
        mask = np.where((np.asarray(self.peak_concentration) >
                        self.threshold_concentration) & (np.asarray(self.peak_concentration) <
                        self.max_peak_concentration))[0] 
						
        beginnings = self.begin_release_index    
        peak_concentration_ts=np.zeros(np.shape(self.net_concentration))	
        
        for i, value in enumerate(beginnings):
            begin = value
            if np.isnan(begin) == 1:
               print('skipping puff' + str(i))
               continue                           
            if i == np.size(beginnings) - 1:
                peak_concentration_ts[begin:]=self.peak_concentration[i]				

            if i < np.size(beginnings) - 1:
                #end=self.end_release_index[i]              
                end = beginnings[i + 1]
                if np.isnan(beginnings[i + 1]) == 1:
                   continue
                peak_concentration_ts[begin:end]=self.peak_concentration[i]
        
        #edit 09/27/2019: set data before first puff to nan. This ensures it is removed by the threshold concentration. 
        peak_concentration_ts[:self.begin_release_index_unmasked[0]]=np.nan               

        mask2 = np.where(np.asarray(peak_concentration_ts) > self.threshold_concentration)						
        
        self.dosage_unmasked = self.dosage	    
        self.dosage = np.asarray(self.dosage)[mask]        
        self.arrival_time = np.asarray(self.arrival_time)[mask]
        self.leaving_time = np.asarray(self.leaving_time)[mask]			
        self.ascent_time = np.asarray(self.ascent_time)[mask]
        self.descent_time = np.asarray(self.descent_time)[mask]
        self.peak_time = np.asarray(self.peak_time)[mask]
        self.peak_concentration = np.asarray(self.peak_concentration)[mask]    
        self.begin_release_index_masked = np.asarray(self.begin_release_index)[mask]          

		# Log outliers in console and to file
        concentration_size = np.size(self.net_concentration)
        puff_size=np.size(self.begin_release_index)
        logger.info('Values below threshold concentration: {} or {:.4f}% \n Puffs below theshold concentration: {} or {:.4f}%'.format(
            concentration_size-np.shape(mask2)[1],
            (concentration_size-np.shape(mask2)[1]) / concentration_size * 100,
            puff_size-np.shape(mask)[0],
            (puff_size-np.shape(mask)[0]) / puff_size * 100            
            
        ))

    def apply_threshold_dosage(self, threshold_dosage=0.):
	
        """ Apply a given threshold concentration to dosage to 
        remove weak puffs. The default value for threshold_concentration 
        is 0. (float).
        
        Parameters
        ----------
        
        
        threshold_dosage: float

        """
	
        
		
        #norm_dist_dosage=self.calc_norm_distribution('dosage')
        #max_dosage=norm_dist_dosage.ppf((np.shape(self.dosage)[0]-5)/np.shape(self.dosage)[0])


        self.threshold_dosage = threshold_dosage
        mask = np.where((np.asarray(self.dosage) >
                        self.threshold_dosage) & (np.asarray(self.dosage) <
                        self.max_dosage))[0] 
        #mask = np.where(np.asarray(self.dosage) >
                        #self.threshold_dosage)[0]
        mask_logger = np.where(np.asarray(self.dosage_unmasked) >
                        self.threshold_dosage)[0]                              
        beginnings = self.begin_release_index        
        dosage_ts=np.zeros(np.shape(self.net_concentration))	
        
        for i, value in enumerate(beginnings):        
            begin = value
            if np.isnan(begin) == 1:
               continue               
            if i == np.size(beginnings) - 1:
                dosage_ts[begin:]=self.dosage_unmasked[i]				

            if i < np.size(beginnings) - 1:
                #end=self.end_release_index[i]              
                end = beginnings[i + 1]
                if np.isnan(beginnings[i + 1]) == 1:
                   continue
                dosage_ts[begin:end]=self.dosage_unmasked[i]
				
         
        dosage_ts[:self.begin_release_index_unmasked[0]]=np.nan
        
        mask2 = np.where(np.asarray(dosage_ts) > self.threshold_dosage)
        
        self.dosage = np.asarray(self.dosage)[mask]
        self.arrival_time = np.asarray(self.arrival_time)[mask]
        self.leaving_time = np.asarray(self.leaving_time)[mask]
        self.ascent_time = np.asarray(self.ascent_time)[mask]
        self.descent_time = np.asarray(self.descent_time)[mask]
        self.peak_time = np.asarray(self.peak_time)[mask]
        self.peak_concentration = np.asarray(self.peak_concentration)[mask]
        self.begin_release_index_masked = np.asarray(self.begin_release_index_masked)[mask]         

		# Log outliers in console and to file
        concentration_size = np.size(self.net_concentration)        
        puff_size=np.size(self.begin_release_index)	   
        logger.info('Values below threshold dosage: {} or {:.4f}% \n Puffs below theshold dosage: {} or {:.4f}%'.format(
            concentration_size-np.shape(mask2)[1],
            (concentration_size-np.shape(mask2)[1]) / concentration_size * 100,
            puff_size-np.shape(mask_logger)[0],
            (puff_size-np.shape(mask_logger)[0]) / puff_size * 100             
        ))		
		
    def clear_zeros(self):
	    
        """ Clear and count zeros in concentration measurements. """
		
		
        concentration_size = np.size(self.net_concentration)

        # Mask zeros
        mask = self.net_concentration > 0
        
        self.time = self.time[np.asarray(mask)]
        self.net_concentration = self.net_concentration[mask]			
        self.signal = self.signal[np.asarray(mask)]    		


        # Log outliers in console and to file
        logger.info('Values below 0: {} or {:.4f}%'.format(
            np.size(np.where(~mask)),
            np.size(np.where(~mask)) / concentration_size * 100
        ))		
				
		
    def plot_puff(self,var1='net_concentration',n_puffs=5,path=None,name=None,full_scale=None,axis_range='auto'):
	
        """ Plot time series of selected variable for first n_puffs puffs.
        Default configuration plots net_concentration for first 5 puffs
        
        Parameters
        ----------
        
        
        var1: str
        n_puffs: int
        path: str
        name: str
        full_scale: str
        axis_range: str

        Returns
        ----------
        
        ret: plt.object
        
        """ 
       
        if hasattr(self,'signal')== False: 
           print('Error: No puff release signal found. Check integrity of dataset.')
           return
        if hasattr(self,var1)== False: 
           print('Error: Selected input varibale not found. Check function input!')
           return 	

        #Output warnings if attempting to plot more than 10 puffs. 
        if n_puffs == 'all':
           n_puffs=np.shape(self.dosage)[0]
           print('Warning: entire dataset being plotted, resulting in a total of '+str(n_puffs)+' plots. Only intedned to be used for diagnositc purposes. Plots will only be saved in specificed path, not displayed in console.')
        elif n_puffs > 10:
           print('Warning: a total of '+str(n_puffs)+' plots will be generated. Consider reducing the number of puffs to be plotted. Plots will only be saved in specificed path, not displayed in console.')	   

        index = 0 
        #plot time series							
        puffs_start=self.begin_release_index	
        #print(np.shape(self.begin_release_index))
        #print(np.shape(self.dosage))
        #print(self.mask[0])		
        for i in self.mask[0][:n_puffs]:
        #edit 02/18/2020: increase index by 1, even if puff is skipped. Previous version caused incorrect plotting of arrival and leaving times. 
            if np.isnan(puffs_start[i]) == 1:
               index=index+1                
               continue   
            if index<(np.shape(self.dosage)[0]-1):
               if np.isnan(puffs_start[i + 1])==1:
                  puff_end=puffs_start[i]+self.min_puff_length                
                  #index=index+1             
                  #continue 
               else:
                  puff_end=puffs_start[i+1]                
            else:
               puff_end=puffs_start[i]+self.min_puff_length             
            if index == np.size(puffs_start) - 1:
               ts_puffs=np.asarray(getattr(self,var1))[puffs_start[i]:]
               signal_n_puffs= np.asarray(self.signal)[puffs_start[i]:]
               time_puffs=np.asarray(self.time)[puffs_start[i]:]
			
			 
            if index < np.size(puffs_start) - 1:			
               #ts_puffs=np.asarray(getattr(self,var1))[puffs_start[i]:puffs_start[i+1]]
               #signal_n_puffs= np.asarray(self.signal)[puffs_start[i]:puffs_start[i+1]]
               #time_puffs=np.asarray(self.time)[puffs_start[i]:puffs_start[i+1]]	
               ts_puffs=np.asarray(getattr(self,var1))[puffs_start[i]:puff_end]
               signal_n_puffs= np.asarray(self.signal)[puffs_start[i]:puff_end]
               time_puffs=np.asarray(self.time)[puffs_start[i]:puff_end]
               if (np.asarray(np.shape(time_puffs)) < np.asarray(self.min_puff_length)):
               #edit 03/26/2020: skip puffs where length of puff is greater than previously
               #determined minimum puff length. This occurs mainly towwards the end of puff 
               #datasets. 
                  continue        
            ret=plt.figure(100+i)
            plt.ioff()
            plt.clf()
            plt.plot(time_puffs,ts_puffs,label=var1,linewidth=5,color='#1f77b4')
            print('Assumes 4.5V as signal threshold, same as in function "from_file"')
            plt.plot(time_puffs,(self.peak_concentration[index]/2)*signal_n_puffs,label='signal',linewidth=5,color='#ff7f0e')
            ret.set_figwidth(26)	
            ret.set_figheight(16)
            if full_scale == 'ms':
               xtick_step = np.int64((time_puffs[self.min_puff_length-1]-time_puffs[0])/5) 
            elif full_scale == 'fs':			   
               xtick_step = np.int64((time_puffs[self.min_puff_length-1]-time_puffs[0])/5) 
            #edit 02/04/2020: set x_tick_step to 500 for non-dimensional data.  
            elif full_scale == 'nd':			   
               xtick_step = np.int64((time_puffs[self.min_puff_length-1]-time_puffs[0])/5)                
            else: 
               print("Error: invalid input for full_scale. Data can only be computed in model scale (full_scale='ms'), full scale (full_scale='fs'), or non-dimensionally (full_scale='nd')") 		   
            #TODO: fix python error ufunc 'true_divide' not supported for the input types, and the inputs could not be safely coerced to any supported types
            #according to the casting rule ''safe'' (09/23/2019). Appears to have fixed itself (09/26/2019). Keep note in code in case the error reappears.  
            #edit 10/25/2019: fix error in x-axis tickmark labeling. Previously only tickmark labels converted to np.int, which led to incorrect tickmark labeling.				
            #Needed to be outcommented because of error: ValueError: Maxium allowed size exceeded
            #plt.xticks(np.arange(np.min(time_puffs),np.max(time_puffs),xtick_step,dtype=np.int64),np.arange(np.min(time_puffs),np.max(time_puffs),xtick_step,dtype=np.int64))
            plt.tick_params(axis='both', labelsize=30)   
            plt.xlim(np.min(time_puffs),np.max(time_puffs))
            if axis_range=='auto':
               []
            elif axis_range=='same':
               plt.ylim(0,1.05*np.asarray(getattr(self,var1).max()))   			
            #edit 07/26/2019: display arrival and leaving time in plot			
            if i >= 0:		           
               plt.axvline(x=self.begin_release_period[i]+self.arrival_time[index],linewidth=5,color='b',linestyle='--')
               plt.axvline(x=self.begin_release_period[i]+self.leaving_time[index],linewidth=5,color='b',linestyle='--')               
            ax=plt.gca()   
            #edit 07/26/2019: no puff before start of data aquisiton
            #edit 09/26/2019: added units to dosage        
            if full_scale=='ms':
               ax.set_title('Puff ' + str(np.int64(self.begin_release_period[i]))+' (Model Scale), Dosage of '+str(np.round(self.dosage[index],1)) + ' ppm$\mathrm{_v}$s',fontsize=40)
            elif full_scale=='fs':
               ax.set_title('Puff ' + str(np.int64(self.begin_release_period[i]))+' (Full Scale), Dosage of '+str(np.round(self.dosage[index],1)) + ' ppm$\mathrm{_v}$s',fontsize=40)
            elif full_scale=='nd':
               ax.set_title('Puff ' + str(np.int64(self.begin_release_period[i]))+' (Non-dimensional), Dosage of '+str(np.round(self.dosage[index],1)) ,fontsize=40)               
            else:
               print("Error: invalid input for full_scale. Data can only be computed in model scale (full_scale='ms'), full scale (full_scale='fs'), or non-dimensionally (full_scale='nd')")                
            plt.legend(fontsize=40)
            #edit 09/26/2019: added units to time
            if full_scale=='nd':
               plt.xlabel('time (-)',fontsize=40)
               plt.ylabel('Concentration (-)',fontsize=40)
               plt.show()
            elif full_scale=='fs' or full_scale=='ms':  
               plt.xlabel('time (s)',fontsize=40)
               plt.ylabel('Concentration (ppmV)',fontsize=40)
               plt.show()
            else:
               print("Error: invalid input for full_scale. Data can only be computed in model scale (full_scale='ms'), full scale (full_scale='fs'), or non-dimensionally (full_scale='nd')")    
			#edit 07/26/2019:save plots
            if path==None:
               []            
            else:
               if not os.path.exists(path+'Puff_Plots/'+name[:-9]+'/'):
                  os.makedirs(path+'Puff_Plots/'+name[:-9]+'/') 				
               if name==None:
                  print('Name of dataset not specified. Plots will not be saved to avoid confusion in the future.')	
               else:
                  if full_scale=='ms':           
                     plt.savefig(path+'Puff_Plots/'+name[:-9]+'/Puff_'+str(np.int64(self.begin_release_period[i]))+'_Model_Scale.png')
                     plt.show()
                  elif full_scale=='fs':
                     plt.savefig(path+'Puff_Plots/'+name[:-9]+'/Puff_'+str(np.int64(self.begin_release_period[i]))+'_Full_Scale.png') 
                  elif full_scale=='nd':
                     plt.savefig(path+'Puff_Plots/'+name[:-9]+'/Puff_'+str(np.int64(self.begin_release_period[i]))+'_Non_Dimensional.png')                     
                  else:
                     print("Error: invalid input for full_scale. Data can only be computed in model scale (full_scale='ms'), full scale (full_scale='fs'), or non-dimensionally (full_scale='nd')")                
                  if n_puffs > 10: 
                     plt.close()
                     
            
            plt.close()
            index=index+1             

        plt.show()  
        return ret 
		
    def plot_mean_puff(self,path,name,stats='off',dist='off',full_scale=None):    
	
        """ Plot time series of mean puff, calcualted above in get_mean_puff
        
        Parameters
        ----------
        
        
        path: str
        name: str
        stats: str
        dist: str
        full_scale: str

        Returns
        ----------
        
         
        ret: plt.object
        
        """ 
            
        if hasattr(self,'mean_puff')== False: 
           print('Error: mean_puff varibale not found. Check function input!')
           return 	

        ts_puffs=np.asarray(self.mean_puff)
        ts_pct10_puffs=np.asarray(self.pct10_puff)	
        ts_pct90_puffs=np.asarray(self.pct90_puff)			
        signal_n_puffs= np.asarray(self.mean_signal)	
        time_puffs=np.asarray(self.time)[:self.min_puff_length]		
        ret=plt.figure(1001)
        plt.clf()        
        plt.plot(time_puffs,ts_puffs,label='mean puff',linewidth=5,color='#1f77b4')
        print('Assumes 4.5V as signal threshold, same as in function "from_file"')
        plt.plot(time_puffs,(np.max(self.mean_puff))*self.mean_signal,label='mean signal',linewidth=5,color='#ff7f0e')
        ret.set_figwidth(26)	
        ret.set_figheight(16)
        if full_scale == 'ms':
           xtick_step = np.int64((time_puffs[self.min_puff_length-1]-time_puffs[0])/5)  
        elif full_scale == 'fs':
           xtick_step = np.int64((time_puffs[self.min_puff_length-1]-time_puffs[0])/5)  
        #edit 02/04/2020: set x_tick_step to 500 for non-dimensional data.      
        elif full_scale == 'nd':			   
           xtick_step = np.int64((time_puffs[self.min_puff_length-1]-time_puffs[0])/5)        
        else: 
           print("Error: invalid input for full_scale. Data can only be computed in model scale (full_scale='ms'), full scale (full_scale='fs'), or non-dimensionally (full_scale='nd')") 
        #edit 10/25/2019: fix error in x-axis tickmark labeling. Previously only tickmark labels converted to np.int, which led to incorrect tickmark labeling.			   
        #ValueError: Maxium allowed size exceeded
        #plt.xticks(np.arange(np.min(time_puffs),np.max(time_puffs),xtick_step,dtype=np.int64),np.arange(np.min(time_puffs),np.max(time_puffs),xtick_step,dtype=np.int64))
        plt.tick_params(axis='both', labelsize=30)   
        plt.xlim(np.min(time_puffs),np.max(time_puffs))	            
        #edit 08/02/2019: added optional plotting of 10th and 90th percentile of puff and signal, in addition to mean puff and mean signal
        if dist=='on':
            plt.plot(time_puffs,ts_pct10_puffs,linewidth=5,color='b',linestyle=':')
            plt.plot(time_puffs,ts_pct90_puffs,linewidth=5,color='b',linestyle=':')			
            plt.plot(time_puffs,(np.max(self.mean_puff))*self.pct10_signal,linewidth=5,color='b',linestyle=':')	
            plt.plot(time_puffs,(np.max(self.mean_puff))*self.pct90_signal,linewidth=5,color='b',linestyle=':')	
        elif dist=='off':
            []
        else: 	
            print('Error: Invalid input for variable dist. Dist can either be set to on (in which case the 10th and 90th\
            percentiles of the puff and signal will be plotted, or off, in which case only the variables mean_puff and mean_signal will me plotted.' )
            return
        ax=plt.gca()
        #edit 08/02/2019: added optional plotting of various statistics		
        if stats == 'on':			
            #edit 09/26/2019: added units to dosage          
            plt.axvline(x=np.mean(self.arrival_time),linewidth=5,color='b',linestyle='--')
            plt.axvline(x=np.mean(self.peak_time),linewidth=5,color='b',linestyle='--')			
            plt.axvline(x=np.mean(self.leaving_time),linewidth=5,color='b',linestyle='--')
            if full_scale=='ms':
               ax.set_title('Mean Puff (Model Scale), with Mean Dosage of '+str(np.round(np.mean(self.dosage),1)) + ' ppm $\mathrm{_v}$s',fontsize=40)
            elif full_scale=='fs':
               ax.set_title('Mean Puff (Full Scale), with Mean Dosage of '+str(np.round(np.mean(self.dosage),1)) + ' ppm $\mathrm{_v}$s',fontsize=40)
            elif full_scale=='nd':
               ax.set_title('Mean Puff (Non-Dimensional), with Mean Dosage of '+str(np.round(np.mean(self.dosage),1)) ,fontsize=40)               
            else:
               print("Error: invalid input for full_scale. Data can only be computed in model scale (full_scale='ms'), full scale (full_scale='fs'), or non-dimensionally (full_scale='nd')")             
        elif stats == 'off':         
            if full_scale=='ms':
               ax.set_title('Mean Puff (Model Scale)',fontsize=40)
            elif full_scale=='fs':
               ax.set_title('Mean Puff (Full Scale)',fontsize=40)
            elif full_scale=='nd':
               ax.set_title('Mean Puff (Non-Dimensional)',fontsize=40)               
            else:
               print("Error: invalid input for full_scale. Data can only be computed in model scale (full_scale='ms'), full scale (full_scale='fs'), or non-dimensionally (full_scale='nd')")             
        else:
            print('Error: Invalid input for variable stats. Stats can either be set to on (in which case mean arrival time, peak time,\
            leaving time, and dosage will be plotted, or off, in which case only the variables mean_puff and mean_signal will me plotted.' )
            return            
        plt.legend(fontsize=40)
        #edit 09/26/2019: added units to time 
        if full_scale=='nd':
            plt.xlabel('time (-)',fontsize=40)
            plt.ylabel('Concentration (-)',fontsize=40)
        elif full_scale=='fs' or full_scale=='ms':   
            plt.xlabel('time (s)',fontsize=40)
            plt.ylabel('Concentration (ppmV)',fontsize=40)
        else:
            print("Error: invalid input for full_scale. Data can only be computed in model scale (full_scale='ms'), full scale (full_scale='fs'), or non-dimensionally (full_scale='nd')")            			
        if path==None:
           []
        else:
           if not os.path.exists(path+'Puff_Plots/'+name[:-9]+'/'):
               os.makedirs(path+'Puff_Plots/'+name[:-9]+'/')         
           if name==None:
              print('Name of dataset not specified. Plot of mean puff will not be saved to avoid confusion in the future.')	
           else:	
              if full_scale=='ms':           
                 plt.savefig(path+'Puff_Plots/'+name[:-9]+'/Mean Puff, Model Scale.png')
              elif full_scale=='fs':
                 plt.savefig(path+'Puff_Plots/'+name[:-9]+'/Mean Puff, Full Scale.png')
              elif full_scale=='nd':
                 plt.savefig(path+'Puff_Plots/'+name[:-9]+'/Mean Puff, Non-Dimensional.png')                  
              else:
                 print("Error: invalid input for full_scale. Data can only be computed in model scale (full_scale='ms'), full scale (full_scale='fs'), or non-dimensionally (full_scale='nd')")                                
              plt.close()  
        return ret 
		
		
    def get_puffs(self):
        """ Returns DataFrame with all puff information. 
        
        Returns
        ----------
        
        return_data: pd.Dataframe
        
        """
      
        data = {'arrival time': self.arrival_time,
                'leaving time': self.leaving_time,
                'peak time': self.peak_time,
                'peak concentration': self.peak_concentration,
                'ascent time': self.ascent_time,
                'descent time': self.descent_time}
        return_data = pd.DataFrame(data=data)

        return return_data
    
		
    def save2file_ms(self, filename, out_dir=None):
        """ Save model scale data from PuffConcentration object to txt file.
        filename must include '.txt' ending. If no out_dir directory is
        provided './' is set as standard.

        Parameters
        ----------
        
        
        filename: str
        out_dir: str
        
        """
        if out_dir is None:
            out_dir = './'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        mask=self.mask
        puffs_header= "variables = \"  time [s] \"     "
        self.number = np.arange(1, np.size(self.begin_release_index) + 1)		
        for i in self.number[mask]:
        #for i in range(np.shape(self.ensemble_mean)[0]):
            if i<10:		
               puffs_header=puffs_header+ " \"       Puff "+str(i)+"       \""	
            elif i>=10 and i<100:		
               puffs_header=puffs_header+ " \"       Puff "+str(i)+"      \""	
            elif i>=100 and i<1000:		
               puffs_header=puffs_header+ " \"      Puff "+str(i)+"      \""
            elif i>=1000 and i<10000:		
               puffs_header=puffs_header+ " \"      Puff "+str(i)+"     \""
            elif i>=10000 and i<100000:		
               puffs_header=puffs_header+ " \"     Puff "+str(i)+"     \""				   
            else:			
               print('Warning: attempting to write ' + str(np.shape(self.number[mask])[0]) + ' ensembles to file. Program currently does not support writing more than 100000 ensembles to file.')			
            #puffs_header=puffs_header+ " \"Puff "+str(i)+"\""			
        output_file_puffs = out_dir + 'puffs_ms_' + filename	
        output_file_stats = out_dir + 'stats_ms_' + filename	
       
        np.savetxt(output_file_puffs, np.vstack((self.time[:self.min_puff_length],
                                           np.squeeze(self.puffs_array))
                                          ).transpose(),
                   fmt='%22.4f',
                   header="General concentration measurement data:" + '\n' +
                          "" + '\n' +
                          "geometric scale: 1:{}".format(float(self.scale))
                          + "" + '\n' +
                          "Variables: x (measurement relativ to source): {} [mm], y (measurement relativ to source): {} [mm], z (measurement relativ to source): {} [mm], "
                          "x_source: {} [mm], y_source: {} [mm], z_source: {} [mm], "
                          "x_measure: {} [mm], y_measure: {} [mm], z_measure: {} [mm], " 
                          "distance beteween source and measurement: {} [mm],"                           
                          "ambient temperature: {:.1f} [°C], ambient pressure: {:.2f} [Pa],"
                          " mass flow rate {:.4f} [kg/s], "
                          "reference length (model): {:.4f} [m], "
                          "Tracer gas: {}, mol. weight tracer: {:.4f} [mol/kg], "
                          "gas factor: {:.6f}, "
						  #calibartion curve: {:.6f}, "
                          "wtref: {:.4f} [m/s]".format(self.x, self.y, self.z,
                                                       self.x_source, self.y_source, self.z_source,
                                                       self.x_measure, self.y_measure, self.z_measure,
                                                       self.distance,                                                        
                                                       self.temperature,
                                                       self.pressure,
                                                       self.mass_flow_rate,
                                                       self.ref_length,
                                                       self.gas_name,
                                                       self.mol_weight,
                                                       self.gas_factor,
                                                       #self.calibration_curve,
                                                       self.wtref_mean)
                          + "" + '\n' +
                          puffs_header,comments='')	
						  

        np.savetxt(output_file_stats, np.vstack((self.number[mask],
                                           self.arrival_time,
                                           self.leaving_time,
                                           self.peak_time,
                                           self.peak_concentration,
                                           self.ascent_time,
                                           self.descent_time)
                                          ).transpose(),
                   fmt='%28.4f',
                   header="General puff concentration measurement data:" + '\n' +
                          "" + '\n' +
                          "Variables: average arrival time: {:.4f}, "
                          "average leaving time: {:.4f}, average peak time {:.4f}, "
                          "average peak concentration: {:.4f}, "
                          "threshold concentration: {:.4f}, threshold_dosage: {:.4f}".format(
                              self.avg_arrival_time,
                              self.avg_leaving_time,
                              self.avg_peak_time,
                              self.avg_peak_concentration,
                              self.threshold_concentration,
                              self.threshold_dosage)
                          + "" + '\n' +
                          "variables = \"    puff number             \" \"    arrival time [s]    \" \"     leaving time [s]     \" \"      peak time [s]       \" "
                          "\"peak concentration [ppm_v]\" \"     ascent time [s]      \" \"     descent time [s]     \"",comments='')					  
						  
    def save2file_fs(self, filename, out_dir=None):
        """ Save full scale data from PuffConcentration object to txt file.
        filename must include '.txt' ending. If no out_dir directory is
        provided './' is set as standard.

        Parameters
        ----------
        
        
        filename: str
        out_dir: str
        
        """
		
        if out_dir is None:
            out_dir = './'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        mask=self.mask
        puffs_header= "variables = \"  time [s] \"     "
        self.number = np.arange(1, np.size(self.begin_release_index) + 1)		
        for i in self.number[mask]:
        #for i in range(np.shape(self.ensemble_mean)[0]):
            if i<10:		
               puffs_header=puffs_header+ " \"       Puff "+str(i)+"       \""	
            elif i>=10 and i<100:		
               puffs_header=puffs_header+ " \"       Puff "+str(i)+"      \""	
            elif i>=100 and i<1000:		
               puffs_header=puffs_header+ " \"      Puff "+str(i)+"      \""
            elif i>=1000 and i<10000:		
               puffs_header=puffs_header+ " \"      Puff "+str(i)+"     \""
            elif i>=10000 and i<100000:		
               puffs_header=puffs_header+ " \"     Puff "+str(i)+"     \""				   
            else:			
               print('Warning: attempting to write ' + str(np.shape(self.number[mask])[0]) + ' ensembles to file. Program currently does not support writing more than 100000 ensembles to file.')			
            #puffs_header=puffs_header+ " \"Puff "+str(i)+"\""			
        output_file_puffs = out_dir + 'puffs_fs_' + filename	
        output_file_stats = out_dir + 'stats_fs_' + filename
        #edit 03/05/2020: moved masking of puffs array to get_mean_puff function		
        np.savetxt(output_file_puffs, np.vstack((self.time[:self.min_puff_length],
                                           np.squeeze(self.puffs_array))
                                          ).transpose(),
                   fmt='%22.4f',
                   header="General concentration measurement data:" + '\n' +
                          "" + '\n' +
                          "geometric scale: 1:{}".format(float(self.scale))
                          + "" + '\n' +
                          "Variables: x (measurement relativ to source): {} [mm], y (measurement relativ to source): {} [mm], z (measurement relativ to source): {} [mm], "
                          "x_source: {} [mm], y_source: {} [mm], z_source: {} [mm], "
                          "x_measure: {} [mm], y_measure: {} [mm], z_measure: {} [mm], " 
                          "distance beteween source and measurement: {} [mm],"                           
                          "ambient temperature: {:.1f} [°C], ambient pressure: {:.2f} [Pa],"
                          " mass flow rate {:.4f} [kg/s], "
                          "reference length (model): {:.4f} [m], "
                          "Tracer gas: {}, mol. weight tracer: {:.4f} [mol/kg], "
                          "gas factor: {:.6f}, "
						  #calibartion curve: {:.6f}, "
                          "wtref: {:.4f} [m/s]".format(self.x, self.y, self.z,
                                                       self.x_source, self.y_source, self.z_source,
                                                       self.x_measure, self.y_measure, self.z_measure,
                                                       self.distance,                                                        
                                                       self.temperature,
                                                       self.pressure,
                                                       self.mass_flow_rate,
                                                       self.ref_length,
                                                       self.gas_name,
                                                       self.mol_weight,
                                                       self.gas_factor,
                                                       #self.calibration_curve,
                                                       self.wtref_mean)
                          + "" + '\n' +
                          puffs_header,comments='')	
						  

        np.savetxt(output_file_stats, np.vstack((self.number[mask],
                                           self.arrival_time,
                                           self.leaving_time,
                                           self.peak_time,
                                           self.peak_concentration,
                                           self.ascent_time,
                                           self.descent_time)
                                          ).transpose(),
                   fmt='%28.4f',
                   header="General puff concentration measurement data:" + '\n' +
                          "" + '\n' +
                          "Variables: average arrival time: {:.4f}, "
                          "average leaving time: {:.4f}, average peak time {:.4f}, "
                          "average peak concentration: {:.4f}, "
                          "threshold concentration: {:.4f}, threshold_dosage: {:.4f}".format(
                              self.avg_arrival_time,
                              self.avg_leaving_time,
                              self.avg_peak_time,
                              self.avg_peak_concentration,
                              self.threshold_concentration,
                              self.threshold_dosage)
                          + "" + '\n' +
                          "variables = \"    puff number             \" \"    arrival time [s]    \" \"     leaving time [s]     \" \"      peak time [s]       \" "
                          "\"peak concentration [ppm_v]\" \"     ascent time [s]      \" \"     descent time [s]     \"",comments='')		

    def save2file_nd(self, filename, out_dir=None):
        """ Save non-dimensional data from PuffConcentration object to txt file.
        filename must include '.txt' ending. If no out_dir directory is
        provided './' is set as standard.
        
        Parameters
        ----------
        
        
        filename: str
        out_dir:str
        
        """
		#edit 02/25/2020: new function, similar to save2file_ms and save2file_fs, but for non-dimensional data     
        if out_dir is None:
            out_dir = './'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        mask=self.mask
        puffs_header= "variables = \"  time [-] \"     "
        self.number = np.arange(1, np.size(self.begin_release_index) + 1)		
        for i in self.number[mask]:
        #for i in range(np.shape(self.ensemble_mean)[0]):
            if i<10:		
               puffs_header=puffs_header+ " \"       Puff "+str(i)+"       \""	
            elif i>=10 and i<100:		
               puffs_header=puffs_header+ " \"       Puff "+str(i)+"      \""	
            elif i>=100 and i<1000:		
               puffs_header=puffs_header+ " \"      Puff "+str(i)+"      \""
            elif i>=1000 and i<10000:		
               puffs_header=puffs_header+ " \"      Puff "+str(i)+"     \""
            elif i>=10000 and i<100000:		
               puffs_header=puffs_header+ " \"     Puff "+str(i)+"     \""				   
            else:			
               print('Warning: attempting to write ' + str(np.shape(self.number[mask])[0]) + ' ensembles to file. Program currently does not support writing more than 100000 ensembles to file.')			
            #puffs_header=puffs_header+ " \"Puff "+str(i)+"\""			
        output_file_puffs = out_dir + 'puffs_nd_' + filename	
        output_file_stats = out_dir + 'stats_nd_' + filename	
        #edit 03/05/2020: moved masking of puffs array to get_mean_puff function        
        np.savetxt(output_file_puffs, np.vstack((self.time[:self.min_puff_length],
                                           np.squeeze(self.puffs_array))
                                          ).transpose(),
                   fmt='%22.4f',
                   header="General concentration measurement data:" + '\n' +
                          "" + '\n' +
                          "geometric scale: 1:{}".format(float(self.scale))
                          + "" + '\n' +
                          "Variables: x (measurement relativ to source): {} [mm], y (measurement relativ to source): {} [mm], z (measurement relativ to source): {} [mm], "
                          "x_source: {} [mm], y_source: {} [mm], z_source: {} [mm], "
                          "x_measure: {} [mm], y_measure: {} [mm], z_measure: {} [mm], " 
                          "distance beteween source and measurement: {} [mm],"                           
                          "ambient temperature: {:.1f} [°C], ambient pressure: {:.2f} [Pa],"
                          " mass flow rate {:.4f} [kg/s], "
                          "reference length (model): {:.4f} [m], "
                          "Tracer gas: {}, mol. weight tracer: {:.4f} [mol/kg], "
                          "gas factor: {:.6f}, "
						  #calibartion curve: {:.6f}, "
                          "wtref: {:.4f} [m/s]".format(self.x, self.y, self.z,
                                                       self.x_source, self.y_source, self.z_source,
                                                       self.x_measure, self.y_measure, self.z_measure,
                                                       self.distance,                                                        
                                                       self.temperature,
                                                       self.pressure,
                                                       self.mass_flow_rate,
                                                       self.ref_length,
                                                       self.gas_name,
                                                       self.mol_weight,
                                                       self.gas_factor,
                                                       #self.calibration_curve,
                                                       self.wtref_mean)
                          + "" + '\n' +
                          puffs_header,comments='')	
						  

        np.savetxt(output_file_stats, np.vstack((self.number[mask],
                                           self.arrival_time,
                                           self.leaving_time,
                                           self.peak_time,
                                           self.peak_concentration,
                                           self.ascent_time,
                                           self.descent_time)
                                          ).transpose(),
                   fmt='%28.4f',
                   header="General puff concentration measurement data:" + '\n' +
                          "" + '\n' +
                          "Variables: average arrival time: {:.4f}, "
                          "average leaving time: {:.4f}, average peak time {:.4f}, "
                          "average peak concentration: {:.4f}, "
                          "threshold concentration: {:.4f}, threshold_dosage: {:.4f}".format(
                              self.avg_arrival_time,
                              self.avg_leaving_time,
                              self.avg_peak_time,
                              self.avg_peak_concentration,
                              self.threshold_concentration,
                              self.threshold_dosage)
                          + "" + '\n' +
                          "variables = \"    puff number             \" \"    arrival time [-]    \" \"     leaving time [-]     \" \"      peak time [-]       \" "
                          "\"  peak concentration [-]  \" \"     ascent time [-]      \" \"     descent time [-]     \"",comments='')	        
    @property
    def max_puffs(self):
        """ Get maximum number of puffs. Deduced from the length of 
        release_length. 
        
        ----------
        Returns

        self.release_length: float
        
        """

        return np.size(self.release_length)

    @property
    def avg_arrival_time(self):
        """ Get average arrival time. 
        
        ----------
        Returns

        self.arrival_time: float
        
        """

        return np.nanmean(self.arrival_time)

    @property
    def avg_leaving_time(self):
        """ Get average leaving time. 
        
        Returns
        ----------
        

        self.leaving_time: float
        
        """

        return np.nanmean(self.leaving_time)

    @property
    def avg_peak_time(self):
        """ Get average peak time. 
        
        Returns
        ----------
        

        self.peak_time: float
        
        """

        return np.nanmean(self.peak_time)

    @property
    def avg_peak_concentration(self):
        """ Get average peak concentration. 
        
        Returns
        ----------
        

        self.peak_concentration: float
        
        """

        return np.nanmean(self.peak_concentration)

    @property
    def avg_ascent_time(self):
        """ Get average ascent time. 
        
        Returns
        ----------
        

        self.ascent_time: float
        
        """

        return np.nanmean(self.ascent_time)

    @property
    def avg_descent_time(self):
        """ Get average descent time. 
        
        Returns
        ----------
        

        self.descent_time: float
        
        """

        return np.nanmean(self.descent_time)

    def to_full_scale(self):
        self.time = self.full_scale_time

    def analyze_temporal_characteristics(self):
        """Analyze temporal characteristics of concentration signal"""
        # Autocorrelation and integral timescale
        self.autocorr = pd.Series(self.net_concentration).autocorr()
        
        # Spectral analysis
        freqs, psd = sc.signal.welch(self.net_concentration, 
                                    fs=1.0/np.mean(np.diff(self.time)))
        self.frequency = freqs
        self.power_spectral_density = psd
        
        # Detect dominant frequencies
        self.dominant_freq = freqs[np.argmax(psd)]

