#! /usr/bin/python3
import numpy as np
import math
import logging
import os
import pandas as pd
import windtunnel as wt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import string
import scipy.stats as sc
import scipy.signal as scSignal
from typing import Optional

# Create logger
logger = logging.getLogger()
__all__ = ['PointConcentration']


# %%#
class PointConcentration(pd.DataFrame):
    """ PointConcentration is a class that holds data collected during
    a continuous release point concentration measurement. The class can hold
    the raw time series, the corresponding wtref and all other quantities
    necessary to analyse the time series. All the information in a
    PointConcentration object can be saved to a txt file.

    Parameters
    ----------
    
    time: np.array
    wtref: np.array
    fast_FID: np.array
    slow_FID: np.array
    open_rate: np.array
    
    
    
    """

    def __init__(self, time, wtref, slow_FID, fast_FID, open_rate):
             
        """ Initialise PointConcentration object. """
        super().__init__()

        self['slow_FID'] = pd.Series(data=slow_FID)  #[ppmV]
        self['fast_FID'] = pd.Series(data=fast_FID)  #[ppmV]

        self.x = None         #[mm]
        self.y = None         #[mm]
        self.z = None         #[mm]
        self.x_source = None  #[mm]
        self.y_source = None  #[mm]
        self.z_source = None  #[mm]
        self.x_measure = None #[mm]
        self.y_measure = None #[mm]
        self.z_measure = None #[mm]         
        self.distance = None         
        self.scale = None     #[-]
        self.wtref_mean = None #[m/s]

        self.time = time      
        self.open_rate = open_rate
        self.wtref = wtref
        
        self.net_concentration = None        #[ppmV]
        self.c_star = None                   #[-]
        self.full_scale_concentration = None #[ppmV]
              
        self.model_scale_time = None      #[s]   #2025: Because of previous logic, save original time in model_scale_time variable, if full_scale= fs/nd /transformed, since self.time overwritten with self.non_dim_time/full_scale
        self.non_dimensional_time = None  #[-] 
        self.full_scale_time = None       #[s] 

        self.calibration_curve = None
        self.calibration_factor = None
        
        self.mass_flow_controller = None  #[l/h]*1/100    #2025: Previously only placeholder, now used to give original possible max. flow rate of controller (f.e. 150l/h -> 150)
        self.mass_flow_rate = None        #[l/h]_amb. #ambient conditions, calibration and gas adjusting of flow rate
       
        self.gas_factor = None  #[-]         #Used to use same settings if f.e. different gas is used
        self.gas_name = None                 #Only description placeholder variable
        self.mol_weight = None  #[g/mol]
        self.temperature = None   #[°C]
        self.temperature_K = None #[K] #placeholder for transforming input to kelvin
        self.pressure = None      #[Pa]
    
        self.ref_height = None    #[m]
        self.ref_length = None    #[m]
        self.full_scale_ref_length = None #[m]
        self.scaling_factor = None        #[-]

        self.full_scale_flow_rate = None   #[kg/s]
        self.full_scale_wtref = None       #[m/s]
        self.full_scale_temp= None   #[°C]    #2025: Add ambient temperature and pressure for full scale transformation
        self.full_scale_temp_K= None #[K]   
        self.full_scale_pressure = None    #[Pa]

        self.standard_temp = 0 #20?  #[°C]
        self.standard_temp_K = None  
        self.standard_pressure = 101325 #[Pa]
        self.R = 8.3144621 #[J/molK]  #universal gas constant      #2025: kJ/kg not unit of universal gas constant
        self.__check_sum = 0

    def __repr__(self):
        """ Return the x, y and z coordinate of the PointConcentration
        object. """
        return 'PointConcentration (x={x}, y={y}, z={z})'.format(x=self.x,
                                                                 y=self.y,
                                                                 z=self.z)

    def __eq__(self, other):
        """ Two PointConcentration objects are considered equal, if their x, y
        and z coordinates are the same. """
        return self.x == other.x and self.y == other.y and self.z == other.z

    @classmethod
    def from_file(cls, filename):
        """ Create PointConcentration object from file. open_rate is converted
        to %."""
	   
        #TODO: open rate = flow rate?		
        time, wtref, slow_FID, fast_FID, open_rate = np.genfromtxt(filename,
                                                                   usecols=(0, 1, 2, 3, 4),
                                                                   unpack=True)

        return cls(time, wtref, slow_FID, fast_FID, open_rate)

    def to_full_scale(self):
        """Convert to full scale while preserving original data"""
        #if not hasattr(self, 'model_scale_time'):
        self.model_scale_time = self.time.copy()
    
        if self.__check_sum >= 8:

            #Scale measurement locations to full scale
            self.x = self.x * self.scale / 1000  # [m]
            self.y = self.y * self.scale / 1000  # [m]
            self.z = self.z * self.scale / 1000  # [m]
            self.x_source = self.x_source * self.scale / 1000  # [m]
            self.y_source = self.y_source * self.scale / 1000  # [m]
            self.z_source = self.z_source * self.scale / 1000  # [m]  
            self.x_measure = self.x_measure * self.scale / 1000  # [m]
            self.y_measure = self.y_measure * self.scale / 1000  # [m]
            self.z_measure = self.z_measure * self.scale / 1000  # [m]  
            self.distance = self.distance * self.scale / 1000  # [m]                                  
            #self.calc_full_scale_flow_rate() 

            self.calc_full_scale_time()			
            self.calc_full_scale_concentration() 
            #self.calc_full_scale_flow_rate()	            

            self.time = self.full_scale_time

            self.net_concentration = self.full_scale_concentration
        
        else:
            raise Exception('Please enter or calculate all full scale data '
                            'necessary!')
                            
                


    def to_non_dimensional(self):
        """ Converts all quantities to non-dimensional, overwriting model scale variables."""
	
        
        if self.__check_sum >= 8:

            #Scale measurement locations to undim.
           self.x = self.x / self.ref_length  # [-]
           self.y = self.y / self.ref_length  # [-]
           self.z = self.z / self.ref_length  # [-]
           self.x_source = self.x_source / self.ref_length  # [-]
           self.y_source = self.y_source / self.ref_length  # [-]
           self.z_source = self.z_source / self.ref_length  # [-]    
           self.x_measure = self.x_measure / self.ref_length  # [-]
           self.y_measure = self.y_measure / self.ref_length  # [-]
           self.z_measure = self.z_measure / self.ref_length  # [-]                

           #Calculate for for use relevant entdimensionalised variables
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
		example_puff_measurement.py."""	
		
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
           print('Error: Cannont find csv file containing ambient conditions in specified directory. Check name and/or location of ambient \
                conditions file. Resorting to input data in example_puff_measurement.py')	
           return		   
        else:	
           ambient_conditions=pd.read_csv(input_file,sep=',',index_col=0) 
        
        if name not in ambient_conditions.keys():
           print('Error: Dataset not found in csv file. Check to make sure that csv file to make sure that the csv file contains all necessary \
                data and is properly formatted. Resorting to input data in example_puff_measurement.py')
           return	

        #list of all variables output by read_ambient_conditions fuction.  
        necessary_keys={'x_source','y_source','z_source','x_measure','y_measure','z_measure','pressure','temperature','calibration_curve','mass_flow_controller','calibration_factor', \
        'scaling_factor','scale','ref_length','ref_height','gas_name','mol_weight','gas_factor','full_scale_wtref','full_scale_flow_rate','full_scale_temp','full_scale_pressure'}
        if not all(name2 in ambient_conditions[name] for name2 in necessary_keys):
           print('Error: csv file does not contain all necessary ambient conditions data. Check to make sure that csv file to make sure that \
                the csv file contains all necessary data and is properly formatted. Resorting to input data in example_puff_measurement.py')
           return			   
       		

        return ambient_conditions	

    def read_ambient_conditions(ambient_conditions,name):
        """Populate individual variables representing ambient conditions based on data
		in ambient_conditions array. """	
    
        x_source=None if ambient_conditions[name]['x_source'] =='None' else np.float64(ambient_conditions[name]['x_source'])
        y_source=None if ambient_conditions[name]['y_source'] =='None' else np.float64(ambient_conditions[name]['y_source'])
        z_source=None if ambient_conditions[name]['z_source'] =='None' else np.float64(ambient_conditions[name]['z_source'])  
        x_measure=None if ambient_conditions[name]['x_measure'] =='None' else np.float64(ambient_conditions[name]['x_measure'])
        y_measure=None if ambient_conditions[name]['y_measure'] =='None' else np.float64(ambient_conditions[name]['y_measure'])
        z_measure=None if ambient_conditions[name]['z_measure'] =='None' else np.float64(ambient_conditions[name]['z_measure']) 
        pressure=None if ambient_conditions[name]['pressure'] =='None' else np.float64(ambient_conditions[name]['pressure'])		
        temperature=None if ambient_conditions[name]['temperature'] =='None' else np.float64(ambient_conditions[name]['temperature'])
        calibration_curve=None if ambient_conditions[name]['calibration_curve'] =='None' else np.float64(ambient_conditions[name]['calibration_curve'])
        mass_flow_controller=None if ambient_conditions[name]['mass_flow_controller'] =='None' else np.float64(ambient_conditions[name]['mass_flow_controller'])
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
        full_scale_temp=None if ambient_conditions[name]['full_scale_temp'] =='None' else np.float64(ambient_conditions[name]['full_scale_temp'])
        full_scale_pressure=None if ambient_conditions[name]['full_scale_pressure'] =='None' else np.float64(ambient_conditions[name]['full_scale_pressure'])	
		

        return x_source,y_source,z_source,x_measure,y_measure,z_measure,pressure,temperature,calibration_curve,mass_flow_controller,\
        calibration_factor, scaling_factor,scale,ref_length,ref_height,gas_name,mol_weight,\
        gas_factor,full_scale_wtref,full_scale_flow_rate,full_scale_temp,full_scale_pressure							

    def ambient_conditions(self, x_source, y_source, z_source, x_measure,y_measure,z_measure,pressure, temperature, calibration_curve,
                           mass_flow_controller, calibration_factor=0):
        """ Collect ambient conditions during measurement. pressure in [Pa],
        temperature in [°C]. """
       
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
        applicable."""
        self.__check_sum = self.__check_sum + 1

        self.scaling_factor = scaling_factor
        self.scale = scale
        self.ref_length = ref_length
        self.ref_height = ref_height
        self.full_scale_ref_length = self.scale * self.ref_length

    def tracer_information(self, gas_name, mol_weight, gas_factor):
        """ Collect information on tracer gas used during measurement.
        Molecular weight in [kg/mol]. """
        #edit 10/24/2019: highly doubtful that weighth of tracer gas is in kg/mol.
        #Typical units for molecular weight are g/mol, 28.97 kg/mol seems outrageously 
        #large a value. 
        #TODO: verify units of molecular weight of gas		
        self.__check_sum = self.__check_sum + 1

        self.gas_name = gas_name
        self.mol_weight = mol_weight
        self.gas_factor = gas_factor

    def full_scale_information(self, full_scale_wtref, full_scale_flow_rate, full_scale_temp,full_scale_pressure):
        """ Collect information on desired full scale information.
        full_scale_wtref in [m/s]. full_scale_flow_rate is automatically
        adjusted to standard atmosphere conditions.
        input in [kg/s], output in [m^3/s]. """

        self.__check_sum = self.__check_sum + 1

        self.full_scale_wtref = full_scale_wtref
        self.full_scale_flow_rate = full_scale_flow_rate

        self.full_scale_temp = full_scale_temp #2025: Add temperature and pressure for fs transformation
        self.full_scale_pressure =   full_scale_pressure

    def convert_temperature(self):
        """ Convert ambient temperature to °K.
        
        """
        #edit 09/19/2019: edited code to account for removal 
        #of variable kelvin_temperature. 		
        self.temperature_K = self.temperature + 273.15
        self.standard_temp_K = self.standard_temp + 273.15
        self.full_scale_temp_K = self.full_scale_temp + 273.15 #2025: Add fs ambient conditions

    def calc_model_mass_flow_rate(self,usingMaxFlowRate="True",applyCalibration="False"):
        """ Calculates the ambient model scale flow rate in Q_amb[l/h] 
        from max flow rate of controller in Q[l/h]*1/100, open rateQ[%:0-10] and ambient temperature and pressure
        taking into account the open_rate, calibration of shutter, ambient temperature pressure and gas factor(Ethan/N2..)
        
        #usingMaxFlowRate: Do calculation based on (maxFlowRate[l/h]*1/100) (mass_flow_controller) overgiven for controller 
        #applyCalibration: Apply overgiven calibration(curve + factor)

        Returns
        ----------
        self.mass_flow_rate: float 
        
         """
        self.__check_sum = self.__check_sum + 1

        #If no gas factor given, ignore
        if(self.gas_factor == None):
            self.gas_factor = 1.0

        #Open_rate * 10 because of range

        #Do calculation based only on (maxFlowRate[l/h]*1/100) overgiven for controller 
        if(usingMaxFlowRate=="True"):
            
            #If calibration values given choose to apply these to correct maxFlowRate
            if(applyCalibration=="False"):
                self.mass_flow_rate = (np.mean(self.open_rate) * 10 * \
                self.gas_factor * self.mass_flow_controller ) * \
                self.temperature_K * self.standard_pressure / \
                (self.pressure * self.standard_temp_K)

            elif(applyCalibration=="True" and self.calibration_curve!=None and self.calibration_factor!=None):
                self.mass_flow_rate = (np.mean(self.open_rate) * 10 * \
                self.gas_factor * self.mass_flow_controller  * \
                self.calibration_curve +self.calibration_factor)  * \
                self.temperature_K * self.standard_pressure / \
                (self.pressure * self.standard_temp_K)

            else:
                print("If applyCalibration remember to set calibration_curve and calibration_factor in the ambient conditions file/code-value")


        #Legacy code: Do calculation only based on given calibration curve and factor which include maxFlowRate of Controller also
        else:
            #Calculation for calibration correction (curve + factor) given
            if(applyCalibration=="True" and self.calibration_curve !=None  and self.calibration_factor != None): #Or calibration_correction
                self.mass_flow_rate = self.gas_factor * (np.mean(self.open_rate) * 10 * \
                self.calibration_curve +self.calibration_factor)  * \
                self.temperature_K * self.standard_pressure / \
                (self.pressure * self.standard_temp_K)
        
            else:
                print("usingMaxFlowRate and/or applyCalibration have to be set to True. If applyCalibration remember to set calibration_curve and calibration_factor in the ambient conditions file/code-value")

                #self.mass_flow_rate = self.gas_factor * (np.mean(self.open_rate) ) * 10 * \
                #self.temperature_K * self.standard_pressure / \
                #(self.pressure * self.standard_temp_K)



        return self.mass_flow_rate

    #Only converts the full scale flow rate to m³/s!!!
    #def _full_scale_flow_rate(self):
    #    """ Convert full scale flow rate unit from kg/s given to m^3/s.
    #    
    #    Returns
    #    ----------
    #    

    #    self.full_scale_flow_rate: float 
        
    #     """
        
    #    self.full_scale_flow_rate = (self.full_scale_flow_rate * self.R *
    #                                 self.standard_temp_K) / \
    #                                (self.standard_pressure * self.mol_weight)

    #    #kg/s -> m³/s 
    #    #self.full_scale_flow_rate = (self.full_scale_flow_rate * self.R *
        #                             self.standard_temp_K) / \
        #                            (self.standard_pressure * self.mol_weight)

    #    return self.full_scale_flow_rate




    #Not needed at the moment  
    #def calc_non_dimensional_flow_rate(self):
        """ Convert flow rate to non-dimensional flow rate [-]. 
        
        Returns
        ----------
        

        self.full_scale_flow_rate: float 
        
        """
       	
        #TODO: fix function!!        
        #self.full_scale_flow_rate = (self.full_scale_flow_rate * self.R *
        #                             self.standard_temp_K) / \
        #                            (self.standard_pressure * self.mol_weight)

    #    return self.full_scale_flow_rate            

    def calc_net_concentration(self):
        """Calculate net concentration with validation
        
        Returns:
            Net concentration array
        """

        self.__check_sum = self.__check_sum + 1		


        if not hasattr(self, 'fast_FID'):
            raise AttributeError("fast_FID data required")
        if not hasattr(self, 'slow_FID'):
            raise AttributeError("slow_FID data required")
        
        self.net_concentration = self.fast_FID - self.slow_FID
        
        # Validate concentration range
        #if min_threshold is not None:
        #    self.net_concentration[self.net_concentration < min_threshold] = 0
        #if max_threshold is not None:
        #    self.net_concentration[self.net_concentration > max_threshold] = max_threshold
        
        return self.net_concentration

    def calc_c_star(self):
        """ Calculate dimensionless concentration. [-] 
        
        Returns
        ----------
        

        self.c_star: float
        
        """
        self.__check_sum = self.__check_sum + 1
        
        #Transform to correct units before = ppmV/1.000.000, /(1000*3600): l/h -> m³/s
        self.c_star =((self.net_concentration/1000000) * self.wtref_mean * self.ref_length ** 2 ) / (self.mass_flow_rate / (1000 * 3600) )  


        # TODO: calc_mass_flow_rate (for Point, Line and Area)
        #self.c_star = self.net_concentration  * self.wtref_mean * \
        #              self.ref_length ** 2 / self.mass_flow_rate * 1000 * 3600

        return self.c_star

    #passt
    def calc_full_scale_concentration(self):
        """ Input: Given full scale flow rate [kg/s], wtref[m/s] and ref length [m]
        #Calculate full scale concentration in [ppmV].
        
        Returns
        ----------
        

        self.full_scale_concentration: float

         """
        if not hasattr(self, 'c_star'):
            raise AttributeError("Calculate c_star before!")
        

        #Transform unit from input full scale flow[kg/s] to [m³/s] for calculation and use full scale ambient conditions
        self.full_scale_flow_rate = (self.full_scale_flow_rate * self.R *self.full_scale_temp_K) / \
                                    (self.full_scale_pressure * self.mol_weight * 1/1000)  #mol weight M: [g/mol]->[kg/mol] 


        #Actual calculation
        self.full_scale_concentration = self.c_star * \
                                        self.full_scale_flow_rate / \
                                        (self.full_scale_ref_length ** 2 *
                                         self.full_scale_wtref) * 1000000   #*1.000.000 ->back to ppmV	
        
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
        """Calculate full scale timesteps in [s]."""
        #if not hasattr(self, 'model_scale_time'):
        self.model_scale_time = self.time.copy()

        if self.wtref_mean is None:
            self.wtref_mean = PointConcentration.calc_wtref_mean()
        
        # Use consistent scaling formula across all concentration measurements
        self.full_scale_time = self.scale * self.wtref_mean / self.full_scale_wtref * \
                              self.model_scale_time
        
        return self.full_scale_time

    def calc_non_dimensional_time(self):
        """Calculate non-dimensional time step [-]."""
        #if not hasattr(self, 'model_scale_time'):
        self.model_scale_time = self.time.copy()

        if self.wtref_mean is None:
            self.wtref_mean = PointConcentration.calc_wtref_mean()

        
        self.non_dimensional_time = self.wtref_mean / self.ref_length * \
                               self.model_scale_time
        
        return self.non_dimensional_time
    

    def clear_zeros(self):
	    
        """ Clear and count zeros in concentration measurements."""
		 
        concentration_size = np.size(self.net_concentration)

        # Mask zeros
        mask = self.net_concentration > 0
        
        self.time = self.time[np.asarray(mask)]			
        self.net_concentration = self.net_concentration[mask]			


        # Log outliers in console and to file
        logger.info('Values below 0: {} or {:.4f}%'.format(
            np.size(np.where(~mask)),
            np.size(np.where(~mask)) / concentration_size * 100
        ))


    
    #2025: Add Percentile calculation for concentration distribution
    def calc_percentiles(self, percentiles=[10, 90], var='net_concentration'):
        """Calculate specified percentiles for the given variable."""
        data = getattr(self, var)
        #data = self.net_concentration 
        results = {}
        for p in percentiles:
            results[p] = np.percentile(data, p)
        return results
    

    def plot_hist_conc(self,n_classes=None,var='net_concentration',path=None,name=None):
        """Creates a historgram point concentration, i.e. continuous release, data.
        
        Parameters
        ----------
        
        
        n_classe: int
        var: str
        path: str 
        name: str

        Returns
        ----------
        

        ret: axes object

        """
       
        #TODO: add units to class mean label		

       

        data=getattr(self,var)
        data_size=np.size(data)		
       

        if n_classes == None:
           n_classes=np.int64(1+math.log10(data_size)/math.log10(2))	

        #calculate class width, class min, class max, and class mean
        class_width=(np.max(data)-np.min(data))/n_classes
        class_min=[np.min(data)]
        class_max=[np.min(data)+class_width]		
        for i in range(n_classes-1):
            class_min=np.append(class_min,class_min[i]+class_width)
            class_max=np.append(class_max,class_max[i]+class_width)	
        class_mean=(class_min+class_max)/2
			
		
        #get class frequency, both absolute and normalized
        class_freq=np.zeros(np.shape(class_min),dtype=np.int64)	
        class_freq_cum=np.zeros(np.shape(class_min),dtype=np.int64)
        class_freq_norm=np.zeros(np.shape(class_min),dtype=np.int64)	
        class_freq_cum_norm=np.zeros(np.shape(class_min),dtype=np.int64)			
        for i in range(n_classes):		
            class_freq[i]=((class_min[i] <= data) & (data < class_max[i])).sum()	
            class_freq_cum[i]= (data < class_max[i]).sum()	
		
        class_freq_norm=class_freq/data_size	
        class_freq_cum_norm= class_freq_cum/data_size	
	
        #plot frequency distribution
        
        var_label=var.replace("_"," ")		
        ret=plt.figure(301)			
        plt.clf()	
        plt.bar(np.linspace(1,np.shape(class_freq_norm)[0],np.shape(class_freq_norm)[0]),class_freq_norm,width=1,align='center')		
        ax=plt.gca()		
        ax.set_title('Frequency Distribution (Model Scale) of '+string.capwords(var_label),fontsize=40)              			
                				
        ret.set_figwidth(26)	
        ret.set_figheight(16)								
        n=n_classes		
        plt.xticks(np.linspace(1,n,n),np.round(class_mean,1))	
        plt.tick_params(axis='both', labelsize=30)
        		
        plt.xlim(0.5,n+0.5)	
        plt.ylim(0,1)
        plt.xlabel('Class Mean',fontsize=40)
        plt.ylabel('Frequency',fontsize=40)		   		
        if path=='none':
           []
        else:	
           if name=='none':
              print('Name of dataset not specified. Plot of frequency distribution will not be saved to avoid confusion in the future.')	
           else:		
              plt.savefig(path + var + '.jpg')
			  
        #plot cumulative frequency distribution
		
        ret=plt.figure(302)			
        plt.clf()	
        plt.bar(np.linspace(1,np.shape(class_freq_norm)[0],np.shape(class_freq_norm)[0]),class_freq_cum_norm,width=1,align='center')		
        ax=plt.gca()		
        ax.set_title('Cumulative Frequency Distribution (Model Scale) of '+string.capwords(var_label),fontsize=40)              			
                				
        ret.set_figwidth(26)	
        ret.set_figheight(16)								
        n=n_classes		
        plt.xticks(np.linspace(1,n,n),np.round(class_mean,1))	
        plt.tick_params(axis='both', labelsize=30)
        		
        plt.xlim(0.5,n+0.5)	
        plt.ylim(0,1)
        plt.xlabel('Class Mean',fontsize=40)
        plt.ylabel('Cumulative Frequency',fontsize=40)		   		
        if path=='none':
           []
        else:	
           if name=='none':
              print('Name of dataset not specified. Plot of frequency distribution will not be saved to avoid confusion in the future.')	
           else:		
              plt.savefig(path + var + '_cumulative.jpg')		

        return ret		

    def save2file_ms(self, filename, out_dir=None):
        """ Save model scale data from PointConcentration object to txt file.
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

        output_file = out_dir + '_ms_' + filename		
        np.savetxt(output_file, np.vstack((self.time,
                                           self.c_star,
                                           self.net_concentration)
                                          ).transpose(),
                   fmt='%.4f',
                   header="General concentration measurement data:" + '\n' +
                          "" + '\n' +
                          "geometric scale: 1:{}".format(float(self.scale))
                          + "" + '\n' +
                          "Variables: x (measurement relativ to source): {} [mm], y (measurement relativ to source): {} [mm], z (measurement relativ to source): {} [mm], "
                          "x_source: {} [mm], y_source: {} [mm], z_source: {} [mm], "
                          "x_measure: {} [mm], y_measure: {} [mm], z_measure: {} [mm], " 
                          "ambient temperature: {:.1f} [°C], ambient pressure: {:.2f} [Pa],"
                          " mass flow rate {:.4f} [kg/s], "
                          "reference length (model): {:.4f} [m], "
                          "Tracer gas: {}, mol. weight tracer: {:.4f} [mol/kg], "
                          "gas factor: {:.6f}, calibartion curve: {:.6f}, "
                          "wtref: {:.4f} [m/s]".format(self.x, self.y, self.z,
                                                       self.x_source, self.y_source, self.z_source,
                                                       self.x_measure, self.y_measure, self.z_measure,
                                                       self.temperature,
                                                       self.pressure,
                                                       self.mass_flow_rate,
                                                       self.ref_length,
                                                       self.gas_name,
                                                       self.mol_weight,
                                                       self.gas_factor,
                                                       self.calibration_curve,
                                                       self.wtref_mean)
                          + "" + '\n' +
                          "\"time [ms]\" \"c_star [-]\" \"net_concentration [ppmV]\" ")

    def save2file_fs(self, filename, out_dir=None):
        """ Save full scale and model scale data from PointConcentration object
        to txt file. filename must include '.txt' ending. If no out_dir
        directory is provided './' is set as standard.
        
        Parameters
        ----------
        
        
        filename: str
        out_dir: str
        
        """
        #edit 05/13/2020: added seperate handling of source and measurement locations        
        if self.__check_sum < 8:
            raise Exception('Please enter or calculate all full scale data '
                            'necessary!')

        if out_dir is None:
            out_dir = './'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        output_file = out_dir + '_fs_' + filename		
        np.savetxt(output_file, np.vstack((self.full_scale_time,
                                           self.c_star,
                                           self.net_concentration,
                                           self.full_scale_concentration)
                                          ).transpose(),
                   fmt='%.4f',
                   header="General concentration measurement data:" + '\n' +
                          "" + '\n' +
                          "geometric scale: 1:{}".format(float(self.scale))
                          + "" + '\n' +
                          "Variables: x (measurement relativ to source): {} [mm], y (measurement relativ to source): {} [mm], z (measurement relativ to source): {} [mm], "
                          "x_source: {} [mm], y_source: {} [mm], z_source: {} [mm], "
                          "x_measure: {} [mm], y_measure: {} [mm], z_measure: {} [mm], " 
                          "ambient temperature: {:.1f} [°C], ambient pressure: {:.2f} [Pa],"
                          " mass flow rate {:.4f} [kg/s], "
                          "reference length (model): {:.4f} [m], "
                          "reference length (full-scale): {:.4f} [m], Tracer gas: {}, "
                          "mol. weight tracer: {:.4f} [mol/kg], "
                          "gas factor: {:.6f}, calibartion curve: {:.6f}, "
                          "wtref: {:.4f} [m/s], full scale flow rate: {:.4f} [m^3/s]".format(self.x, self.y, self.z,
                              self.x_source, self.y_source, self.z_source,
                              self.x_measure, self.y_measure, self.z_measure,
                              self.temperature,
                              self.pressure,
                              self.mass_flow_rate,
                              self.ref_length,
                              self.full_scale_ref_length,
                              self.gas_name,
                              self.mol_weight,
                              self.gas_factor,
                              self.calibration_curve,
                              self.wtref_mean,
                              self.full_scale_flow_rate)
                          + "" + '\n' +
                          "\"full scale time [s]\" \"c_star [-]\" "
                          "\"net_concentration [ppmV]\" \"full_scale_concentration [ppmV]\"")
                          
    def save2file_nd(self, filename, out_dir=None):
        """ Save non-dimensional data from PointConcentration object
        to txt file. filename must include '.txt' ending. If no out_dir
        directory is provided './' is set as standard.
        
        Parameters
        ----------
        
        
        filename: str
        out_dir: str
        
        """
            
        if self.__check_sum < 8:
            raise Exception('Please enter or calculate all full scale data '
                            'necessary!')

        if out_dir is None:
            out_dir = './'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        output_file = out_dir + '_nd_' + filename		
        np.savetxt(output_file, np.vstack((self.time,
                                           self.net_concentration)
                                          ).transpose(),
                   fmt='%.4f',
                   header="General concentration measurement data:" + '\n' +
                          "" + '\n' +
                          "geometric scale: 1:{}".format(float(self.scale))
                          + "" + '\n' +
                          "Variables: x (measurement relativ to source): {} [mm], y (measurement relativ to source): {} [mm], z (measurement relativ to source): {} [mm], "
                          "x_source: {} [mm], y_source: {} [mm], z_source: {} [mm], "
                          "x_measure: {} [mm], y_measure: {} [mm], z_measure: {} [mm], " 
                          "ambient temperature: {:.1f} [°C], ambient pressure: {:.2f} [Pa],"
                          " mass flow rate {:.4f} [kg/s], "
                          "reference length (model): {:.4f} [m], "
                          "reference length (full-scale): {:.4f} [m], Tracer gas: {}, "
                          "mol. weight tracer: {:.4f} [mol/kg], "
                          "gas factor: {:.6f}, calibartion curve: {:.6f}, "
                          "wtref: {:.4f} [m/s], full scale flow rate: {:.4f} [m^3/s]".format(self.x, self.y, self.z,
                              self.x_source, self.y_source, self.z_source,
                              self.x_measure, self.y_measure, self.z_measure,
                              self.temperature,
                              self.pressure,
                              self.mass_flow_rate,
                              self.ref_length,
                              self.full_scale_ref_length,
                              self.gas_name,
                              self.mol_weight,
                              self.gas_factor,
                              self.calibration_curve,
                              self.wtref_mean,
                              self.full_scale_flow_rate)
                          + "" + '\n' +
                          "\"non-dimensional time [s]\" "
                          "\"net_concentration [-]]\"")                          

    def save2file_avg(self, filename, out_dir=None):
        """ Save average full scale and model scale data from
        PointConcentration object to txt file. filename must include '.txt'
        ending. If no out_dir directory is provided './' is set as standard.
        
        Parameters
        ----------
        
        
        filename: str
        out_dir: str
        
        """
       
        #if self.__check_sum < 8:
        #    raise Exception('Please enter or calculate all full scale data '
        #                    'necessary!')

        if out_dir is None:
            out_dir = './'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        output_file = out_dir + '_avg_' + filename

        np.savetxt(output_file, np.vstack((np.nanmean(self.c_star),
                                           np.nanmean(self.net_concentration),
                                           np.nanmean(
                                               self.full_scale_concentration))
                                          ).transpose(),
                   fmt='%.4f',
                   header="General concentration measurement data:" + '\n' +
                          "" + '\n' +
                          "geometric scale: 1:{}".format(float(self.scale))
                          + "" + '\n' +
                          "Variables: x (measurement relativ to source): {} [mm], y (measurement relativ to source): {} [mm], z (measurement relativ to source): {} [mm], "
                          "x_source: {} [mm], y_source: {} [mm], z_source: {} [mm], "
                          "x_measure: {} [mm], y_measure: {} [mm], z_measure: {} [mm],"
                          "ambient temperature: {:.1f} [°C], ambient pressure: {:.2f} [Pa],"
                          "mass flow rate {:.4f} [kg/s], "
                          "reference length (model): {:.4f} [m], "
                          "reference length (full-scale): {:.4f} [m], Tracer gas: {}, "
                          "mol. weight tracer: {:.4f} [mol/kg], "
                          "gas factor: {:.6f}, calibartion curve: {:.6f}, "
                          "wtref: {:.4f} [m/s], full scale flow rate: {:.4f} [m^3/s]".format(self.x, self.y, self.z,
                              self.x_source, self.y_source, self.z_source,
                              self.x_measure, self.y_measure, self.z_measure,
                              self.temperature,
                              self.pressure,
                              self.mass_flow_rate,
                              self.ref_length,
                              self.full_scale_ref_length,
                              self.gas_name,
                              self.mol_weight,
                              self.gas_factor,
                              self.calibration_curve,
                              self.wtref_mean,
                              self.full_scale_flow_rate)
                          + "" + '\n' +
                          "\"c_star [-]\" \"net_concentration [ppmV]\" "
                          "\"full_scale_concentration [ppmV]\"")


    def save2file_fullStats(self, filename, out_dir=None):
        """ Save average full scale and model scale calculated stats data (mean, percentage,skews..) from
        PointConcentration object to txt file. filename must include '.txt'
        ending. If no out_dir directory is provided './' is set as standard.
        
        Parameters
        ----------
        
        
        filename: str
        out_dir: str
        
        """
       
        #if self.__check_sum < 8:
        #    raise Exception('Please enter or calculate all full scale data '
        #                    'necessary!')

        if out_dir is None:
            out_dir = './'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        output_file = out_dir + '_stats_' + filename

        np.savetxt(output_file, np.vstack((
                                           np.nanmean(self.c_star),
                                           np.nanmean(self.net_concentration),
                                           np.nanmean(
                                               self.full_scale_concentration),
                                           np.percentile(self.c_star,95),
                                           np.percentile(self.net_concentration,95),
                                           np.percentile(self.full_scale_concentration,95),
                                           np.percentile(self.c_star,5),
                                           np.percentile(self.net_concentration,5),
                                           np.percentile(self.full_scale_concentration,5),
                                           np.max(self.c_star) / np.mean(self.c_star),
                                           np.max(self.net_concentration) / np.mean(self.net_concentration),
                                           np.max(self.full_scale_concentration) / np.mean(self.full_scale_concentration)
                                            )).transpose(),
                   fmt='%.4f',
                   header="General concentration measurement data:" + '\n' +
                          "" + '\n' +
                          "geometric scale: 1:{}".format(float(self.scale))
                          + "" + '\n' +
                          "Variables: x (measurement relativ to source): {} [mm], y (measurement relativ to source): {} [mm], z (measurement relativ to source): {} [mm], "
                          "x_source: {} [mm], y_source: {} [mm], z_source: {} [mm], "
                          "x_measure: {} [mm], y_measure: {} [mm], z_measure: {} [mm],"
                          "ambient temperature: {:.1f} [°C], ambient pressure: {:.2f} [Pa],"
                          "mass flow rate {:.4f} [kg/s], "
                          "reference length (model): {:.4f} [m], "
                          "reference length (full-scale): {:.4f} [m], Tracer gas: {}, "
                          "mol. weight tracer: {:.4f} [mol/kg], "
                          "gas factor: {:.6f}, calibartion curve: {:.6f}, "
                          "wtref: {:.4f} [m/s], full scale flow rate: {:.4f} [m^3/s]".format(self.x, self.y, self.z,
                              self.x_source, self.y_source, self.z_source,
                              self.x_measure, self.y_measure, self.z_measure,
                              self.temperature,
                              self.pressure,
                              self.mass_flow_rate,
                              self.ref_length,
                              self.full_scale_ref_length,
                              self.gas_name,
                              self.mol_weight,
                              self.gas_factor,
                              self.calibration_curve,
                              self.wtref_mean,
                              self.full_scale_flow_rate)
                          + "" + '\n' +
                          "\"Means: c_star [-]\" \"net_concentration [ppmV]\" "
                          "\"full_scale_concentration [ppmV]\""
                          + "" + #'\n' +
                          "\"Percentiles 95: c_star [-]\" \"net_concentration [ppmV]\" "
                          "\"full_scale_concentration [ppmV]\""
                          + "" + #'\n' +
                          "\"Percentiles 5: c_star [-]\" \"net_concentration [ppmV]\" "
                          "\"full_scale_concentration [ppmV]\""
                          + "" + #'\n' +
                          "\"Peak2MeanRatio: c_star [-]\" \"net_concentration [ppmV]\" "
                          "\"full_scale_concentration [ppmV]\"")



    def calculate_turbulence_intensity(self,dimensionless="False",returnDistribution="False",returnMetrics="False"):
        """
        Calculate turbulence intensity from wind measurements
        If dimensionless=True:  entdimensionlise
        Default: Returns: Dictionary containing turbulence intensity metrics
        If returnDistribution=True: Calculates and gives back also distribution as normal distribution as array
        If returnMetrics=True: Calculates and gives back Dictionary of turbulence intensity and metrics of distribution (mean,std,skew,kurtosis)
        """
        if(dimensionless=="True"):
            wtref = ( self.wtref / self.ref_length ) * self.time
        else:
            wtref = self.wtref

         # Calculate turbulence intensity I
        turbulence_intensity = np.std(wtref) / np.mean(wtref)

        ##Further options
        # Calculate distribution parameters as normal distributed
        if(returnDistribution=="True"):
            print("currently not implemented")
            #x = np.linspace(min(wtref), max(wtref), 100)
            #turbulence_intensity_distribution = sc.norm.pdf(x, loc=np.mean(wtref), scale=np.std(wtref))
            return #return turbulence_intensity_distribution
    
        # Calculates and gives back dictionary of metrics of turbulence intensity distribution
        if(returnMetrics=="True"):
            skewness = sc.skew(wtref)
            kurtosis = sc.kurtosis(wtref)

            turbulence_intensity_metrics = {
                'std_velocity': np.std(wtref),
                'mean_velocity': np.mean(wtref),
                'turbulence_intensity': turbulence_intensity,
                'skewness': skewness,
                'kurtosis': kurtosis
            }
            return turbulence_intensity_metrics 

        #Else just return turbulence_intensity value
        return turbulence_intensity
    

    def calculate_intermittency(self, dimensionless=False, threshold_method='mean',threshold_type="percentage", threshold_factor=1.5):
        """
        Calculate intermittency factor for concentration time series.
        
        Parameters:
        -----------
        dimensionless : bool, optional
            Whether to use dimensionless concentration (c_star) or net_concentration
        threshold_method : str, optional
            Method to determine threshold ('mean', 'median', 'std')
        threshold_factor : float, optional
            Factor to multiply with threshold base value
            
        Returns:
        --------
        dict: Dictionary containing intermittency metrics
        """
        time = self.time
        # Select if concentration data dimensionless or not
        if dimensionless:
            concentration = self.c_star
        else:
            concentration = self.net_concentration
        
        #Calculate threshold based on given ratio (threshold)
        if threshold_type=="ratio":
            if threshold_method == 'mean':
                threshold = np.mean(concentration) * threshold_factor
            elif threshold_method == 'median':
                threshold = np.median(concentration) * threshold_factor
            elif threshold_method == 'std':
                threshold = np.mean(concentration) + np.std(concentration) * threshold_factor
            else:
                raise ValueError("threshold_method must be 'mean', 'median', or 'std'")
        #Threshold given as absolute value
        elif threshold_type=="absolute":
            threshold = threshold_factor
         
        # Calculate intermittency factor/indicator of intermittency
        above_threshold = concentration > threshold  #as fraction of time signal exceeds threshold
        intermittency_factor = np.mean(above_threshold)
        #intermittency_factor = np.sum(above_threshold) / len(above_threshold)

        # Identify threshold crossings for event analysis
        crossings = np.diff(above_threshold.astype(int))
        rising_edges = np.where(crossings == 1)[0]
        falling_edges = np.where(crossings == -1)[0]

        if len(rising_edges) > 0 and len(falling_edges) > 0:
            if rising_edges[0] > falling_edges[0]:
                falling_edges = falling_edges[1:]
            if len(rising_edges) > len(falling_edges):
                rising_edges = rising_edges[:len(falling_edges)]
            
            #Event durations
            event_durations = time[falling_edges] - time[rising_edges]
            mean_duration = np.mean(event_durations) if len(event_durations) > 0 else 0
            #Length between events
            if len(rising_edges) > 1:
                event_intervals = time[rising_edges[1:]] - time[falling_edges[:-1]] 
                mean_interval = np.mean(event_intervals) if len(event_intervals) > 0 else 0
            else:
                print("Not enough rising edges for interval detection (found)")
                mean_interval = 0
        else:
            print("Not enough events for event detection (found)")
            mean_duration = 0
            mean_interval = 0
        
        return {
            "threshold_factor": threshold_factor,
            "threshold_method": threshold_method,
            "calculated threshold": threshold,
            "intermittency_factor": intermittency_factor,
            "event_count": len(rising_edges) if len(rising_edges) <= len(falling_edges) else len(falling_edges),
            "mean_event_duration": mean_duration,
            "mean_event_interval": mean_interval,
        }
        

    def analyze_concentration_fluctuations(self,dimensionless="False" ,intermittency_threshold=1.5,plot=True,errorConc=None,errorType=None, threshold_method='mean',threshold_type="ratio"):
        """
        Quick analysis of concentration fluctuations.
        
        Parameters:
        -----------
        time : array-like
            Time measurements
        concentration : array-like
            Concentration measurements
        plot : bool, optional
            Whether to generate plots (default True)

        Options:
        ---------
        Add Errorbars:
        If errorConc != None -> overgive errorvalue for concentration to visualise uncertainty as errorbars in plots
        If errorType !=None /"percentage"/"absolute" type of overgiving error
        
        Returns:
        --------
        dict: Fluctuation characteristics
        """
        time = self.time
        if(dimensionless=="True"):
             concentration = self.c_star
        else:
            concentration = self.net_concentration

        # Key metrics
        peakToMeanRatio = np.max(concentration) / np.mean(concentration)
        intermittency_factor = self.calculate_intermittency(threshold_method=threshold_method,threshold_factor=intermittency_threshold,threshold_type=threshold_type)["intermittency_factor"]
        turbulence_intensity_v = self.calculate_turbulence_intensity(dimensionless=dimensionless,returnDistribution="False",returnMetrics="False")
        
         # Sampling frequency
        fs = 1.0 / np.mean(np.diff(time))
        # Spectral analysis, using Welch's PSD
        freqs, psd = scSignal.welch(concentration, fs=fs)
        peak_freq = freqs[np.argmax(psd)]
        high_freq_power = np.sum(psd[freqs >= 1.0])
        total_power = np.sum(psd)
        fluctuation_intensity = high_freq_power / total_power

       
        if plot:
            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.plot(time, concentration.to_numpy())
            plt.grid(True)
            plt.title('Concentration Time Series')
            
            plt.subplot(122)
            plt.semilogy(freqs, psd)
            plt.title('Power Spectral Density')
            plt.tight_layout()
            plt.grid(True)
            plt.xlabel("frequency[Hz]")
            plt.ylabel("Power Spectral Density[(ppm)²/Hz]")
            plt.show()
        
        return {
            'wtref mean': np.mean(self.wtref),
            'wtref std' : np.std(self.wtref),
            'wtref turbulence intensity': turbulence_intensity_v, 
            'peakToMeanRatio' : peakToMeanRatio,
            'std_concentration': np.std(concentration),
            'intermittency factor for' + str(intermittency_threshold) : intermittency_factor,
            #'peak_frequency': peak_freq,
            #'fluctuation_intensity': fluctuation_intensity,
            
        }
    

    def downAverage(self, averageInterval,measurementFreq=0.005, columns=["net_concentration"]):
        """
        Down-average specified columns by grouping data into time intervals.
        
        Args:
            averageInterval: Time interval in seconds for averaging
            columns: List of column names to average
        """
        #Assuming 0.005s measurmenet frequency
        rows_per_interval = int(averageInterval / measurementFreq)
        
        for col in columns:
            data = getattr(self, col)  # Get the time series array
        
            data = data.to_numpy()

            # Reshape and average in chunks
            n_complete_intervals = len(data) // rows_per_interval
            reshaped = data[:n_complete_intervals * rows_per_interval].reshape(-1, rows_per_interval)
            averaged = np.mean(reshaped, axis=1)
            
            # Handle remaining samples if any
            if len(data) % rows_per_interval:
                remaining = np.mean(data[n_complete_intervals * rows_per_interval:])
                averaged = np.concatenate([averaged, [remaining]])
            
            setattr(self, col, pd.Series(averaged))  # Update the attribute

        return 

# %%
