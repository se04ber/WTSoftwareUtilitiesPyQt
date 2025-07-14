#! /usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import numpy.matlib
import math
import logging
import os
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import string

# Create logger
logger = logging.getLogger()
__all__ = ['EnsembleAnalysis']

# Add at the beginning of the file, after imports
SAVE_FOLDERS = [
    'windtunnel-master',
    'windtunnel-master-backup'
]

class EnsembleAnalysis(pd.DataFrame):
    """ EnsembleAnalysis is a class that holds data from the output of 
	PuffConcentration.get_results, to allow for a statistical ensemble
	analysis, similarly to the C program developed by Anne Philip. For more 
	details on the ensemble analysis method, see Bachelor Thesis of Anne 
	Philip (2010). 
	The EnsembleAnalysis class inherits from pandas.DataFrame, thus 
	offers all of the functionality offered by pandas (e.g. DataFrame.plot.hist(),
	DataFrame.to_excel(), or DataFrame.rolling().mean()) All the information in a
    PuffConcentration object can be saved to a txt file, as well as all
    file type offered by pandas.

    Parameters
    ----------

    data: np.array
    
    """

    def __init__(self,data):
        """ Initialise EnsembleAnalysis object. """
		
        super().__init__()

        self.data = data		
        self.ensemble_min = None
        self.ensemble_max = None	
        self.ensemble_mean = None
        self.ensemble_var = None
        self.ensemble_std = None
        self.class_width = None	
        self.n_classes = None	
        self.n_classes_raw = None			
        self.class_min = None
        self.class_max = None
        self.class_center = None
        self.class_freq = None
        self.class_freq_norm = None		
        #self.wtref = wtref
        #self.net_concentration = None		
        #self.open_rate = open_rate  # [%]
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
        self.calibration_curve = None
        self.calibration_factor = None	
        self.mass_flow_controller = None	
        self.ref_height = None
        self.ref_length = None	
        self.scaling_factor = None			
        #self.begin_release_period = None	
        #self.end_release_period = None		
        #self.begin_release_index = None	
        #self.end_release_index = None
        self.gas_factor = None
        self.gas_name = None
        self.mol_weight = None			
        self.temperature = None	
        self.standard_temp_K = None			
        self.pressure = None		
        #self.mask = None		
        #self.release_length = None
        #self.residence_time = None
        #self.arrival_time = None		
        #self.leaving_time = None
        #self.ascent_time = None
        #self.descent_time = None
        #self.peak_time = None
        #self.peak_concentration = None
        #self.dosage = None
        #self.puffs_array = None
        #self.signal_array = None				
        #self.mean_puff = None
        #self.pct90_puff = None	
        #self.pct10_puff = None			
        #self.mean_signal = None	
        #self.pct90_signal = None	
        #self.pct10_signal = None				
        #self.min_puff_length = None		
        #self.puff_deviations = None
        #self.threshold_concentration = None
        #self.number = None


    @classmethod
    def from_results(cls, data):
        """ Create object from output of PuffConcentration.
        
        Returns
        ----------
        data:
        
        
        
        """
			
		
		
        return cls(data)	
		
    def ambient_conditions(self, x_source, y_source, z_source, x_measure, y_measure, z_measure, pressure, temperature, calibration_curve,
                           mass_flow_controller, calibration_factor=0):
        """ 
        Collect ambient conditions during measurement. pressure in [Pa],
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
        
        self.gas_name = gas_name
        self.mol_weight = mol_weight
        self.gas_factor = gas_factor			
		
    def convert_temperature(self):
        """ Convert ambient temperature to °K. """
        
        self.temperature_K = self.temperature + 273.15
        self.standard_temp_K = self.standard_temp + 273.15		

    def get_ensembles(self,ensemble_size):
        """
        Determine composition of the individual ensembles, based on ensemble size. Output is an array of indices for each
        ensemble
        
        
        Returns
        ----------
        puff_numbers: array-like
        
        """
     
        
        if ensemble_size>np.size(self.data):
           print('Error: ensemble size greater than number of data points! Use smaller ensemble size and/or check the dataset. Also consider checking any thershold applying algorithms.')
           return 
        elif ensemble_size<2:	
           print('Error: ensemble size less than two, which make no sense. Check ensemble size!')
           return 		
         
        #calculate number of ensembles		
        n_ensembles=np.shape(self.data)[0]
		#create two consecutive arrays of all puff numbers, this is necessary because ensembles are effectively cyclic. See figure 2 in Philip (2010). 
        puff_indices=np.append(np.linspace(0,np.shape(self.data)[0]-1,np.shape(self.data)[0]),(np.linspace(0,np.shape(self.data)[0]-1,np.shape(self.data)[0])))
		#create arrray of puff numbers to be included in each ensemble. Convert to type integer to avoid problems when using this to index an array.
        puff_numbers=np.zeros((n_ensembles,ensemble_size)).astype(int)
        for i in range(n_ensembles):
            puff_numbers[i,:]=puff_indices[i:i+ensemble_size]

        return puff_numbers
	


    def get_ensemble_min(self):
        """
        Calculate minimum value of each individual ensemble. Output is array of values, with the row denoting the ensemble
		size, and the column the ensemble number
        
        """
        
        self.ensemble_min=np.zeros((np.size(self.data),np.shape(self.data)[0]))		
        for i in range(np.size(self.data)):
            if i<2:
               self.ensemble_min[i,:]=np.nan
            else: 
               puff_numbers=self.get_ensembles(ensemble_size=i)	
               self.ensemble_min[i,:]=np.matlib.repmat(self.data,np.shape(self.data)[0],1)[0,puff_numbers].min(axis=1)
			   
    def get_ensemble_max(self):
        """
        
        Calculate maximum value of each individual ensemble. Output is array of values, with the row denoting the ensemble
		size, and the column the ensemble number
        
        """
        
		
        self.ensemble_max=np.zeros((np.size(self.data),np.shape(self.data)[0]))		
        for i in range(np.size(self.data)):
            if i<2:
               self.ensemble_max[i,:]=np.nan
            else: 
               puff_numbers=self.get_ensembles(ensemble_size=i)	
               self.ensemble_max[i,:]=np.matlib.repmat(self.data,np.shape(self.data)[0],1)[0,puff_numbers].max(axis=1)

    def get_ensemble_mean(self):
        """
        Calculate mean value of each individual ensemble. Output is array of values, with the row denoting the ensemble size,
		and the column the ensemble number
        
        """
       
		
        self.ensemble_mean=np.zeros((np.size(self.data),np.shape(self.data)[0]))		
        for i in range(np.size(self.data)):
            if i<2:
               self.ensemble_mean[i,:]=np.nan
            else: 
               puff_numbers=self.get_ensembles(ensemble_size=i)	
               self.ensemble_mean[i,:]=np.matlib.repmat(self.data,np.shape(self.data)[0],1)[0,puff_numbers].mean(axis=1)	
			   
    def get_ensemble_variance(self):
        """
        Calculate mean value of each individual ensemble. Output is array of values, with the row denoting the ensemble size,
		and the column the ensemble number
        
        """
        
		
        self.ensemble_var=np.zeros((np.size(self.data),np.shape(self.data)[0]))		
        for i in range(np.size(self.data)):
            if i<2:
               self.ensemble_var[i,:]=np.nan
            else: 
               puff_numbers=self.get_ensembles(ensemble_size=i)	
               self.ensemble_var[i,:]=np.matlib.repmat(self.data,np.shape(self.data)[0],1)[0,puff_numbers].var(axis=1)	

    def get_ensemble_variance(self):
        """Calculate mean value of each individual ensemble. Output is array of values, with the row denoting the ensemble size,
		and the column the ensemble number"""
       
		
        self.ensemble_std=np.zeros((np.size(self.data),np.shape(self.data)[0]))		
        for i in range(np.size(self.data)):
            if i<2:
               self.ensemble_std[i,:]=np.nan
            else: 
               puff_numbers=self.get_ensembles(ensemble_size=i)	
               self.ensemble_std[i,:]=np.matlib.repmat(self.data,np.shape(self.data)[0],1)[0,puff_numbers].std(axis=1)		
			   
    def calc_n_classes(self,ensemble_size,n=None):
        """
        
        Calculate number of classes, based on ensemble size. 
        Method based on original C Program by Anne Philip.
        
        Parameters
        ----------
        
        
        ensemble_size: int
        n: int


        Returns
        ----------
        

        n_classes: int
        
        
        """
       
        if ensemble_size>np.size(self.data):
           print('Error: ensemble size greater than number of data points! Use smaller ensemble size and/or check the dataset. Also consider checking any thershold applying algorithms.')
           return 
        elif ensemble_size<2:	
           print('Error: ensemble size less than two, which make no sense. Check ensemble size!')
           return 	

        if n==None:
           n_classes_raw=np.int64(1+math.log10(ensemble_size)/math.log10(2))	
           n_classes=np.int64((n_classes_raw-1)*2)
        else:
           n_classes_raw=n
           n_classes=np.int64((n_classes_raw-1)*2)           		
        self.n_classes.astype(int)		


        return n_classes		
		
    def calc_class_width(self,n=None):
        """
        Compute class width for each ensemble size and number
        
        """
       
		
        self.class_width=np.zeros(np.shape(self.ensemble_min))
        self.n_classes=np.zeros(np.shape(self.ensemble_min))
        self.n_classes_raw=np.zeros(np.shape(self.ensemble_min))		
        for i in range(np.shape(self.ensemble_min)[0]):
            if i<2:
               self.class_width[i,:]=np.nan
               self.n_classes[i,:]=np.nan
               self.n_classes_raw[i,:]=np.nan			   
            else: 
               n_classes=self.calc_n_classes(ensemble_size=i,n=n)
               self.n_classes[i,:]=np.int64(n_classes)
               self.n_classes_raw[i,:]=(np.int64(n_classes)/2)+1			   
               for j in range(np.shape(self.ensemble_min)[1]):
                   #edit 08/13/2019: added multiplicative factor of 2 to class widhth, as in function mantioned above from Anne Philip's C program.  			   
                   self.class_width[i,j]=2*(self.ensemble_max[i,j]-self.ensemble_min[i,j])/self.n_classes[i,j]
       		   				   
    def calc_class_boundaries(self):
        """
        Calculates boundaries of the classes as a function of ensemble size and number. Method based on original C Program by Anne Philip. Output is a 3d array of values, 
		with the 1st dimension denoting the ensemble size, the second dimension denoting the ensemble number, and the 
		third dimension the class number
        
        """

		#Similar to subsection of CalculationAndWritingFrequencyDistribution function from original C program written by Anne Philip.
		#See Bachelor Thesis of Anne Philipp (2010) for more details. Requires calc_class_width to be called before calling function. 
				
        if self.class_width is None:
           print('Error: class widths not found. Make sure that calc_class_width is called before calling calc_class_boundaries')
           return	
        if self.n_classes is None:
           print('Error: number of classes not found. Make sure that calc_class_width is called before calling calc_class_boundaries')
           return	
        if self.n_classes_raw is None:
           print('Error: number of raw classes not found. Make sure that calc_class_width is called before calling calc_class_boundaries')
           return		   

        if (np.int64(np.nanmax(self.n_classes_raw)) - (np.nanmax(self.n_classes_raw))) != 0:	
            print('Error: number of classes must be an integer. Aborting Script.' )
            return         
        self.class_min=np.zeros((np.shape(self.ensemble_min)[0],np.shape(self.ensemble_min)[1],np.int64(np.nanmax(self.n_classes_raw))))
        self.class_max=np.zeros((np.shape(self.ensemble_min)[0],np.shape(self.ensemble_min)[1],np.int64(np.nanmax(self.n_classes_raw))))
        self.class_center=np.zeros((np.shape(self.ensemble_min)[0],np.shape(self.ensemble_min)[1],np.int64(np.nanmax(self.n_classes_raw))))
        self.class_min[:]=np.nan
        self.class_max[:]=np.nan
        self.class_center[:]=np.nan				
        for i in range(np.shape(self.class_min)[0]):
            if i<2:
               self.class_min[i,:,:]=np.nan
               self.class_max[i,:,:]=np.nan
               self.class_center[i,:,:]=np.nan			   
            else: 		   
               for j in range(np.shape(self.class_min)[1]):	
                   if (np.int64(self.n_classes_raw[i,j]) - (self.n_classes_raw[i,j])) != 0:				  
                       print('Error: number of classes must be an integer. Aborting Script.' )
                       return				  
                   for k in range(np.int64(self.n_classes_raw[i,j])):						   
                       self.class_min[i,j,k]=self.ensemble_min[i,j]+self.class_width[i,j]*(k-0.5)  
                       self.class_max[i,j,k]=self.ensemble_min[i,j]+self.class_width[i,j]*(k+0.5) 
                       self.class_center[i,j,k]=(self.class_min[i,j,k]+self.class_max[i,j,k])/2 					   

    def get_class_frequency(self):
        """
        Returns the number of data points inside the individual classes, based on the class boundaries calculated in calc_class_boundaries. Method based on original C Program 
		by Anne Philip. Output is a 3d array of values, with the 1st dimension denoting the ensemble size, the second dimension denoting the ensemble number, and the 
		third dimension the class number. As in Anne Philips program, the classes have closed intervals at the lower boundary, and open interval at the upper boundary, i.e. points on the class
		boundary are assigned to the interval above the boundary.
        
        """
		

        if self.class_width is None:
           print('Error: class widths not found. Make sure that calc_class_width is called before calling get_class_frequency')
           return	
        if self.n_classes is None:
           print('Error: number of classes not found. Make sure that calc_class_width is called before calling get_class_frequency')
           return
        if self.n_classes_raw is None:
           print('Error: number of raw classes not found. Make sure that calc_class_width is called before calling get_class_frequency')
           return			   
        if self.class_min is None:
           print('Error: lower class boundaries not found. Make sure that calc_class_boundaries is called before calling get_class_frequency')
           return	
        if self.class_max is None:
           print('Error: upper class boundaries not found. Make sure that calc_class_boundaries is called before calling get_class_frequency')
           return	
        if  self.class_center is None:
           print('Error: center of classes not found. Make sure that calc_class_boundaries is called before calling get_class_frequency')
           return			   

        if (np.int64(np.nanmax(self.n_classes_raw)) - (np.nanmax(self.n_classes_raw))) != 0:	
            print('Error: number of classes must be an integer. Aborting Script.' )
            return               	
        self.class_freq=np.zeros((np.shape(self.ensemble_min)[0],np.shape(self.ensemble_min)[1],np.int64(np.nanmax(self.n_classes_raw))))
        self.class_freq_norm=np.zeros((np.shape(self.ensemble_min)[0],np.shape(self.ensemble_min)[1],np.int64(np.nanmax(self.n_classes_raw))))			
        self.class_freq[:]=np.nan
        self.class_freq_norm[:]=np.nan		
        for i in range(np.shape(self.class_min)[0]):
            if i<2:
               self.class_freq[i,:,:]=np.nan
               self.class_freq_norm[i,:,:]=np.nan			   
            else: 	
               ensemble_puffs=np.matlib.repmat(self.data,np.shape(self.data)[0],1)[0,self.get_ensembles(ensemble_size=i)]	
               #for j in range(np.shape(self.class_min)[1]):				
                   #ensemble_puffs=self.data[self.get_ensembles(ensemble_size=i)[j,:]]			   
               if (np.int64(self.n_classes_raw[i,0]) - (self.n_classes_raw[i,0])) != 0:				  
                   print('Error: number of classes must be an integer. Aborting Script.' )
                   return						
               for k in range(np.int64(self.n_classes_raw[i,0])):
                   self.class_freq[i,:,k]=((self.class_min[i,:,k]<=ensemble_puffs.transpose()) & (ensemble_puffs.transpose() < self.class_max[i,:,k])).transpose().sum(axis=1)
                   self.class_freq_norm[i,:,k]=((self.class_min[i,:,k]<=ensemble_puffs.transpose()) & (ensemble_puffs.transpose() < self.class_max[i,:,k])).transpose().sum(axis=1)/i			   

    def calc_puff_statistics(self, x_source, y_source, z_source, x_measure, y_measure, z_measure, pressure,temperature,wtref,wdir):
        """ Performs basic statistical analysis (mean and standard deviation) of puffs as well as puff info. Similar to 
        CalculatePuffStatistics in the original c program written by Anne Philip.
        
        Parameters
        ----------
        


        x_source: float
        y_source: float 
        z_source: float 
        x_measure: float 
        y_measure: float 
        z_measure: float 
        pressure: float
        wtref: np.array
        wdir: float
        
        """
	   
        
        self.stat_mean = None	
        self.stat_std = None
        self.x_source = None
        self.y_source = None
        self.z_source = None
        self.x_measure = None
        self.y_measure = None
        self.z_measure = None
        self.x = None
        self.y = None
        self.z = None
        self.pressure = None
        self.temperature = None
        self.wtref = None
        self.wdir = None	

        self.stat_mean=self.data.mean()		
        self.stat_std=self.data.std()
		
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
        self.pressure = pressure
        self.temperature = temperature
        self.wtref = wtref
        self.wdir = wdir	
            
    def plot_convergence_ensemble(self,key=None,path=None,name=None,conv_step=1,full_scale=None):
        """Plot convergence analysis of puff data based on the calculated ensemble means
        

        Parameters
        ----------
        

        key: str
        path: str
        name: str
        conv_step: int
        full_scale: str


        Return
        ----------
        
        
        ret: plt.object
        
        """
       
        #TODO: add units, requires adding a unit attribute in the data dictionaries
 
        if self.ensemble_mean is None:
           print('Error: ensemble means not found. Make sure that get_ensemble_mean is called before calling plot_convergence_ensemble')
           return		   

        ensemble_size_array = np.linspace(0,np.shape(self.data)[0]-1,np.shape(self.data)[0])	
        plt.ioff()	
        ret=plt.figure(200)
        plt.clf()		
        for j in range(0,np.shape(self.ensemble_mean)[0],conv_step):
            plt.scatter(ensemble_size_array,self.ensemble_mean[:,j],color='b',s=5,marker='.')
        ax=plt.gca()            
        if full_scale=='ms':
           ax.set_title('Convergence Analysis (Model Scale) of ' + string.capwords(key),fontsize=40)        
           #plt.title('Convergence Analysis (Model Scale) of ' + string.capwords(key) )
        elif full_scale=='fs':
           ax.set_title('Convergence Analysis (Full Scale) of ' + string.capwords(key),fontsize=40)        
           #plt.title('Convergence Analysis (Full Scale) of ' + string.capwords(key) )           
        elif full_scale=='nd':
           ax.set_title('Convergence Analysis (Non-Dimensional) of ' + string.capwords(key),fontsize=40)        
           #plt.title('Convergence Analysis (Full Scale) of ' + string.capwords(key) )           
        else:
           print("Error: invalid input for full_scale. Data can only be computed in model scale (full_scale='ms'), full scale (full_scale='fs'), or non-dimensionally (full_scale='nd')")  

        ret.set_figwidth(26)	
        ret.set_figheight(16) 
        n=np.shape(self.ensemble_mean)[0]
        plt.xticks(np.arange(1,n,50),np.arange(1,n,50,dtype=np.int64))	
        plt.tick_params(axis='both', labelsize=30)   
        plt.xlim(1,n)	        
        plt.xlabel('Ensemble Size',fontsize=40)
        plt.ylabel('Ensemble Mean',fontsize=40)	

        if not os.path.exists(path+'Puff_Plots/'+name[:-9]+'/'):
           os.makedirs(path+'Puff_Plots/'+name[:-9]+'/') 			
        if path=='none':
           []
        else:	
           if name=='none':
              print('Name of dataset not specified. Plot of mean puff will not be saved to avoid confusion in the future.')	
           else:	
              if full_scale=='ms':           
                 plt.savefig(path + 'Puff_Plots/' + name[:-9] + '/Convergence Analysis of ' + string.capwords(key) + ', Model Scale.png')
              elif full_scale=='fs':
                 plt.savefig(path + 'Puff_Plots/' + name[:-9] + '/Convergence Analysis of ' + string.capwords(key) + ', Full Scale.png') 
              elif full_scale=='nd':
                 plt.savefig(path + 'Puff_Plots/' + name[:-9] + '/Convergence Analysis of ' + string.capwords(key) + ', Non-Dimensional.png')          
              else:
                 print("Error: invalid input for full_scale. Data can only be computed in model scale (full_scale='ms'), full scale (full_scale='fs'), or non-dimensionally (full_scale='nd')")            
              #plt.savefig(path + 'Puff_Plots/' + name[:-9] + '/Convergence Analysis of ' + string.capwords(key) + '.png')
        plt.show()
        plt.close()	

        return ret		

    def plot_class_statistics(self,key=None,path=None,name=None,full_scale=None): 		
	
        """Plot histogram of class frequencies, one for each ensemble size
        
        Parameters
        ----------
        

        key: str
        path: str
        name: str
        full_scale: str


        Return
        ----------
        
        
        ret: plt.object
        
        
        """
      
        #TODO: add units, requires adding a unit attribute in the data dictionaries

        if self.class_freq_norm is None:
           print('Error: Normalized class frequencies not found. Make sure that get_class_frequency is called before calling plot_class_statitics')
           return		   

        ret=plt.figure(201)		
        for i in range(np.shape(self.class_freq_norm)[0]):		   
            if i<2:
               continue
            else:
                plt.ioff()		
                plt.clf()				
                #plt.bar(np.linspace(1,np.shape(self.class_freq_norm[i,:,:])[1],np.nanmean(self.class_freq_norm[i,:,:],0),width=1,align='center')	
                #plt.bar(np.linspace(1,np.shape(self.class_freq_norm[i,:,:])[1],np.nanmean(self.class_freq_norm[i,:,:],0)))
                plt.bar(np.linspace(1,np.shape(self.class_freq_norm[i,:,:])[1],np.shape(self.class_freq_norm[i,:,:])[1]),np.nanmean(self.class_freq_norm[i,:,:],0),width=1,align='center')	
                ax=plt.gca()
                if full_scale=='ms':
                   ax.set_title('Frequency distribution (Model Scale) of '+string.capwords(key)+' for ensembles with size '+str(i),fontsize=40)
                elif full_scale=='fs':
                   ax.set_title('Frequency distribution (Full Scale) of '+string.capwords(key)+' for ensembles with size '+str(i),fontsize=40)
                elif full_scale=='nd':
                   ax.set_title('Frequency distribution (Non-Dimensional) of '+string.capwords(key)+' for ensembles with size '+str(i),fontsize=40)                
                else:
                   print("Error: invalid input for full_scale. Data can only be computed in model scale (full_scale='ms'), full scale (full_scale='fs'), or non-dimensionally (full_scale='nd')")                 			
                				
                ret.set_figwidth(26)	
                ret.set_figheight(16)								
                n=np.int64(self.n_classes_raw[i,0])		
                plt.xticks(np.linspace(1,n,n),np.linspace(1,n,n,dtype=np.int64))	
                plt.tick_params(axis='both', labelsize=30)

				
                plt.xlim(0.5,n+0.5)	
                #plt.ylim(0,1)
                plt.xlabel('Class',fontsize=40)					
                plt.ylabel('Frequency',fontsize=40)			

                

                if not os.path.exists(path+'Puff_Plots/'+name[:-9]+'/Frequency_Distributions/'):
                   os.makedirs(path+'Puff_Plots/'+name[:-9]+'/Frequency_Distributions/') 			
                if path=='none':
                   []
                else:	
                   if name=='none':
                      print('Name of dataset not specified. Frequency distribution plots will not be saved to avoid confusion in the future.')	
                   else:		
                      if full_scale=='ms':           
                         plt.savefig(path + 'Puff_Plots/' + name[:-9] + '/Frequency_Distributions/Frequency_Distribution_Model_Scale_'+string.capwords(key)+'_ensemble_size_' + str(i)+'.png')
                      elif full_scale=='fs':
                         plt.savefig(path + 'Puff_Plots/' + name[:-9] + '/Frequency_Distributions/Frequency_Distribution_Full_Scale_'+string.capwords(key)+'_ensemble_size_' + str(i)+'.png')
                      elif full_scale=='nd':
                         plt.savefig(path + 'Puff_Plots/' + name[:-9] + '/Frequency_Distributions/Frequency_Distribution_Non_Dimensional_'+string.capwords(key)+'_ensemble_size_' + str(i)+'.png')                         
                      else:
                         print("Error: invalid input for full_scale. Data can only be computed in model scale (full_scale='ms'), full scale (full_scale='fs'), or non-dimensionally (full_scale='nd')")                     
                      #plt.savefig(path + 'Puff_Plots/' + name[:-9] + '/Frequency_Distributions/Frequency_Distribution_'+string.capwords(key)+'_ensemble_size_' + str(i)+'.png')	

        plt.show()
        return ret	

    def save2file_ms_ensemble(self, filename, key=None, out_dir=None):
        """ Save model scale data from PuffConcentration object to txt file.
        filename must include '.txt' ending. If no out_dir directory is
        provided './' is set as standard.
        
        Parameters
        ----------
        
        
        filename: str
        out_dir:str
        
        """
		
        
        if key==None:
           print('Error: Unspecified key. Unable to save convergence data and class data to file. Please specify key in input of save2file_ms_ensemble!')		
        if out_dir is None:
            out_dir = './'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        output_file_conv = out_dir + 'conv_ms_' + key.replace(" ","_") + '_' + filename	
        output_file_class = out_dir + 'class_ms_' + key.replace(" ","_") + filename
        class_header="variables = \" ensemble size        \""
        for i in range(np.shape(self.ensemble_mean)[0]):
            if i+1<10:		
               class_header=class_header+ " \"  Ensemble number "+str(i+1)+"  \""	
            elif i+1>=10 and i+1<100:		
               class_header=class_header+ " \"  Ensemble number "+str(i+1)+" \""	
            elif i+1>=100 and i+1<1000:		
               class_header=class_header+ " \" Ensemble number "+str(i+1)+" \""
            elif i+1>=1000 and i+1<10000:		
               class_header=class_header+ " \" Ensemble number "+str(i+1)+"\""
            elif i+1>=10000 and i+1<100000:		
               class_header=class_header+ " \"Ensemble number "+str(i+1)+"\""				   
            else:
               print('Warning: attempting to write ' + str(np.shape(self.ensemble_mean)[0]) + ' ensembles to file. Program currently does not support writing more than 100000 ensembles to file.')			
        n_ensembles=np.linspace(0,np.shape(self.ensemble_mean)[0]-1,np.shape(self.ensemble_mean)[0],dtype=np.int64)	
        ensemble_size_label=np.broadcast_to(np.linspace(0,np.shape(self.class_min)[0]-1,np.shape(self.class_min)[0]).transpose()[:,np.newaxis,np.newaxis],np.shape(self.class_min))
        ensemble_number_label=np.broadcast_to(np.linspace(1,np.shape(self.class_min)[1],np.shape(self.class_min)[1])[np.newaxis,:,np.newaxis],np.shape(self.class_min))	
        class_number_label=np.broadcast_to(np.linspace(1,np.shape(self.class_min)[2],np.shape(self.class_min)[2])[np.newaxis,np.newaxis,:],np.shape(self.class_min))
        #remove nan values, since tecplot does not accept non-numerical values in input dataset. 		
        tecplot_class_min=np.reshape(self.class_min,np.size(self.class_min))
        tecplot_mask=~np.isnan(tecplot_class_min)	
        np.savetxt(output_file_class, np.vstack((np.reshape(ensemble_size_label,np.size(self.class_min))[tecplot_mask],
                                          np.reshape(ensemble_number_label,np.size(self.class_min))[tecplot_mask],
                                          np.reshape(class_number_label,np.size(self.class_min))[tecplot_mask],										  
                                          np.reshape(self.class_min,np.size(self.class_min))[tecplot_mask],
                                          np.reshape(self.class_max,np.size(self.class_max))[tecplot_mask],
                                          np.reshape(self.class_max-self.class_min,np.size(self.class_max))[tecplot_mask])).transpose(),								  
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
                          "reference length (model): {:.4f} [m], "
                          "Tracer gas: {}, mol. weight tracer: {:.4f} [mol/kg], "
                          "gas factor: {:.6f}".format(self.x, self.y, self.z,
                                                       self.x_source, self.y_source, self.z_source,
                                                       self.x_measure, self.y_measure, self.z_measure,  
                                                       self.distance,                                                     
                                                       self.temperature,
                                                       self.pressure,
                                                       self.ref_length,
                                                       self.gas_name,
                                                       self.mol_weight,
                                                       self.gas_factor)
                          + "" + '\n' +
						  "variables = \" ensemble size \" \"   ensemble number  \" \"   class number     \" \"      class min     \" \"      class max     \" \"     class width    \" ",comments='')	
	
		
        np.savetxt(output_file_conv, np.vstack((n_ensembles[2:],self.ensemble_mean[2:,:].transpose())).transpose(),		
                   fmt='%23.4f',
                   header=("General concentration measurement data:" + '\n' +
                          "" + '\n' +
                          "geometric scale: 1:{}".format(float(self.scale))
                          + "" + '\n' +
                          "Variables: x: {} [mm], y: {} [mm], z: {} [mm], "
                          "ambient temperature: {:.1f} [°C], ambient pressure: {:.2f} [Pa],"
                          "reference length (model): {:.4f} [m], "
                          "Tracer gas: {}, mol. weight tracer: {:.4f} [mol/kg], "
                          "gas factor: {:.6f}".format(self.x, self.y, self.z,
                                                       self.temperature,
                                                       self.pressure,
                                                       self.ref_length,
                                                       self.gas_name,
                                                       self.mol_weight,
                                                       self.gas_factor)
                          + "" + '\n' +
						 class_header).format('%.4f'),comments='')	

    def save2file_fs_ensemble(self, filename, key=None, out_dir=None):
        """ Save full scale data from PuffConcentration object to txt file.
        filename must include '.txt' ending. If no out_dir directory is
        provided './' is set as standard.
        
        Parameters
        ----------
        
        
        filename: str
        out_dir: str
        
        """
		
        
        if key==None:
           print('Error: Unspecified key. Unable to save convergence data and class data to file. Please specify key in input of save2file_ms_ensemble!')		
        if out_dir is None:
            out_dir = './'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        output_file_conv = out_dir + 'conv_fs_' + key.replace(" ","_") + '_' + filename	
        output_file_class = out_dir + 'class_fs_' + key.replace(" ","_") + filename
        class_header="variables = \" ensemble size        \""
        for i in range(np.shape(self.ensemble_mean)[0]):
            if i+1<10:		
               class_header=class_header+ " \"  Ensemble number "+str(i+1)+"  \""	
            elif i+1>=10 and i+1<100:		
               class_header=class_header+ " \"  Ensemble number "+str(i+1)+" \""	
            elif i+1>=100 and i+1<1000:		
               class_header=class_header+ " \" Ensemble number "+str(i+1)+" \""
            elif i+1>=1000 and i+1<10000:		
               class_header=class_header+ " \" Ensemble number "+str(i+1)+"\""
            elif i+1>=10000 and i+1<100000:		
               class_header=class_header+ " \"Ensemble number "+str(i+1)+"\""				   
            else:
               print('Warning: attempting to write ' + str(np.shape(self.ensemble_mean)[0]) + ' ensembles to file. Program currently does not support writing more than 100000 ensembles to file.')			
        n_ensembles=np.linspace(0,np.shape(self.ensemble_mean)[0]-1,np.shape(self.ensemble_mean)[0],dtype=np.int64)	
        ensemble_size_label=np.broadcast_to(np.linspace(0,np.shape(self.class_min)[0]-1,np.shape(self.class_min)[0]).transpose()[:,np.newaxis,np.newaxis],np.shape(self.class_min))
        ensemble_number_label=np.broadcast_to(np.linspace(1,np.shape(self.class_min)[1],np.shape(self.class_min)[1])[np.newaxis,:,np.newaxis],np.shape(self.class_min))	
        class_number_label=np.broadcast_to(np.linspace(1,np.shape(self.class_min)[2],np.shape(self.class_min)[2])[np.newaxis,np.newaxis,:],np.shape(self.class_min))
        #remove nan values, since tecplot does not accept non-numerical values in input dataset. 		
        tecplot_class_min=np.reshape(self.class_min,np.size(self.class_min))
        tecplot_mask=~np.isnan(tecplot_class_min)	
        np.savetxt(output_file_class, np.vstack((np.reshape(ensemble_size_label,np.size(self.class_min))[tecplot_mask],
                                          np.reshape(ensemble_number_label,np.size(self.class_min))[tecplot_mask],
                                          np.reshape(class_number_label,np.size(self.class_min))[tecplot_mask],										  
                                          np.reshape(self.class_min,np.size(self.class_min))[tecplot_mask],
                                          np.reshape(self.class_max,np.size(self.class_max))[tecplot_mask],
                                          np.reshape(self.class_max-self.class_min,np.size(self.class_max))[tecplot_mask])).transpose(),								  
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
                          "reference length (model): {:.4f} [m], "
                          "Tracer gas: {}, mol. weight tracer: {:.4f} [mol/kg], "
                          "gas factor: {:.6f}".format(self.x, self.y, self.z,
                                                       self.x_source, self.y_source, self.z_source,
                                                       self.x_measure, self.y_measure, self.z_measure,  
                                                       self.distance,                                                     
                                                       self.temperature,
                                                       self.pressure,
                                                       self.ref_length,
                                                       self.gas_name,
                                                       self.mol_weight,
                                                       self.gas_factor)
                          + "" + '\n' +
						  "variables = \" ensemble size \" \"   ensemble number  \" \"   class number     \" \"      class min     \" \"      class max     \" \"     class width    \" ",comments='')	
	
		
        np.savetxt(output_file_conv, np.vstack((n_ensembles[2:],self.ensemble_mean[2:,:].transpose())).transpose(),		
                   fmt='%23.4f',
                   header=("General concentration measurement data:" + '\n' +
                          "" + '\n' +
                          "geometric scale: 1:{}".format(float(self.scale))
                          + "" + '\n' +
                          "Variables: x: {} [mm], y: {} [mm], z: {} [mm], "
                          "ambient temperature: {:.1f} [°C], ambient pressure: {:.2f} [Pa],"
                          "reference length (model): {:.4f} [m], "
                          "Tracer gas: {}, mol. weight tracer: {:.4f} [mol/kg], "
                          "gas factor: {:.6f}".format(self.x, self.y, self.z,
                                                       self.temperature,
                                                       self.pressure,
                                                       self.ref_length,
                                                       self.gas_name,
                                                       self.mol_weight,
                                                       self.gas_factor)
                          + "" + '\n' +
						 class_header).format('%.4f'),comments='')						                           
        
    def save2file_nd_ensemble(self, filename, key=None, out_dir=None):
        """ Save full scale data from PuffConcentration object to txt file.
        filename must include '.txt' ending. If no out_dir directory is
        provided './' is set as standard.
        
        Parameters
        ----------
        
        
        filename: str
        out_dir:str
        
        
        """
		
        
        if key==None:
           print('Error: Unspecified key. Unable to save convergence data and class data to file. Please specify key in input of save2file_ms_ensemble!')		
        if out_dir is None:
            out_dir = './'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        output_file_conv = out_dir + 'conv_nd_' + key.replace(" ","_") + '_' + filename	
        output_file_class = out_dir + 'class_nd_' + key.replace(" ","_") + filename
        class_header="variables = \" ensemble size        \""
        for i in range(np.shape(self.ensemble_mean)[0]):
            if i+1<10:		
               class_header=class_header+ " \"  Ensemble number "+str(i+1)+"  \""	
            elif i+1>=10 and i+1<100:		
               class_header=class_header+ " \"  Ensemble number "+str(i+1)+" \""	
            elif i+1>=100 and i+1<1000:		
               class_header=class_header+ " \" Ensemble number "+str(i+1)+" \""
            elif i+1>=1000 and i+1<10000:		
               class_header=class_header+ " \" Ensemble number "+str(i+1)+"\""
            elif i+1>=10000 and i+1<100000:		
               class_header=class_header+ " \"Ensemble number "+str(i+1)+"\""				   
            else:
               print('Warning: attempting to write ' + str(np.shape(self.ensemble_mean)[0]) + ' ensembles to file. Program currently does not support writing more than 100000 ensembles to file.')			
        n_ensembles=np.linspace(0,np.shape(self.ensemble_mean)[0]-1,np.shape(self.ensemble_mean)[0],dtype=np.int64)	
        ensemble_size_label=np.broadcast_to(np.linspace(0,np.shape(self.class_min)[0]-1,np.shape(self.class_min)[0]).transpose()[:,np.newaxis,np.newaxis],np.shape(self.class_min))
        ensemble_number_label=np.broadcast_to(np.linspace(1,np.shape(self.class_min)[1],np.shape(self.class_min)[1])[np.newaxis,:,np.newaxis],np.shape(self.class_min))	
        class_number_label=np.broadcast_to(np.linspace(1,np.shape(self.class_min)[2],np.shape(self.class_min)[2])[np.newaxis,np.newaxis,:],np.shape(self.class_min))
        #remove nan values, since tecplot does not accept non-numerical values in input dataset. 		
        tecplot_class_min=np.reshape(self.class_min,np.size(self.class_min))
        tecplot_mask=~np.isnan(tecplot_class_min)	
        np.savetxt(output_file_class, np.vstack((np.reshape(ensemble_size_label,np.size(self.class_min))[tecplot_mask],
                                          np.reshape(ensemble_number_label,np.size(self.class_min))[tecplot_mask],
                                          np.reshape(class_number_label,np.size(self.class_min))[tecplot_mask],										  
                                          np.reshape(self.class_min,np.size(self.class_min))[tecplot_mask],
                                          np.reshape(self.class_max,np.size(self.class_max))[tecplot_mask],
                                          np.reshape(self.class_max-self.class_min,np.size(self.class_max))[tecplot_mask])).transpose(),								  
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
                          "reference length (model): {:.4f} [m], "
                          "Tracer gas: {}, mol. weight tracer: {:.4f} [mol/kg], "
                          "gas factor: {:.6f}".format(self.x, self.y, self.z,
                                                       self.x_source, self.y_source, self.z_source,
                                                       self.x_measure, self.y_measure, self.z_measure,  
                                                       self.distance,                                                     
                                                       self.temperature,
                                                       self.pressure,
                                                       self.ref_length,
                                                       self.gas_name,
                                                       self.mol_weight,
                                                       self.gas_factor)
                          + "" + '\n' +
						  "variables = \" ensemble size \" \"   ensemble number  \" \"   class number     \" \"      class min     \" \"      class max     \" \"     class width    \" ",comments='')	
		
        np.savetxt(output_file_conv, np.vstack((n_ensembles[2:],self.ensemble_mean[2:,:].transpose())).transpose(),		
                   fmt='%23.4f',
                   header=("General concentration measurement data:" + '\n' +
                          "" + '\n' +
                          "geometric scale: 1:{}".format(float(self.scale))
                          + "" + '\n' +
                          "Variables: x: {} [mm], y: {} [mm], z: {} [mm], "
                          "ambient temperature: {:.1f} [°C], ambient pressure: {:.2f} [Pa],"
                          "reference length (model): {:.4f} [m], "
                          "Tracer gas: {}, mol. weight tracer: {:.4f} [mol/kg], "
                          "gas factor: {:.6f}".format(self.x, self.y, self.z,
                                                       self.temperature,
                                                       self.pressure,
                                                       self.ref_length,
                                                       self.gas_name,
                                                       self.mol_weight,
                                                       self.gas_factor)
                          + "" + '\n' +
						 class_header).format('%.4f'),comments='')        			

    def get_ensemble_statistics(self):
        """Calculate ensemble statistics including mean, variance, and standard deviation
        
        The statistics are calculated for increasing ensemble sizes to study convergence.
        For each ensemble size i, multiple ensembles are created by sampling i measurements.
        
        Returns:
            None - stores results in:
                self.ensemble_mean: Mean for each ensemble size and sample
                self.ensemble_var: Variance for each ensemble size and sample  
                self.ensemble_std: Standard deviation for each ensemble size and sample
                self.ensemble_cv: Coefficient of variation for each ensemble size
        """
        n_samples = np.size(self.data)
        n_ensembles = np.shape(self.data)[0]
        
        # Initialize arrays
        self.ensemble_mean = np.zeros((n_samples, n_ensembles))
        self.ensemble_var = np.zeros((n_samples, n_ensembles))
        self.ensemble_std = np.zeros((n_samples, n_ensembles))
        self.ensemble_cv = np.zeros(n_samples)  # Coefficient of variation
        
        # Calculate statistics for each ensemble size
        for i in range(n_samples):
            if i < 2:
                # Need at least 2 samples for variance/std
                self.ensemble_mean[i,:] = np.nan
                self.ensemble_var[i,:] = np.nan
                self.ensemble_std[i,:] = np.nan
                self.ensemble_cv[i] = np.nan
            else:
                # Get random ensembles of size i
                puff_numbers = self.get_ensembles(ensemble_size=i)
                
                # Replicate data array for vectorized calculations
                data_matrix = np.matlib.repmat(self.data, n_ensembles, 1)[0,puff_numbers]
                
                # Calculate statistics
                self.ensemble_mean[i,:] = np.mean(data_matrix, axis=1)
                self.ensemble_var[i,:] = np.var(data_matrix, axis=1)
                self.ensemble_std[i,:] = np.std(data_matrix, axis=1)
                
                # Calculate coefficient of variation across ensembles
                self.ensemble_cv[i] = np.std(self.ensemble_mean[i,:]) / np.mean(self.ensemble_mean[i,:])

    def plot_ensemble_convergence(self, key=None, path=None, name=None):
        """Plot convergence analysis of ensemble statistics
        
        Args:
            key: Variable name for plot title
            path: Path to save plot
            name: Name for saved plot file
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot mean convergence
        ensemble_sizes = np.arange(1, len(self.ensemble_mean)+1)
        mean_of_means = np.nanmean(self.ensemble_mean, axis=1)
        std_of_means = np.nanstd(self.ensemble_mean, axis=1)
        
        ax1.plot(ensemble_sizes, mean_of_means, 'b-', label='Mean')
        ax1.fill_between(ensemble_sizes, 
                         mean_of_means - std_of_means,
                         mean_of_means + std_of_means,
                         alpha=0.2)
        ax1.set_xlabel('Ensemble Size')
        ax1.set_ylabel('Ensemble Mean')
        ax1.set_title(f'Convergence of Mean for {key}')
        
        # Plot coefficient of variation
        ax2.plot(ensemble_sizes, self.ensemble_cv, 'r-', label='CV')
        ax2.set_xlabel('Ensemble Size')
        ax2.set_ylabel('Coefficient of Variation')
        ax2.set_title('Convergence of Ensemble Variability')
        
        plt.tight_layout()
        
        # Save plot if path provided
        if path and name:
            plt.savefig(os.path.join(path, f'ensemble_convergence_{key}.png'))
        
        return fig

    def get_convergence_metrics(self, threshold=0.05):
        """Calculate convergence metrics for ensemble statistics
        
        Args:
            threshold: Relative change threshold for convergence
            
        Returns:
            dict: Convergence metrics including:
                - Minimum ensemble size for mean convergence
                - Minimum ensemble size for variance convergence
                - Final coefficient of variation
        """
        # Calculate relative changes
        mean_changes = np.abs(np.diff(np.nanmean(self.ensemble_mean, axis=1))) / \
                      np.nanmean(self.ensemble_mean[1:], axis=1)
        var_changes = np.abs(np.diff(np.nanmean(self.ensemble_var, axis=1))) / \
                     np.nanmean(self.ensemble_var[1:], axis=1)
                     
        # Find where changes become small enough
        mean_converged = np.where(mean_changes < threshold)[0]
        var_converged = np.where(var_changes < threshold)[0]
        
        return {
            'mean_convergence_size': mean_converged[0] if len(mean_converged) > 0 else None,
            'variance_convergence_size': var_converged[0] if len(var_converged) > 0 else None,
            'final_cv': self.ensemble_cv[-1]
        }

# Replace any existing save path references to use the new SAVE_FOLDERS
def save_results(data, filename):
    for folder in SAVE_FOLDERS:
        save_path = os.path.join(folder, 'concentration', filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # ... existing save logic ...
