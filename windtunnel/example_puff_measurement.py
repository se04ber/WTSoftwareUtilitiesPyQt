# -*- coding: utf-8 -*-
import windtunnel as wt
import time
import matplotlib.pyplot as plt
import pandas as pd
# This is an example script for the use of a PuffConcentration object.
# The functionality of the PuffConcentration class is shown, as well
# as some of the functionality inherited from pandas.Dataframe.
# The PuffConcentration class returns a DataFrame, using the standard
# measurement txt-file output as input. The columns in the input file
# are expected to be time, wtref, slow FID, fast ID, release signal and
# open_rate.

start = time.time()
# Path to your data (current path leads to example data)
#path = '\\\\ewtl2\\work\\Johannes\Puff_Beispiele\\'
path = '/home/sabrina/Desktop/Schreibtisch/Arbeit_2025/windtunnel_software/Data/'

#edit 02/18/2020: new variable to specify name of csv file which contains ambient conditions data. If given dataset
#is not found in the given file, the program resosrts to the default values specified below. 
#csv_file='Q2_Ambient_Conditions.csv'
csv_file='ambient_conditions.csv'

# Name of your measurement
#namelist = ['Q2_170_P09.txt.ts#0','Q2_170_P10.txt.ts#0']
#namelist = ['BFS_BD3_MP01_000_01.ts#0','BFS_BD3_MP01_000_02.ts#0']    
namelist = ['BFS_BD3_MP01_000_01.ts#0']#,'BFS_BD3_MP01_000_02.ts#0']           
#Define user input variables
#Set theshold peak concentration (ppm, model scale). All puffs with a peak concentration
#less than this concentration will be ignored in the analysis.  
threshold_concentration=0#
#Set theshold dosage (ppmvs, model scale). All puffs with a total dose less
#than this concentration will be ignored in the analysis.  
threshold_dosage=0
#edit 03/18/2020: added variable 'n_exclude,' which specified how many outliers 
#to remove at the top of the measurements. Setting 'n_exclude' to None (without qutation marks) will 
#automatically select number of outliers to remove based on sample size. To turn
#off the removal of outliers, set 'n_exclude' to zero.  
n_exclude=None
#edit 02/04/2020: added variable 'time_threshold' to control dosage threshold to use for computing characteristic
#puff times. Note that a 'default' agreed-upon value for this variable 5%, however,this fails to properly
#capture the start and end times of several puffs.	
#edit 09/19/2019: added variable full_scale to determine whether to perform analysis at full scale or model scale. 
#Set full_scale to 'fs' to perform analysis at full scale, 'ms' for model scale, or 'both' for both full scale


#Debug
path = '/home/sabrina/Desktop/Schreibtisch/Arbeit_2025/windtunnel_software/Data/PuffData/'

#Specify name of csv file which contains ambient conditions data. 
#If given dataset is not found in the given file, the program resosrts to the default values specified below. 
csv_file='ambient_conditions.csv'

# Name of your measurement
#namelist = ['Q2_170_P09.txt.ts#0','Q2_170_P10.txt.ts#0']   
#namelist = ['BFS_BD3_MP01_000_01.ts#0']#,'BFS_BD3_MP01_000_02.ts#0']  
namelist = ['MS_S2P2_PM_194.dat.ts#2'] #PuffData example



#and model scale. 
time_threshold=0.05
if time_threshold != 0.05:
    print('Warning: threshold dosage used to compute characteristic start and end times set to '+str(100*time_threshold)+'%, which does not equal the default value of 5%. Consider using default value!')
#edit 02/04/2020: added non-dimensional mode
full_scale='ms'
#edit 09/19/2019: added a priori information necessary for full scale analysis. Potential for GUI usage
#at a futuretime
#edit 02/21/2020: added seperate input of source location (x_source, y_source, z_source) and measurement location (x_measure, y_measure, z_measure)
#edit 05/20/2020: the proposed GUI is well under development, but has been moved to a separate
#script, titled "PAPE_GUI_code_point.py."
#edit 07/23/2020: GUI has been restructured. Standalone script "PAPE_GUI_code_point.py" has 
#been abandoned in favor of the function "standard_point_analysis.py" in the windtunnel
#package. This avoids having to use the insecure method with the open and exec functions
#in the GUI script. 
x_source=0
y_source=0
z_source=0
x_measure=855.16
y_measure=176.29
z_measure=162
pressure=1009.38
temperature=23.5
#full_scale_wtref=0
wdir=0
#edit 10/21/2019: fix spelling error (calibration is not spelled with two ls)
calibration_curve=0.3 #0.3 oder 3
mass_flow_controller='X'
calibration_factor=1
scaling_factor=0.637
scale=250
ref_length=1/250
ref_height=None
gas_name='C12'
mol_weight=28.97
gas_factor=0.5
full_scale_wtref=6
full_scale_flow_rate=0.5

#edit 02/25/2020: added ability to run script in basic mode (if 'functions_mode' variable is set to 'basic'), which runs only the core features of the script,
#but much faster than full mode (if 'functions_mode' variable is set to 'full'), which runs all functions of the script. 
functions_mode='full'

#edit 05/31/2020: added abillity to determine y-axis range in puff plots using variable axis_range. Current options include 'auto' (whih determines y-axis limits
#automatically for each individual puff seperately), and 'same' (which sets y-axis limits from 0 to 1.05 times the maximum concentration in the time series). Potentially 
#add option to manually specify axis limits in the future. 
axis_range='auto'

#todo: add units (09/18/2019)

""" Perform a routine data analysis of puff concentration data. Runs essentially all analysis routines 
available in windtunnel package"""
# edit 07/23/2020: new function, performs routine data analysis of puff data. Runs essentially all
# analysis routines available in windtunnel package. Based largely on example_puff_analysis.py script.
# Improved and more secure communication with GUI. Makes PAPE_GUI_code_puff.py redundant. Largely
# replaces example_puff_analysis.py.


# Initialise dict object to store instances of PuffConcentration.
# If you only have one file to analyse you can remove the loops
# and replace them with:
# mydata = PuffConcentration.from(path + file)
# mydata.calc_net_concentration()
# etc.
# edit 09/19/2019: added dictionaries for full scale analysis
# edit 01/14/2020: added dictionaries for non-dimensional data analysis
conc_ts = {}
conc_ts.fromkeys(namelist)
conc_ts_fs = conc_ts
conc_ts_nd = conc_ts
dict_conc_ts = conc_ts
dict_conc_nd = conc_ts
# edit 08/05/2019: new dictionary called ensemble_ts, stores data for performing ensemble analysis
ensemble_ts = {}
ensemble_ts.fromkeys(namelist)
ensemble_ts_fs = ensemble_ts
ensemble_ts_nd = ensemble_ts
# edit 08/08/2019: new dictionary to store statistics
statistics = {}
statistics.fromkeys(namelist)
statistics_fs = statistics
statistics_nd = statistics
for name in namelist:
    # edit 10/21/2019:added option to read ambient conditions from csv file
    ambient_conditions = wt.PuffConcentration.get_ambient_conditions(path=path, name=name, input_file=path + csv_file)
    if ambient_conditions is None:
        []
    else:
        x_source, y_source, z_source, x_measure, y_measure, z_measure, pressure, temperature, wdir, calibration_curve, mass_flow_controller, calibration_factor, scaling_factor, scale, ref_length, \
        ref_height, gas_name, mol_weight, gas_factor, full_scale_wtref, full_scale_flow_rate = wt.PuffConcentration.read_ambient_conditions(
            ambient_conditions, name)
    files = wt.get_files(path, name)
    conc_ts[name] = {}
    conc_ts[name].fromkeys(files)
    conc_ts_fs[name] = conc_ts[name]
    # edit 08/05/2019: new dictionary called ensemble_ts, stores data for performing ensemble analysis.
    ensemble_ts[name] = {}
    ensemble_ts[name].fromkeys(files)
    ensemble_ts_fs[name] = ensemble_ts[name]
    # edit 08/08/2019: new dictionary to store statistics
    statistics[name] = {}
    statistics[name].fromkeys(namelist)
    statistics_fs[name] = statistics[name]
    for file in files:
        conc_ts[name][file] = wt.PuffConcentration.from_file(path + file)
        # edit 09/19/2019: added calculations necessary for full scale analysis.
        conc_ts[name][file].ambient_conditions(x_source=x_source, y_source=y_source, z_source=z_source,
                                               x_measure=x_measure, y_measure=y_measure, z_measure=z_measure,
                                               pressure=pressure,
                                               temperature=temperature,
                                               calibration_curve=calibration_curve,
                                               mass_flow_controller=mass_flow_controller,
                                               calibration_factor=calibration_factor)
        # conc_ts[name][file].scaling_information(scaling_factor=0.637,scale=250,
        # ref_length=1/250,ref_height=None)
        conc_ts[name][file].scaling_information(scaling_factor=scaling_factor, scale=scale,
                                                ref_length=ref_length, ref_height=ref_height)
        # conc_ts[name][file].tracer_information(gas_name='C12',
        # mol_weight=28.97/1000,
        # gas_factor=0.5)
        conc_ts[name][file].tracer_information(gas_name=gas_name,
                                               mol_weight=mol_weight,
                                               gas_factor=gas_factor)
        # conc_ts[name][file].full_scale_information(full_scale_wtref=6,
        # full_scale_flow_rate=0.5)
        conc_ts[name][file].full_scale_information(full_scale_wtref=full_scale_wtref,
                                                   full_scale_flow_rate=full_scale_flow_rate)
        conc_ts[name][file].convert_temperature()
        conc_ts[name][file].calc_wtref_mean()
        conc_ts[name][file].calc_model_mass_flow_rate()
        conc_ts[name][file].calc_net_concentration()
        # edit 10/10/2019: moved offset correction to before clear zeros.
        # conc_ts[name][file].offset_correction()
        # edit 07/24/2019: clear all data points with a negative net_concentration. Function can be turned on or off
        # conc_ts[name][file].clear_zeros()
        conc_ts[name][file].calc_c_star()

        conc_ts_fs[name][file] = conc_ts[name][file]
        # conc_ts_fs[name][file].to_full_scale()

        if full_scale == 'ms':
            dict_conc_ts = conc_ts
            dict_ensemble_ts = ensemble_ts
            dict_statistics = statistics
        elif full_scale == 'fs':
            dict_conc_ts = conc_ts_fs
            dict_conc_ts[name][file].to_full_scale()
            dict_ensemble_ts = ensemble_ts_fs
            #           dict_ensemble_ts[name][file].to_full_scale()
            dict_statistics = statistics_fs
        #           dict_statistics[name][file].to_full_scale()
        elif full_scale == 'nd':
            # edit 01/14/2020: added option to perform data anlysis in non-dimensional mode
            dict_conc_ts = conc_ts_nd
            print(dict_conc_ts[name][file].keys())
            dict_conc_ts[name][file].to_non_dimensional()
            dict_ensemble_ts = ensemble_ts_fs
            #           dict_ensemble_ts[name][file].to_full_scale()
            dict_statistics = statistics_fs
        #           dict_statistics[name][file].to_full_scale()
        else:
            print(
                "Error: invalid input for full_scale. Data can only be computed in model scale (full_scale='ms'), full scale (full_scale='fs'), or non-dimensionally (full_scale=nd).")


        #Example puff signal
        #Add zeros to make to puff
        #import pandas as pd 
        #import matplotlib.pyplot as plt
        #c_star = dict_conc_ts[next(iter(dict_conc_ts))][next(iter(dict_conc_ts[next(iter(dict_conc_ts))]))].c_star
        #idx = c_star.index
        # Add multiple zeros (e.g., 5) at each end
        #num_zeros = 50000
        #new_idx_start = [idx[0] - (i+1) for i in range(num_zeros)]
        #new_idx_end = [idx[-1] + (i+1) for i in range(num_zeros)]
        # Create Series with zeros
        #start_zeros = pd.Series(0, index=new_idx_start)
        #end_zeros = pd.Series(0, index=new_idx_end)
        # Concatenate the Series
        #c_star = dict_conc_ts[next(iter(dict_conc_ts))][next(iter(dict_conc_ts[next(iter(dict_conc_ts))]))].c_star
        #c_star_modified = pd.concat([start_zeros, c_star, end_zeros])
        #c_star_modified = pd.concat([c_star,c_star_modified])

        # Update the dictionary
        #dict_conc_ts[next(iter(dict_conc_ts))][next(iter(dict_conc_ts[next(iter(dict_conc_ts))]))].c_star = c_star_modified
        plt.plot(conc_ts[next(iter(dict_conc_ts))][next(iter(dict_conc_ts[next(iter(dict_conc_ts))]))].c_star)
        plt.show()


        dict_conc_ts[name][file].detect_begin_release_period()
        dict_conc_ts[name][file].begin_release_index_unmasked = dict_conc_ts[name][file].begin_release_index
        dict_conc_ts[name][file].detect_end_release_period()
        dict_conc_ts[name][file].calc_release_length()
        dict_conc_ts[name][file].get_dosage()
        # dict_conc_ts[name][file].get_mean_puff()
        dict_conc_ts[name][file].detect_arrival_time(time_threshold=time_threshold)
        dict_conc_ts[name][file].detect_leaving_time(time_threshold=time_threshold)
        dict_conc_ts[name][file].get_residence_time()
        dict_conc_ts[name][file].get_peak_concentration()
        dict_conc_ts[name][file].get_peak_time()
        dict_conc_ts[name][file].get_mask(threshold_concentration=threshold_concentration,
                                          threshold_dosage=threshold_dosage, n_exclude=n_exclude)
        # edit 02/28/2020: moved get_mean_puff to after get_mask to allow referencing of mask variable in get_mean_puff variable
        dict_conc_ts[name][file].get_mean_puff() 
        dict_conc_ts[name][file].get_ascent_time()
        dict_conc_ts[name][file].get_descent_time()
        # Pass a threshold concentration to the results
        # in order to remove all puffs with a maximum
        # concentration beneath the threshold.
        dict_conc_ts[name][file].apply_threshold_concentration(threshold_concentration=threshold_concentration)

        # Pass a threshold dosage to the results
        # in order to remove all puffs with a total dosage
        # beneath the threshold.
        dict_conc_ts[name][file].apply_threshold_dosage(threshold_dosage=threshold_dosage)
        # Test each puff against the average puff of the
        # measurement. Save the results in a variable
        deviations = dict_conc_ts[name][file].check_against_avg_puff()
        # Save output to a variable
        # edit 08/08/2019: renamed get_puff_statistics function to get_puffs. This is to avoid confusion with the
        # new calc_puff_statistics function whicch calculates the actual statistics.
        results = dict_conc_ts[name][file].get_puffs()
        #results = dict_conc_ts[name][file].get_puffs()
        results_keylist = results.keys().tolist()
        # edit 08/05/2019: new dictionary (same as above) and class to perform ensemble analysis
        dict_ensemble_ts[name][file] = {}
        dict_ensemble_ts[name][file].fromkeys(results_keylist)
        dict_statistics[name][file] = {}
        dict_statistics[name][file].fromkeys(results_keylist)
        for key in results_keylist:
            print(key)
            dict_ensemble_ts[name][file][key] = wt.EnsembleAnalysis.from_results(results[key])
            dict_ensemble_ts[name][file][key].ambient_conditions(x_source=x_source, y_source=y_source,
                                                                 z_source=z_source, x_measure=x_measure,
                                                                 y_measure=y_measure, z_measure=z_measure,
                                                                 pressure=pressure,
                                                                 temperature=temperature,
                                                                 calibration_curve=calibration_curve,
                                                                 mass_flow_controller=mass_flow_controller,
                                                                 calibration_factor=calibration_factor)

            dict_ensemble_ts[name][file][key].scaling_information(scaling_factor=scaling_factor, scale=scale,
                                                                  ref_length=ref_length, ref_height=ref_height)
            dict_ensemble_ts[name][file][key].tracer_information(gas_name=gas_name,
                                                                 mol_weight=mol_weight,
                                                                 gas_factor=gas_factor)
            # conc_ts[name][file].tracer_information(gas_name='C12',
            dict_ensemble_ts[name][file][key].get_ensemble_min()
            # edit 08/08/2019: added functions get_ensemble_max, get_ensemble_mean, get_ensemble_variance, and plot_convergence_ensemble.
            dict_ensemble_ts[name][file][key].get_ensemble_max()
            dict_ensemble_ts[name][file][key].get_ensemble_mean()
            dict_ensemble_ts[name][file][key].get_ensemble_variance()
            #outcommend after
            #dict_ensemble_ts[name][file][key].plot_convergence_ensemble(key=key, name=name, path=path,
            #                                                            full_scale=full_scale)

            # edit 08/12/2019: added calculation of classes. See Bachelor Thesis of Anne Philip (2010) for more details
            if functions_mode == 'basic':
                []
            elif functions_mode == 'full':
                dict_ensemble_ts[name][file][key].calc_class_width(n=5)
                dict_ensemble_ts[name][file][key].calc_class_boundaries()
                # edit 08/13/2019: added functions get_class_frequency and plot_class_statistics
                dict_ensemble_ts[name][file][key].get_class_frequency()

                # edit 02/25/2020: added saving of full scale and non-dimensional data
                # edit 07/23/2020: check to make sure that directory where data is to be saved exists
                wt.check_directory(path + 'Puff_Data\\' + name[:name.find('.')] + '\\')
                if full_scale == 'ms':
                    dict_ensemble_ts[name][file][key].save2file_ms_ensemble(file, key,
                                                                            out_dir=path + 'Puff_Data\\' + name[
                                                                                                           :name.find(
                                                                                                               '.')] + '\\')
                elif full_scale == 'fs':
                    dict_ensemble_ts[name][file][key].save2file_fs_ensemble(file, key,
                                                                            out_dir=path + 'Puff_Data\\' + name[
                                                                                                           :name.find(
                                                                                                               '.')] + '\\')
                elif full_scale == 'nd':
                    dict_ensemble_ts[name][file][key].save2file_nd_ensemble(file, key,
                                                                            out_dir=path + 'Puff_Data\\' + name[
                                                                                                           :name.find(
                                                                                                               '.')] + '\\')
                else:
                    print(
                        "Error: invalid input for full_scale. Data can only be computed in model scale (full_scale='ms'), full scale (full_scale='fs'), or non-dimensionally (full_scale=nd).")
                dict_ensemble_ts[name][file][key].plot_class_statistics(key=key, name=name, path=path,
                                                                        full_scale=full_scale)
            else:
                print(
                    "Error: invalid input for functions_mode. Program can only be run in basic mode (functions_mode='basic') or full mode (functions_mode='full').")

                # edit 08/08/2019: added calculation of statistical values
            dict_statistics[name][file][key] = wt.EnsembleAnalysis.from_results(results[key])
            dict_statistics[name][file][key].calc_puff_statistics(x_source=x_source, y_source=y_source,
                                                                  z_source=z_source, x_measure=x_measure,
                                                                  y_measure=y_measure, z_measure=z_measure,
                                                                  pressure=pressure, temperature=temperature,
                                                                  wtref=full_scale_wtref, wdir=wdir)

            # Save DataFrame to txt file
        # dict_conc_ts[name][file].save2file(file)
        # edit 02/25/2020: added saving of full scale and non-dimensional data
        # edit 07/23/2020: check to make sure that directory where data is to be saved exists
        wt.check_directory(path + 'Puff_Data\\' + name[:name.find('.')] + '\\')
        if full_scale == 'ms':
            dict_conc_ts[name][file].save2file_ms(file, out_dir=path + 'Puff_Data\\' + name[:name.find('.')] + '\\')
        elif full_scale == 'fs':
            dict_conc_ts[name][file].save2file_fs(file, out_dir=path + 'Puff_Data\\' + name[:name.find('.')] + '\\')
        elif full_scale == 'nd':
            dict_conc_ts[name][file].save2file_nd(file, out_dir=path + 'Puff_Data\\' + name[:name.find('.')] + '\\')
        else:
            print(
                "Error: invalid input for full_scale. Data can only be computed in model scale (full_scale='ms'), full scale (full_scale='fs'), or non-dimensionally (full_scale=nd).")
            # Save DataFrame to excel file
        writer = pd.ExcelWriter(path + 'test.xlsx')
        results.to_excel(writer, sheet_name='Puff Test')
        # edit 07/24/19: plot time series
        # edit 09/23/19: added proper plotting of full scale variables
        if functions_mode == 'basic':
            dict_conc_ts[name][file].plot_puff(path=path, name=name, full_scale=full_scale, n_puffs=5,
                                               axis_range=axis_range)
        elif functions_mode == 'full':
            dict_conc_ts[name][file].plot_puff(path=path, name=name, full_scale=full_scale, n_puffs='all',
                                               axis_range=axis_range)
        else:
            print(
                "Error: invalid input for functions_mode. Program can only be run in basic mode (functions_mode='basic') or full mode (functions_mode='full').")

        dict_conc_ts[name][file].plot_mean_puff(path=path, name=name, stats='on', dist='off', full_scale=full_scale)

    # Preliminary hist plots of the results DataFrame.
plt.figure(0)
results['peak concentration'].plot.hist(title='Peak Concentration')
plt.figure(1)
results['peak time'].plot.hist(title='Peak Time')
plt.figure(2)
results['arrival time'].plot.hist(title='Arrival Time')
plt.figure(3)
results['leaving time'].plot.hist(title='Leaving Time')
plt.figure(4)
results['ascent time'].plot.hist(title='Ascent Time')
plt.figure(5)
results['descent time'].plot.hist(title='Descent Time')



# Initialize and process data as normal
puff_conc = PuffConcentration(time, wtref, slow_FID, fast_FID, open_rate)

# Run individual analyses
puff_conc.calculate_statistics()
puff_conc.calculate_turbulence_statistics()

# Or run complete analysis
results = puff_conc.run_full_analysis()

print(results)
