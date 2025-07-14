# -*- coding: utf-8 -*-

import windtunnel as wt
import numpy as np

# This is an example script for the use of a PointConcentration object.
# The functionality of the PointConcentration class is shown, as well
# as some of the functionality inherited from pandas.Dataframe.
# The PointConcentration class returns a DataFrame, using the standard
# measurement txt-file output as input. The columns in the input file
# are expected to be time, wtref, slow FID, fast ID, release signal and
# open_rate, where release signal will be ignored.

# Path to your data (current path leads to example data)
#path = '\\\\ewtl2\\work\\Johannes\Puff_Beispiele\\'
path_dir = "/home/sabrina/Desktop/Schreibtisch/Arbeit_2025/WTSoftwareUtilitiesShare"
#sys.path.append(os.path.abspath(path_dir))
import windtunnel as wt

#Path to your input data
#path = f"{path_dir}/ExampleData/InputData/Concentration/"
path = f"{path_dir}/ExampleData/InputData/Beispiel Umrechnung zur Kontrolle/"
# Name of your measurement files prefix
#namelist = ['BFS_BD3_MP01_000']
namelist = ['UBA_GA_02_04_01_000_1_001']

#Path to your output folder for average files and plots
output_path = f"{path_dir}/ExampleData/Results/"

#For PointData for the functions to work the columns of the file should be: time, wtref, slow_FID, fast_FID, open_rate
#(See manual for the description of the variables)

#Name of csv file which contains ambient conditions data. Multiple diff. ambient conditions for diff datasets can be read-in at ones
#If no file given or configuration wrong, the program ressorts to try reading-in given values manually below. 
#csv_file='ambient_conditions.csv'

#csv_file= f"{path_dir}/ExampleData/ParameterFiles/ambient_conditions.csv"
csv_file= f"{path_dir}/ExampleData/ParameterFiles/ambient_conditions_.UBA_GA.csv"
parameters_PerFolder = False
        

#Variables and Parameters if no ambient_conditions.csv file overgiven

#If at the end calculate entdimensionalised or full scale transform quantities
#Default: nd:entdimensionalise, ms:model scale, fs:full scale.    
full_scale='ms'  

#Source location  [mm]
x_source=0
y_source=0
z_source=0

#Source mass flow controller, calibration settings
mass_flow_controller=0.3 #[l/h]*1/100 #'X'  #Controller(settings) used, just a name placeholder for orientation, not used yet
#If calibration performed on a controller, corrects actual max. flow capacity of controller
calibration_curve=0.3     #0.3 oder 3
calibration_factor=1      #

#Gas characteristics
gas_name='C12'           #Just placeholder name variable for orientation, not used for anything
gas_factor=0.5   #[-]    #!!! Needs to be calculate/specificate f.e. if gas changes 
mol_weight=29.0 #28.97 #Air [g/mol]


#Measurement location [mm]
x_measure=1020 #855.16
y_measure= 0    #176.29
z_measure= 5     #162

#Surrounding conditions
pressure=101325         #100938  #[Pa]
temperature=20             #23.5  #[°C]

#Model to Reality scaling
scale=400                     #250      #Model/Reality
scaling_factor=0.5614882               #0.637       #USA1 to selected ref pos.?
ref_length=1/400              #1/250           #Lref
ref_height=100/400            #None            #Href
full_scale_wtref=10             #6         #Uref_fullscale
full_scale_flow_rate=0.002     #Q_amb[kg/s]?   #0.5   #Qv_fullscale
full_scale_temp=20             #[°C]
full_scale_pressure=101325   #[Pa]
#Q_ambient[kg/s] ->  Q[m³/s]=Q[kg/s]*R*T/(M*p)

#edit 07/23/2020: added variable wdir for wind direction. To be implemented in future. ##
#wdir=0
#edit 07/23/2020: added variable axis_range. Reserved for future implementation of axis range specification, 
#analogously to puff mode
#axis_range='auto'


#edit 07/23/2020: added variable axis_range. Reserved for future implementation of axis range specification, 
#analogously to puff mode
axis_range='auto'

conc_ts = {}
conc_ts.fromkeys(namelist)
conc_ts_fs = conc_ts
conc_ts_nd = conc_ts
dict_conc_ts = conc_ts
dict_conc_nd = conc_ts
dict_conc_fs = conc_ts
data_dict = {}
data_dict.fromkeys(namelist)
for name in namelist:
    # edit 10/21/2019:added option to read ambient conditions from csv file
    ambient_conditions = wt.PointConcentration.get_ambient_conditions(path=path, name=name, input_file=csv_file)
    if ambient_conditions is None:
        []
    else:
        x_source, y_source, z_source, x_measure, y_measure, z_measure, pressure, temperature, calibration_curve, mass_flow_controller, calibration_factor, scaling_factor, scale, ref_length, \
        ref_height, gas_name, mol_weight, gas_factor, full_scale_wtref, full_scale_flow_rate, full_scale_temp, full_scale_pressure = wt.PointConcentration.read_ambient_conditions(
            ambient_conditions, name)
    files = wt.get_files(path, name)
    conc_ts[name] = {}
    conc_ts[name].fromkeys(files)
    for file in files:
        if parameters_PerFolder == False:
            #Read in ambient condition column for each ts
            ambient_conditions = wt.PointConcentration.get_ambient_conditions(path=path, name=file, input_file=csv_file)
            #Else read/use given default from cell above
            if ambient_conditions is None:
                []
            #Read ambient conditions from csv file
            else:
                x_source, y_source, z_source, x_measure, y_measure, z_measure, pressure, temperature, calibration_curve, mass_flow_controller, calibration_factor, scaling_factor, scale, ref_length, \
                ref_height, gas_name, mol_weight, gas_factor, full_scale_wtref, full_scale_flow_rate, full_scale_temp, full_scale_pressure = wt.PointConcentration.read_ambient_conditions(
                ambient_conditions, file)

        conc_ts[name][file] = wt.PointConcentration.from_file(path + file)
        # edit 09/19/2019: edited code to avvound for moving a priori information to beginning of script.
        # conc_ts[name][file].ambient_conditions(x=855.16,y=176.29,z=162,pressure=1009.38,
        # temperature=23.5,
        # calibration_curve=0.3,#0.3 oder 3
        # mass_flow_controller='X',
        # calibration_factor=1)
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
                                                   full_scale_flow_rate=full_scale_flow_rate,
                                                   full_scale_temp = full_scale_temp,
                                                   full_scale_pressure=full_scale_pressure)
        conc_ts[name][file].convert_temperature()
        conc_ts[name][file].calc_wtref_mean()
        conc_ts[name][file].calc_model_mass_flow_rate(usingMaxFlowRate="True")
        conc_ts[name][file].calc_net_concentration()
        # edit 07/24/2019: clear all data points with a negative net_concentration. Function can be turned on or off
        #conc_ts[name][file].clear_zeros()
        conc_ts[name][file].calc_c_star()

        # edit 07/27/2020: added options for outputting data in full-scale, model scale, and non-dimensionally.
        if full_scale == 'ms':
            dict_conc_ts = conc_ts
        elif full_scale == 'fs':
            dict_conc_ts = conc_ts_fs
            dict_conc_ts[name][file].to_full_scale()
        elif full_scale == 'nd':
            dict_conc_ts = conc_ts_nd
            dict_conc_ts[name][file].to_non_dimensional()
        else:
            print(
                "Error: invalid input for full_scale. Data can only be computed in model scale (full_scale='ms'), full scale (full_scale='fs'), or non-dimensionally (full_scale=nd).")
        dict_conc_ts[name][file].plot_hist_conc(path=path, name=name)
        #dict_conc_ts[name][file].plot_hist_conc(path=path, name=name)
        # Save full scale results in a variable.
        # to_full_scale() will only work if all
        # information necessary has already been
        # given and computed.
        # data_dict[name] = conc_ts[name][file].to_full_scale()
        # Save full scale results. Requires to_full_scale()
        # edit 10/21/2019. Save to path of data, not to installation path of windtunnel!
        wt.check_directory(path + 'Point_Data\\' + name[:name.find('.')] + '\\')
        # dict_conc_ts[name][file].save2file_fs(file,out_dir=path+'Point_Data\\'+name[:name.find('.')]+'\\')
        if full_scale == 'ms':
            dict_conc_ts[name][file].save2file_ms(file, out_dir=path + 'Point_Data\\' + name[:name.find('.')] + '\\')
        elif full_scale == 'fs':
            dict_conc_ts[name][file].save2file_fs(file, out_dir=path + 'Point_Data\\' + name[:name.find('.')] + '\\')
        elif full_scale == 'nd':
            dict_conc_ts[name][file].save2file_nd(file, out_dir=path + 'Point_Data\\' + name[:name.find('.')] + '\\')
        else:
            print(
                "Error: invalid input for full_scale. Data can only be computed in model scale (full_scale='ms'), full scale (full_scale='fs'), or non-dimensionally (full_scale=nd).")
            # Save model scale results
        # conc_ts[name][file].save2file_ms(file,out_dir=path+'Puff_Data\\'+name[:name.find('.')]+'\\')
        # Save average values. Requires to_full_scale()
        # conc_ts[name][file].save2file_avg(file,out_dir=path+'Puff_Data\\'+name[:name.find('.')]+'\\')





from windtunnel.concentration.utils import batch_combine_data, load_combined_data_from_csv

path_dir = "/home/sabrina/Desktop/Schreibtisch/Arbeit_2025/WTSoftwareUtilitiesShare"
output_path = f"{path_dir}/ExampleData/Results/"

file_path = "/home/sabrina/Desktop/Schreibtisch/Arbeit_2025/windtunnel_software/Data/"
file_names = [
    "Point_Data_stats\BFS_BD3_MP01_00\_stats_BFS_BD3_MP01_000_01.ts#0",
    "Point_Data_stats\BFS_BD3_MP01_00\_stats_BFS_BD3_MP01_000_02.ts#0"
]
batch_combine_data(file_path, file_names, output_path, 
                  folder_combined="stats_only", 
                  output_filename="stats_measurements.csv",
                  file_type="stats")



#Try fluctuations analysis
conc_ts[name][file].analyze_concentration_fluctuations()

#from windtunnel.concentration.CompareDatasets import compare_point_concentrations
#compare_point_concentrations(conc_ts[namelist[0]][namelist[0]], conc_ts[namelist[1]][namelist[1]])


#Try overview showing
from windtunnel.concentration.CompareDatasets import compare_point_concentrations_2
from windtunnel.concentration.CompareDatasets import compare_point_concentrations_3
from windtunnel.concentration.CompareDatasets import powerDensityPlot
#functionsForOverview = ["all"]
functionsForOverview = [
    "Histogram",
    "Pdf",
    "Cdf",
    "Means",
    "QuantilPlot",
    "ScatterPlot",
    "ResidualPlot",
    "Autocorrelation",
    "PowerDensity"
]


#powerDensityPlot([conc_ts[namelist[0]][namelist[0]], conc_ts[namelist[1]][namelist[1]]],dimensionless="False",plot=True,labels=None,xLabel=None,yLabel=None,xAchse=None,yAchse=None)
    
#compare_point_concentrations_2([conc_ts[namelist[0]][namelist[0]], conc_ts[namelist[1]][namelist[1]], conc_ts[namelist[0]][namelist[0]]],functionsForOverview)
#compare_point_concentrations_3([conc_ts[namelist[0]][namelist[0]], conc_ts[namelist[1]][namelist[1]]],functionsForOverview)


from windtunnel.concentration.utils import stl_to_2d_plot, add_crosses, show_multiple_projections
#from windtunnel.concentration.utils import plot_stl_3d, add_crosses_3d
import matplotlib.pyplot as plt
import numpy as np

# Path to your STL file
stl_file = "/home/sabrina/Schreibtisch/Arbeit_2025/FreeCAD/20240206_BfS_model_scale_complete.stl"
    
points_fs = [ 
    (-408,0),
    (-408,0),
    (-388,-42),
    (-372,108),
    (-340,0)
]
#Read in from avg_fs files
values_fs= [0.24967,1.48855,0.00608,0.00004,0.12276]

#for full scale trafo
scale=400   

#for lat/lon trafo
r_earth = 6371 * 10**3
ref_position=(48.137154, 11.576124) #lat,lon
#Average concentration values
# Define thresholds and corresponding colors
thresholds = [10, 20, 30, 40]
colors = ['blue','green', 'yellow', 'orange', 'red']

#Call stl to polygon print
fig, ax = stl_to_2d_plot(stl_file, projection='xy',toFullScale="True",scaling=scale, toLatLon="True",ref_coords=ref_position)
# Add crosses to the plot#
add_crosses(ax, points_fs, values=values_fs, thresholds=thresholds, colors=colors,size=80, linewidth=1.5)
ax.set_xlabel("X_fs[m]")
ax.set_ylabel("Y_fs[m]")
plt.tight_layout()
plt.savefig("stl_with_crosses.png", dpi=300)
plt.show()

