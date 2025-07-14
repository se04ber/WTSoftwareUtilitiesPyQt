# -*- coding: utf-8 -*-

import numpy as np
import logging
import windtunnel as wt
import matplotlib.pyplot as plt


# Create logger
logger = logging.getLogger()

#%%#
# This is an example script. It can be modified to your personal needs.
# Specify path to data, path to wtref, output paths for plots and txt file, 
# file type for plots, name of files, scale and desired mode of analysis.
# Input paths for data and wtref with a list of names of the measurement files
#path = 'Z:/projects/BFS/raw measurements/boundary layer/time series/' # path to timeseries folder
#path="/home/sabrina/Schreibtisch/Arbeit_2025/windtunnel_software/Data/"
path ="/home/sabrina/48/LDA/"
#wtref_path = '/home/sabrina/' #'Z:/projects/BFS/raw measurements/boundary layer/wtref/'
wtref_path="/home/sabrina/48/USA/Flowtest_UV_48"
namelist = ['Flowtest_UV_48.000001.txt']
#namelist = ['BFS_BD3_MP01_000_01']
#namelist = ['BFS_BD3_MP01_000_01.ts#0']#['BFS_BL_DOC_UV_01'] 

#txt_path = 'path/to your/txt-spectra and mean folder/'
#txt_path = '/home/sabrina/Desktop/Schreibtisch/Arbeit_2025/windtunnel_software/Data/' #None
txt_path = "/home/sabrina/48/USA/"
#edit 06/20/19: set ref_path to none for unknown reference path

#ref_path  =None
ref_path ="/home/sabrina/48/USA/"
#ref_path = '//cifs-isi03.cen.uni-hamburg.de/ewtl/work/_EWTL Software/Python/Reference data/'

#plot_path = '//cifs-isi03.cen.uni-hamburg.de/ewtl/projects/UBA/Raw measurements/Boundary Layer/plots/'
#plot_path = 'path/to your/plot-folder'
plot_path = '/home/sabrina/Desktop/Schreibtisch/Arbeit_2025/windtunnel_software/TestPlots/'
out_dir = '/home/sabrina/Desktop/Schreibtisch/Arbeit_2025/' #path/to your/txt-folder/for Timeseries-txt/'


path ="/home/sabrina/Desktop/Schreibtisch/Arbeit_2025/FreeCAD/UBA/vertikales Profil/Zeitserien/"
wtref_path="/home/sabrina/Desktop/Schreibtisch/Arbeit_2025/FreeCAD/UBA/vertikales Profil/wtref/"
namelist = ['UBA_BL_Rep2025_UW_01']


#ref_path = None
header_information = {"[main project name]" : 'BFS',
                      "[sub project name]" : 'Boundary layer',
                      "[wind tunnel facility]" : 'WOTAN',
                      "[model scale]" : 400,
                      "[Zref - model scale[mm]]" : 400,
                      "[wind direction [°]]" : 0,
                      "[Lref - model scale[m]]" : 1/400,
                      "[number of columns]" : 3,
                      "[directly measured components]" : 'UW',
                      "[confidence intervall (U,V,W)/Uref[-]]" : 0.03
    }

scale_factor = 1

file_type = 'png'
scale = header_information["[model scale]"]

checker = True

#plot scatter
plot = True
scatter = True
save_data = True
new_VDI_ref = True #False #True

#Customized (Turbulenz-)Plots variables
z_axis_lim = (None,None) #(min,max) both values must be given or None
dpi= 300
shown_z_value = (None, None) #(min, max) masking data higher or lower 

#edit 08/08/2019: add errors for all quantities
u_err=0
v_err=0
#u_v_error is for plotting wind data, where u and v are plotted on one plot. 
#It is assumed in this case the error to be plotted is the u error. 
#TODO: modify  code to allow for seperate errors for u and v on the same plot. 
u_v_err=u_err
I_u_err=0
I_v_err=0
flux_err=0
lux_err=0

# 1 = vertical profile
# 2 = lateral profile
# 3 = convergence test
# 4 = Reynolds Number Independence
# 5 = Langitudinal profil

mode = 1
if mode == 2:
    outdata_path = '/path/to/your/outdata_lat/'# format in npz
else:
    outdata_path = plot_path# '/path/to/your/outdata/'# format in npz

# Check if all necessary output directories exist
wt.check_directory(plot_path)
wt.check_directory(txt_path)

time_series = {}
time_series.fromkeys(namelist)

#edit 06/20/19: create seperate time series for equidistant data
time_series_eq = {}
time_series_eq.fromkeys(namelist)

#set data_nd to 1 if using non-dimensional data
data_nd=0
# Gather all files into Timeseries objects, save raw timeseries

for name in namelist:
    files = wt.get_files(path,name)
    time_series[name] = {}
    time_series[name].fromkeys(files)
    time_series_eq[name] = {}
    time_series_eq[name].fromkeys(files)    
    for i,file in enumerate(files):
        ts = wt.Timeseries.from_file(path+file)            
        ts.get_wind_comps(path+file)
        ts.get_wtref(wtref_path,name,index=i, vscale=scale_factor) #'' = name
        # edit 6/20/19: Assume that input data is dimensional, not non-dimensional
        if data_nd == 0:
           print('Warning: Assuming that data is dimensional. If using non-dimensional input data, set variable data_nd to 1')
           ts.nondimensionalise()
        else:
           if data_nd == 1:
              []
           else:
              print('Warning: data_nd can only be 1 (for non-dimensional input data) or 0 (for dimensional input data)')        
        #edit 06/20/19: added seperate functionto  calculate equidistant timesteps             
        ts.adapt_scale(scale, Lref = header_information["[Lref - model scale[m]]"])         
        ts.mask_outliers()
        ts_eq=ts
        ts_eq.calc_equidistant_timesteps()  
        ts.index=ts.t_arr         
        ts.weighted_component_mean
        ts_eq.weighted_component_mean
        ts.weighted_component_variance
        ts_eq.weighted_component_variance
        ts.mean_magnitude
        ts_eq.mean_magnitude
        ts.mean_direction
        ts_eq.mean_direction
        ts.save2file(file, out_dir = out_dir, header_information=header_information)     
        time_series[name][file] = ts
        time_series_eq[name][file] = ts_eq




if files==[]:
   raise Exception('No Matching File Names Found. Please check namelist and/or path!') 

if checker == True:                   
    for name in namelist:
        # Check if positions in all files match for vertical profile
        files = wt.get_files(path, name)
        if mode == 1 or mode == 3 or mode == 4:
            for i in range(np.size(files)-2):
                if time_series[name][files[i]].x != time_series[name][files[i+1]].x:
                    raise Exception('Positions do not match! Check data file.')
                if time_series[name][files[i]].y != time_series[name][files[i+1]].y:
                   raise Exception('Positions do not match! Check data file.')
        # Check if positions in all files match for horizontal profile
        if mode == 2:
            for i in range(np.size(files)-2):
                if time_series[name][files[i]].x != time_series[name][files[i+1]].x:
                    raise Exception('Positions do not match! Check data file.')
                if time_series[name][files[i]].z != time_series[name][files[i+1]].z:
                    raise Exception('Positions do not match! Check data file.')
               
        if mode == 5:
            for i in range(np.size(files)-2):
                if time_series[name][files[i]].y != time_series[name][files[i+1]].y:
                    raise Exception('Positions do not match! Check data file.')
                if time_series[name][files[i]].z != time_series[name][files[i+1]].z:
                    raise Exception('Positions do not match! Check data file.')


# Iniate first layer of dictionaries for results
wind_comps = {}
wind_comps.fromkeys(namelist)
wind_stats = {}
wind_stats.fromkeys(namelist)
turb_data = {}
turb_data.fromkeys(namelist)
lux_data = {}
lux_data.fromkeys(namelist)
spectra_data = {}
spectra_data.fromkeys(namelist)

for name in namelist:
    #edit 6/12/19: add variables n_outliers_u and n_outliers_v to keep track of number of outliers
    n_outliers_u=0
    n_outliers_v=0
    # Iniate second layer of dictionaries for results 
    files = wt.get_files(path,name)
    wind_comps[name] = {}
    wind_comps[name].fromkeys(files)
    if mode != 3:
        wind_stats[name] = {}
        wind_stats[name].fromkeys(files)
        turb_data[name] = {}
        turb_data[name].fromkeys(files)
        lux_data[name] = {}
        lux_data[name].fromkeys(files)
        spectra_data[name] = {}
        spectra_data[name].fromkeys(files)
    for file in files:
        #edit 6/12/19: add variables n_outliers_u and n_outliers_v to keep track of number of outliers        
        n_outliers_u=n_outliers_u+time_series[name][file].n_outliers_u
        n_outliers_v=n_outliers_v+time_series[name][file].n_outliers_v
        #edit 6/12/19: add u and v outliers to determien total 
        wind_comps[name][file] = time_series[name][file].wind_comp1,\
                                 time_series[name][file].wind_comp2
       
        if mode == 3:
            # Perform convergence test for the two wind components and plot results
            # Average u and v component for different (default) intervals

            u_convergence = wt.convergence_test(time_series_eq[name][file].u_eq)
            v_convergence = wt.convergence_test(time_series_eq[name][file].v_eq)
            # Plot convergence test results for both components. The plot is saved in plot_path,
            # specified at the beginning of this example script.
            fig1,ax1 = plt.subplots(1)
            wt.plot_convergence_test(u_convergence,ylabel=time_series[name][file].wind_comp1+'_mean',ax=ax1)
            plt.tight_layout()
            fig1.savefig(plot_path + 'convergence_' + time_series[name][file].wind_comp1+'_mean' + '.' + file_type,
                        dpi=1000,bbox_inches='tight')
            fig2,ax2 = plt.subplots(1)
            wt.plot_convergence_test(v_convergence,ylabel=time_series[name][file].wind_comp2+'_mean',ax=ax2)
            plt.tight_layout()
            fig2.savefig(plot_path + 'convergence_' + time_series[name][file].wind_comp2+'_mean' + '.' + file_type,
                        dpi=1000,bbox_inches='tight')
    if checker == True:                   
        for name in namelist:
            # Check if positions in all files match for vertical profile
            files = wt.get_files(path, name)
            for file in files:
                # Calculate mean wind quantities
                dt = time_series[name][file].t_eq[1] - time_series[name][file].t_eq[0]
                wind_stats[name][file] = wt.calc_wind_stats_wght(time_series[name][file].t_transit,time_series[name][file].u,
                                                            time_series[name][file].v)
                turb_data[name][file] = wt.calc_turb_data(time_series[name][file].u.dropna(),
                                                        time_series[name][file].v.dropna())
                #edit 06/20/2019: changed script to adapt to dimensional and non-dimensional input data
                if data_nd==0:
                    lux_data[name][file] = wt.calc_lux_data(dt,
                                                        (time_series[name][file].u_eq.dropna().values*
                                                        time_series[name][file].wtref))
                
                #lux_data[name][file] = wt.calc_lux_data(dt,
                #                                     (time_series[name][file].u_eq.dropna().values))#*
                                                        #time_series[name][file].wtref/scale_factor))
                if data_nd==1:
                    lux_data[name][file] = wt.calc_lux_data(dt,
                                                        (time_series[name][file].u_eq.dropna().values))        
                
            if (mode == 1 or mode == 2 or mode== 5) and scatter:
                
                # Plot scatter plot of raw data
                plt.figure(files.index(file)+100)
                wt.plots.plot_scatter(time_series[name][file].u,
                                    time_series[name][file].v)
                plt.savefig(plot_path + 'scatter_' + file[:-4] + '.' + file_type)
                
                # Plot histograms of each component
                plt.figure(files.index(file)+200)
                wt.plots.plot_hist(time_series[name][file].u)
                plt.savefig(plot_path + 'hist_u_' + file[:-4] + '.' + file_type)
                plt.figure(files.index(file)+300)
                wt.plots.plot_hist(time_series[name][file].v)
                plt.savefig(plot_path + 'hist_v_' + file[:-4] + '.' + file_type)

                
                #edit 4/7/19: changed script to use dimensional wind values for calculating spectra to make sure that frequency is dimensionless
                spectra_data[name][file] = wt.calc_spectra(
                                                    time_series_eq[name][file].u_eq.dropna()*time_series_eq[name][file].wtref,
                                                    time_series_eq[name][file].v_eq.dropna()*time_series_eq[name][file].wtref,
                                                    time_series_eq[name][file].t_eq[~np.isnan(time_series_eq[name][file].t_eq)],
                                                    time_series_eq[name][file].z)
                # Save spectra data
                np.savetxt(txt_path + 'spectra_' + file[:-4] + '.txt',
                        np.vstack((spectra_data[name][file][0],
                                    spectra_data[name][file][1],
                                    spectra_data[name][file][2],
                                    spectra_data[name][file][3])).transpose(),
                        fmt='%.8f',
                        header="dimensionless spectra - smoothed according to reduced frequency bins"+'\n'+\
                        "frequency=0 where no energy content"+'\n'+\
                        "format: standard numpy.genfromtxt()"+'\n'+\
                        "variables = \"f_sm\" \"S_uu_sm\" \"S_vv_sm\" \"S_uv_sm\" ")
                
                # Plot spectra
                plt.figure(files.index(file)+400)
                wt.plots.plot_spectra(spectra_data[name][file][0],
                                    spectra_data[name][file][1],
                                    spectra_data[name][file][2],
                                    spectra_data[name][file][4],
                                    spectra_data[name][file][5],
                                    wind_comps[name][file],
                                    time_series[name][file].z,
                                    ref_path=ref_path)
                
                plt.savefig(plot_path + 'spectra_' + file[:-4] + '.' + file_type)
                plt.close('all')

                print('\n Start wavelet analysis for {}'.format(file))
                wavelet, scale = wt.calc_wavelet_transform(time_series[name][file].u_eq,
                                                        time_series[name][file].t_eq,
                                                        wavelet='morlet')
                # y_val = time_series[name][file].y-y_val_shift
                y_val = time_series[name][file].z            
                f_scale = y_val/(scale * np.mean(time_series[name][file].u_eq))

                # plot wavelet transform
                plt.figure(55)
                plt.style.use('classic')    
                wt.plots.plot_wavelet_transform(wavelet, 
                                        scale, 
                                        time_series[name][file].u_eq, 
                                        time_series[name][file].t_eq,
                                        y_val)

                plt.savefig(plot_path +  'wavelet_analysis_' + file +  '.' + file_type,
                            bbox_inches='tight', dpi=300)
                print(' Saved plot to:'+ plot_path + 'wavelet_analysis_' + file +  '.' + file_type)
                plt.close(55)
                print(' Finished wavelet analysis for {}'.format(file))



    # Initiate lists for all quantitites
    x = []
    y = []
    heights = []
    mean_mag = []
    u_mean = []
    u_mean_wght = []
    u_std = []
    u_std_wght = []
    v_mean = []
    v_mean_wght = []
    v_std = []
    v_std_wght = []
    I_u = []
    I_v = []
    fluxes = []
    lux = []
    wdir = []
    wtref = []
    
    for file in files:
        # Gather all quantities for a complete profile
        mean_mag.append(time_series[name][file].mean_magnitude)
        u_mean.append(np.mean(time_series[name][file].u))
        u_std.append(np.std(time_series[name][file].u))
        wtref.append(time_series[name][file].wtref)
        if mode !=4:
            x.append((time_series[name][file].x))
            y.append((time_series[name][file].y))
            heights.append((time_series[name][file].z))
            u_mean_wght.append(time_series[name][file].weighted_component_mean[0])
            u_std_wght.append(np.sqrt(time_series[name][file].weighted_component_variance[0]))
            v_mean.append(np.mean(time_series[name][file].v))
            v_mean_wght.append(time_series[name][file].weighted_component_mean[1])
            v_std.append(np.std(time_series[name][file].v))
            v_std_wght.append(np.sqrt(time_series[name][file].weighted_component_variance[1]))
            wdir.append(time_series[name][file].mean_direction)
            I_u.append(turb_data[name][file][0])
            I_v.append(turb_data[name][file][1])
            fluxes.append(turb_data[name][file][2])
            lux.append(lux_data[name][file])

    from flow.stats import calc_alpha  
    calc_alpha(u_mean, heights,d0=0.,BL_height=600.,BL=[])

    if save_data:

        wt.check_directory(outdata_path)
        outfile = outdata_path + name + '.npz'
        np.savez(outfile, x=x,y=y,heights=heights,mean_mag=mean_mag,u_mean=u_mean,v_mean=v_mean
                 ,I_u=I_u,I_v=I_v,fluxes=fluxes,lux=lux,wtref=wtref)

    if mode == 4:
        # Perform and plot results of Reynolds Number Independence test, no
        # output saved as txt, as programme ends at "break"
        plt.figure(0)
        wt.plots.plot_Re_independence(mean_mag, wtref, yerr=u_err,ymin=0,ymax=1)
        #wt.plots.plot_Re_independence(u_std, wtref, yerr=u_err,ymin=0,ymax=0.3)
        plt.tight_layout()
        plt.savefig(plot_path + 'Re_u' + name + '.' + file_type)
        break

   # Save quantities for vertical and lateral profiles    
    np.savetxt(txt_path + name + '_turb.txt',
               #np.vstack((x, y, heights, mean_mag, u_mean, u_mean_wght, v_mean,
               #           v_mean_wght, u_std, u_std_wght, v_std, v_std_wght,
               #           I_u, I_v, lux, fluxes, wdir, wtref)).transpose(),
               np.vstack((wdir, x, y, heights, wtref, u_mean, v_mean, u_std, v_std, fluxes, I_u, I_v, lux)).transpose(),
               fmt='%.8f', header=('General Information:\n\n'
                                   '{} [main project name]'.format(header_information['[main project name]'])+'\n'
                                   '{} [sub project name]'.format(header_information['[sub project name]'])+'\n'
                                   '{} [wind tunnel facility]'.format(header_information['[wind tunnel facility]'])+'\n'
                                   '1:{} [model scale]'.format(header_information['[model scale]'])+'\n'
                                   '{} [Zref - model scale[mm]]'.format(header_information['[Zref - model scale[mm]]'])+'\n'
                                   '{} [Zref - full scale[m]]'.format(header_information['[Zref - model scale[mm]]']/1000*header_information['[model scale]']) + '\n'
                                   '1:{} [Lref - model scale[m]]'.format(1/header_information['[Lref - model scale[m]]']) + '\n'
                                   '{} [Lref - full scale[m]]'.format(header_information['[Lref - model scale[m]]']/header_information['[model scale]']) + '\n\n'
                                   'Flow measurements:\n\n'
                                   '13 [number of columns]\n'
                                   '{}{} [directly measured components]'.format(wind_comps[name][file][0].upper(),wind_comps[name][file][1].upper())+'\n'
                                   'Winddirection [deg], X_fs [m], Y_fs [m], Z_fs [m], Uref-model [m/s], {c1}/Uref [-], {c2}/Uref [-], {c1}_rms [-], {c2}_rms [-],'.format(c1 = wind_comps[name][file][0].upper(), c2 = wind_comps[name][file][1].upper())+\
                                   ' {c3}\'{c4}\', I_{c3} [-], I_{c4} [-], L_ux (full scale) [m]'.format(c3 = wind_comps[name][file][0], c4 = wind_comps[name][file][1])))
               
               
               
               
               
               
               
                                   #('flow and turbulence parameters\n'
                                   #'units: dimensionless!\n'
                                   #'format: standard numpy.genfromtxt()\n'
                                   #'variables = \"x\" \"y\" \"z\" \"M\" '
                                   #'\"{0}_mean\" \"{0}_mean_wght\" '
                                   #'\"{1}_mean\" \"{1}_mean_wght\" \"{0}_std\"'
                                   #' \"{0}_std_wght\" \"{1}_std\" '
                                   #'\"{1}_std_wght\" \"I_{0}\" \"I_{1}\" '
                                   #'\"L{0}x\" \"{0}\'{1}\'_flux\" \"wdir\" '
                                   #'\"wtref\"'.format(wind_comps[name][file][0],
                                   #                   wind_comps[name][file][1])))  
    if mode == 1 and plot:
        # Plot results of a vertical profile
        # Wind components
        plt.figure(0, dpi=dpi)
        ret, lgd = wt.plots.plot_winddata(mean_mag,u_mean,v_mean,heights,yerr=u_v_err)
        plt.savefig(plot_path + 'wind_data_' + name + '.' + file_type,
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
    
        # Wind components, logarithmic y-axis
        plt.figure(1, dpi=dpi)
        ret, lgd = wt.plots.plot_winddata_log(mean_mag,u_mean,v_mean,heights,xerr=u_v_err)
        plt.tight_layout()
        plt.savefig(plot_path + 'wind_data_log_' + name + '.' + file_type,
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        
        # Turbulence intensity of the first component
        fig = plt.figure(2, dpi=dpi)
        wt.plots.plot_turb_int(I_u,heights,yerr=I_u_err,component='I_'+ts.wind_comp1,ref_path=ref_path,new_ref=new_VDI_ref,cut = shown_z_value)
        if (z_axis_lim[0] != None) and (z_axis_lim[1] != None):
            plt.ylim(z_axis_lim)
        plt.tight_layout()
        plt.savefig(plot_path + 'I_'+ts.wind_comp1+'_' + name + '.' + file_type)
        
        original_size  = fig.get_size_inches()
        original_width, original_height = original_size
        new_width = original_width/3
    
        fig = plt.figure(8, figsize = (new_width,original_height), dpi=dpi)
        wt.plots.plot_turb_int(I_u,heights,yerr=I_u_err,component='I_'+ts.wind_comp1,
                               ref_path=ref_path,new_ref=new_VDI_ref,cut = shown_z_value,
                               shown_Legend= False)
        if (z_axis_lim[0] != None) and (z_axis_lim[1] != None):
            plt.ylim(z_axis_lim)
        ax = plt.gca()
        ticks = np.linspace(0.05, 0.30, 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{tick:.2f}' for tick in ticks])
        #plt.xticks(rotation = 45)
        plt.subplots_adjust(left=0.25, bottom = 0.12)
        plt.tight_layout()
        plt.savefig(plot_path + 'I_'+ts.wind_comp1+'_' + name + '_shrink' + '.' + file_type)
        
        # Turbulence intensity of the second component
        plt.figure(3, dpi=dpi)
        wt.plots.plot_turb_int(I_v,heights,yerr=I_v_err,component='I_'+ts.wind_comp2,ref_path=ref_path,new_ref=new_VDI_ref,cut = shown_z_value)
        if (z_axis_lim[0] != None) and (z_axis_lim[1] != None):
            plt.ylim(z_axis_lim)
        plt.tight_layout()
        plt.savefig(plot_path + 'I_'+ts.wind_comp2+'_' + name + '.' + file_type)
        
        plt.figure(7,figsize=(new_width,original_height), dpi=dpi)
        wt.plots.plot_turb_int(I_v,heights,yerr=I_v_err,component='I_'+ts.wind_comp2,
                               ref_path=ref_path,new_ref=new_VDI_ref,cut = shown_z_value,
                               shown_Legend = False)
        if (z_axis_lim[0] != None) and (z_axis_lim[1] != None):
            plt.ylim(z_axis_lim)
        ax = plt.gca()
        ticks = np.linspace(0.04, 0.22, 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{tick:.2f}' for tick in ticks])
        #plt.xticks(rotation = 45)
        plt.subplots_adjust(left=0.25, bottom = 0.12)
        plt.tight_layout()
        plt.savefig(plot_path + 'I_'+ts.wind_comp2+'_' + name + '_shrink'+ '.' + file_type)
    
        # Profile of the fluxes
        plt.figure(4, dpi=dpi) 
        wt.plots.plot_fluxes(fluxes,heights,yerr=flux_err,component='w')
        plt.tight_layout()
        plt.savefig(plot_path + 'fluxes_' + name + '.' + file_type)
    
        # Profiles of the fluxes, logarithmic y-axis
        plt.figure(5, dpi=dpi)
        wt.plots.plot_fluxes_log(fluxes,heights,yerr=flux_err,component='w')
        plt.tight_layout()
        plt.savefig(plot_path + 'fluxes_log_' + name + '.' + file_type)
    
        # Double logarithmic profile of Lux data
        plt.figure(6, dpi=dpi)
        wt.plots.plot_lux(lux,heights,yerr=lux_err,component='w',ref_path=ref_path,new_ref=new_VDI_ref)
        plt.tight_layout()
        plt.savefig(plot_path + 'Lux_' + name + '.' + file_type)
        plt.close('all')
    if mode == 2:
        # Results of a lateral profile
        # Wind components
        plt.figure(0, dpi=dpi)
        ret, lgd = wt.plots.plot_winddata(mean_mag,u_mean,v_mean,y,yerr=u_v_err,lat=True)
        plt.tight_layout()
        plt.savefig(plot_path + 'wind_data_' + name + '.' + file_type,
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        
        # Turbulence intensity of the first component
        plt.figure(1, dpi=dpi)
        if (z_axis_lim[0] != None) and (z_axis_lim[1] != None):
            plt.ylim(z_axis_lim)
        wt.plots.plot_turb_int(I_u,y,yerr=I_u_err,lat=True,new_ref=new_VDI_ref, cut = shown_z_value)
        plt.tight_layout()
        plt.savefig(plot_path + 'I_u_' + name + '.' + file_type)
    
        # Turbulence intensity of the second component
        plt.figure(2, dpi=dpi)
        if (z_axis_lim[0] != None) and (z_axis_lim[1] != None):
            plt.ylim(z_axis_lim)
        wt.plots.plot_turb_int(I_v,y,yerr=I_v_err,component='I_w',lat=True,new_ref=new_VDI_ref,cut = shown_z_value)
        plt.tight_layout()
        plt.savefig(plot_path + 'I_v_' + name + '.' + file_type)
    
        # Profile of the fluxes
        plt.figure(3, dpi=dpi) 
        wt.plots.plot_fluxes(fluxes,y,yerr=flux_err,component='v',lat=True)
        plt.tight_layout()
        plt.savefig(plot_path + 'fluxes_' + name + '.' + file_type)
        
        # Profiles of the fluxes, logarithmic y-axis
        plt.figure(5, dpi=dpi)
        wt.plots.plot_fluxes_log(fluxes,heights,yerr=flux_err,component='v')
        plt.tight_layout()
        plt.savefig(plot_path + 'fluxes_log_' + name + '.' + file_type)
    
        # Lateral profile of Lux data
        plt.figure(4, dpi=dpi)
        wt.plots.plot_lux(lux,y,yerr=lux_err,component='w',lat=True,new_ref=new_VDI_ref)
        plt.tight_layout()
        plt.savefig(plot_path + 'Lux_' + name + '.' + file_type)
        
    if mode == 5:
        # Results of a lateral profile
        # Wind components
        plt.figure(0, dpi=dpi)
        ret, lgd = wt.plots.plot_winddata(mean_mag,u_mean,v_mean,y,yerr=u_v_err,lat=True)
        plt.tight_layout()
        plt.savefig(plot_path + 'wind_data_' + name + '.' + file_type,
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        
        # Turbulence intensity of the first component
        plt.figure(1, dpi=dpi)
        if (z_axis_lim[0] != None) and (z_axis_lim[1] != None):
            plt.ylim(z_axis_lim)
        wt.plots.plot_turb_int(I_u,y,yerr=I_u_err,lat=True,new_ref=new_VDI_ref, cut = shown_z_value)
        plt.tight_layout()
        plt.savefig(plot_path + 'I_u_' + name + '.' + file_type)
    
        # Turbulence intensity of the second component
        plt.figure(2, dpi=dpi)
        if (z_axis_lim[0] != None) and (z_axis_lim[1] != None):
            plt.ylim(z_axis_lim)
        wt.plots.plot_turb_int(I_v,y,yerr=I_v_err,component='I_w',lat=True,new_ref=new_VDI_ref,cut = shown_z_value)
        plt.tight_layout()
        plt.savefig(plot_path + 'I_v_' + name + '.' + file_type)
    
        # Profile of the fluxes
        plt.figure(3, dpi=dpi) 
        wt.plots.plot_fluxes(fluxes,y,yerr=flux_err,component='v',lat=True)
        plt.tight_layout()
        plt.savefig(plot_path + 'fluxes_' + name + '.' + file_type)
        
        # Profiles of the fluxes, logarithmic y-axis
        plt.figure(5, dpi=dpi)
        wt.plots.plot_fluxes_log(fluxes,heights,yerr=flux_err,component='v')
        plt.tight_layout()
        plt.savefig(plot_path + 'fluxes_log_' + name + '.' + file_type)
    
        # Lateral profile of Lux data
        plt.figure(4,dpi=dpi)
        wt.plots.plot_lux(lux,y,yerr=lux_err,component='w',lat=True,new_ref=new_VDI_ref)
        plt.tight_layout()
        plt.savefig(plot_path + 'Lux_' + name + '.' + file_type)    


# %%
