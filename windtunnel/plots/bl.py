#! /usr/bin/python3
# -*- coding: utf-8 -*-
""" Plotting tools for boundary layer assessment. """
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy import signal
import windtunnel as wt

plt.rcParams["figure.figsize"] = (9,6)
plt.rcParams.update({'font.size': 15})

__all__ = [
    'plot_scatter',
    'plot_hist',
    'plot_turb_int',
    'plot_fluxes',
    'plot_fluxes_log',
    'plot_winddata',
    'plot_winddata_log',
    'plot_lux',
    'plot_spectra',
    'plot_Re_independence',
    'plot_repeat',
    'turb_refernce_plot',
    'plot_convergence_test',
    'plot_convergence',
    'plot_JTFA_STFT',
    'plot_stdevs',
    'plot_perturbation_rose',
    'plot_arrival_law',
    'plot_transit_time_distribution',
    'plot_wavelet_transform',
]

def plot_wrapper(x, y, lat=False, ax=None, **kwargs):
    """ Plot helper function to switch abscissa and ordinate.

    Parameters
    ----------
    

    x: array like
    y: array like
    lat: boolean
    ax: axes object
    kwargs : arbitrary

    Returns
    ----------
    

    ret: axes object

    """
    if ax is None:
        ax = plt.gca()
        
    if lat:
        abscissa, ordinate = (ax.yaxis, ax.xaxis)
        xdata, ydata = y, x
    else:
        abscissa, ordinate = (ax.xaxis, ax.yaxis)
        xdata, ydata = x, y
        
    ret = ax.plot(xdata, ydata, **kwargs)
    abscissa.set_label_text('x-data')
    ordinate.set_label_text('y-data')
    
    return ret

def plot_scatter(x,y,std_mask=5.,ax=None,**kwargs):
    """Creates a scatter plot of x and y. All outliers outside of 5 STDs of the
    components mean value are coloured in orange.

    Parameters
    ----------
    

    x: array like
    y: array like
    std_mask: float
    ax: axes object
    kwargs : arbitrary

    Returns
    ----------
    

    ret: axes object
    """
    # Get current axis
    if ax is None:
       ax = plt.gca()
       
    # Find outliers
    u_mask = x<(std_mask*np.std(x)+np.mean(x))
    v_mask = y<(std_mask*np.std(y)+np.mean(y))
    mask = np.logical_and(u_mask, v_mask)

    x_clean = x[mask]
    y_clean = y[mask]
    
    x_outliers = x[~mask]
    y_outliers = y[~mask]
    # Plot
    ret = ax.scatter(x_clean,y_clean, **kwargs)
    ax.scatter(x_outliers,y_outliers, color='orange', **kwargs)
    ax.set_ylabel(r'$w$ (ms$^{-1}$)')
    ax.set_xlabel(r'$u$ (ms$^{-1}$)')
    ax.grid()
    
    return ret

def plot_scatter_wght(transit_time,x,y,std_mask=5.,ax=None,**kwargs):
    """Creates a scatter plot of x and y using time transit time weighted 
    statistics. All outliers outside of 5 STDs of the components mean value are
    coloured in orange, as default.

    Parameters
    ----------
    

    transit_time: array like
    x: array like
    y: array like
    std_mask: float
    ax: axes object
    kwargs : arbitrary

    Returns
    ----------
    

    ret: axes object
    """
    # Get current axis
    if ax is None:
       ax = plt.gca()
       
    # Find outliers
    x_mask = x<(std_mask*(np.sqrt(
                          wt.transit_time_weighted_var(transit_time,x))+
                          wt.transit_time_weighted_mean(transit_time,x)))
    y_mask = y<(std_mask*(np.sqrt(
                          wt.transit_time_weighted_var(transit_time,y))+
                          wt.transit_time_weighted_mean(transit_time,y)))

    mask = np.logical_and(x_mask, y_mask)

    x_clean = x[mask]
    y_clean = y[mask]
    
    x_outliers = x[~mask]
    y_outliers = y[~mask]
    
    # Plot
    ret = ax.scatter(x_clean,y_clean, **kwargs)
    ax.scatter(x_outliers,y_outliers, color='orange', **kwargs)
    ax.set_ylabel(r'$w$ (ms$^{-1}$)')
    ax.set_xlabel(r'$u$ (ms$^{-1}$)')
    ax.grid()
    
    return ret
    
def plot_hist(data,ax=None,**kwargs):
    """Creates a histogram for data.

    Parameters
    ----------
    

    data: array like
    ax: axis object
    
    Returns
    ----------
    

    ret: 
    """
    
    # Get current axis
    if ax is None:
       ax = plt.gca()
       
    #  Calculate bin size and normalise count
    data=np.asarray(data)
    count,bins = np.histogram(data[~np.isnan(data)],
                              bins=np.linspace(np.nanmin(data),np.nanmax(data),
                              np.max([np.nanmin([25,(int(np.nanmax(data)-
                                      np.nanmin(data))+1)*5]),15])))    
    
    count = (count/np.size(data))*100.
    
    # Plots bar-plot using np.histogram data
    ret = ax.bar(bins[:-1],count,width = np.nanmean(np.diff(bins)), 
            color='cornflowerblue')
    ticks=bins[:-1]+0.5*np.nanmean(np.diff(bins))
    ax.set_xticks(ticks.round(2))
    for tick in ax.get_xticklabels():
        tick.set_rotation(55)
    ax.set_xlim([ticks.min()-0.5*np.nanmean(np.diff(bins)),
              ticks.max()+0.5*np.nanmean(np.diff(bins))])
    ax.set_ylabel('relative Frequency (%)')
    ax.grid('on')
    
    return ret

def plot_turb_int(data,heights,yerr=0,component='I_u',var_lat='Y',lat=False,
                  ref_path=None, ax=None, new_ref=True,manuel_VDI=False,values_VDI={}, yAchse=None,xAchse=None,
                    cut = (None, None), shown_Legend = True, Labels=None,xLabel=None,yLabel=None, LegendSize=1.0,dataColor="darkred",dataSize=5.0,
                      **kwargs):
    """ Plots turbulence intensities from data with VDI reference data for 
    their respective height. yerr specifies the uncertainty. Its default value
    is 0. If lat is True then a lateral profile is created.

    Parameters
    ----------
    

    data: array like
    heights: array like
    yerr: float
    component: string
    var_lat: string, integer, float
    lat: boolean
    ref_path: string
    ax: axes object
    kwargs : arbitrary

    Returns
    ----------
    

    ret: axes object
    """

    if Labels ==None:
        Labels=['VDI slightly rough (lower bound)',
                'VDI moderately rough (lower bound)',
                'VDI rough (lower bound)',
                'VDI very rough (lower bound)',
                f'turbulence intensity {component}'
                ]
        
    if ax is None:
       ax = plt.gca()

    data = np.asarray(data)
    heights = np.asarray(heights)
    if (cut[0] or cut[1]) != None:
        if cut[0] != None: 
            mask_lower = np.where(heights <= cut[0], np.nan, True)
            data = data * mask_lower
        if cut[1] != None: 
            mask_higher = np.where(heights >= cut[1], np.nan, True)
            data = data * mask_higher

    if lat == False:
        if new_ref==True:
            z,I_u,I_v,I_w = wt.get_new_turb_reference_values(manuel_VDI=manuel_VDI,values_VDI=values_VDI)
        
            if component=='I_u':
                slight = np.vstack([z,I_u[0]])
                moderate = np.vstack([z,I_u[1]])
                rough = np.vstack([z, I_u[2]])
                very_rough = np.vstack([z, I_u[3]])
            elif component == 'I_v':
                slight = np.vstack([z,I_v[0]])
                moderate = np.vstack([z,I_v[1]])
                rough = np.vstack([z, I_v[2]])
                very_rough = np.vstack([z, I_v[3]])
            elif component == 'I_w':
                slight = np.vstack([z,I_w[0]])
                moderate = np.vstack([z,I_w[1]])
                rough = np.vstack([z, I_w[2]])
                very_rough = np.vstack([z, I_w[3]])
        
        if new_ref==False:
            slight, moderate, rough, very_rough =\
                wt.get_turb_reference_values(component)

        s = ax.plot(slight[1, :], slight[0, :],
                     '-k', linewidth=0.5,label=Labels[0])
        m = ax.plot(moderate[1, :], moderate[0, :],
                     '--k', linewidth=0.5,label=Labels[1])
        r = ax.plot(rough[1, :], rough[0, :],
                     '-.k', linewidth=0.5, label=Labels[2])
        vr = ax.plot(very_rough[1, :], very_rough[0, :]
                     , ':k', linewidth=0.5,label=Labels[3])

    ret = []
    if lat == False:
        l = ax.errorbar(data,heights,xerr=np.ones_like(data) * yerr,
                        fmt='o',color=dataColor,ms=dataSize, #dodgerblue
                        ecolor="black",capsize=3.0,
                            label=Labels[4], linewidth=0.3,**kwargs)


    else:
        l = ax.errorbar(heights,data,xerr=np.ones_like(data) * yerr,
                        fmt='o', color=dataColor,ms=dataSize,
                        ecolor="black",capsize=3.0,
                          label=Labels[4], linewidth=0.3, **kwargs)
            

            
    ret.append(l)
        
    ax.grid(True)
    if lat == False:
        if shown_Legend ==True:
            ax.legend(loc="upper right",
                   fontsize=LegendSize)
        ax.set_xlabel(xLabel if xLabel is not None else r'turbulence intensity ' + component + ' (-)') #fontsize=11
        ax.set_ylabel(yLabel if yLabel is not None else 'z full-scale (m)')
        ax.set_xlim(xAchse if xAchse is not None else (0.0,0.30))
        ax.set_ylim(yAchse if yAchse is not None else (0.0,100.0))
    else:
        if shown_Legend ==True:
            ax.legend(loc="upper right",numpoints=1,fontsize=LegendSize)
        ax.set_ylabel(xLabel if xLabel is not None else var_lat+' full-scale (m)')
        ax.set_xlabel(yLabel if yLabel is not None else r'turbulence intensity ' + component + ' (-)') #fontsize=11
        ax.set_xlim(xAchse if xAchse is not None else (0.0,0.30))
        ax.set_ylim(yAchse if xAchse is not None else (0.0,100.0))
    

    return ret

def plot_fluxes(data, heights, yerr=0, component='v', var_lat='Y', lat=False,
                xLabel=None,yLabel=None,xAchse=None,yAchse=None,Labels=None,showLegend=False, cut=(None,None),cut2=(None,None),dataColor="darkred",dataSize=1.0,
                ax=None, sfc_height=60., **kwargs):
    """ Plots fluxes from data for their respective height with a 10% range of
    the low point mean. yerr specifies the uncertainty. Its default value is 0.
    WARNING: Data must be made dimensionless before plotting! If lat is True 
    then a lateral profile is created.
    
    Parameters
    ----------
    
    data: list or np.array
    heights: list or np.array
    yerr: float
    component: string
    var_lat: boolean
    lat: boolean
    ax: axis passed to function
    sfc_height: float

    Returns
    ----------
    

    ret: list
    """

    if ax is None:
        ax = plt.gca()

    data = np.asarray(data)
    heights = np.asarray(heights)
    data_full = data
        # Choose if values should be colored differently
    if (cut[0] or cut[1]) != None:
        if cut[0] != None:
            data_inside = data * np.where(heights <= cut[0], np.nan, True)
            data_outside_lower = data * np.where(heights >= cut[0], np.nan, True)
            heights_outside_lower = heights * np.where(heights >= cut[0], np.nan, True)
        if cut[1] != None:
            data_inside = data * np.where(heights >= cut[1], np.nan, True)
            data_outside_upper = data * np.where(heights <= cut[1], np.nan, True)
            heights_outside_upper = heights * np.where(heights <= cut[1], np.nan, True)
        data = data_inside

    # Handle cut2 parameter - completely mask out data outside cut2 range
    if (cut2[0] or cut2[1]) != None:
        if cut2[0] != None:
            # Mask out data below cut2[0]
            data = data * np.where(heights >= cut2[0], True, np.nan)
            if 'data_outside_lower' in locals():
                data_outside_lower = data_outside_lower * np.where(heights >= cut2[0], True, np.nan)
                heights_outside_lower = heights_outside_lower * np.where(heights >= cut2[0], True, np.nan)
            if 'data_outside_upper' in locals():
                data_outside_upper = data_outside_upper * np.where(heights >= cut2[0], True, np.nan)
                heights_outside_upper = heights_outside_upper * np.where(heights >= cut2[0], True, np.nan)
        if cut2[1] != None:
            # Mask out data above cut2[1]
            data = data * np.where(heights <= cut2[1], True, np.nan)
            if 'data_outside_lower' in locals():
                data_outside_lower = data_outside_lower * np.where(heights <= cut2[1], True, np.nan)
                heights_outside_lower = heights_outside_lower * np.where(heights <= cut2[1], True, np.nan)
            if 'data_outside_upper' in locals():
                data_outside_upper = data_outside_upper * np.where(heights <= cut2[1], True, np.nan)
                heights_outside_upper = heights_outside_upper * np.where(heights <= cut2[1], True, np.nan)

    ret = []
    for flux, height in zip(data, heights):
        if lat == False:
            l = ax.errorbar(flux,height,xerr=yerr,fmt='o',color=dataColor,ms=dataSize,
            ecolor="black",capsize=3.0,
            **kwargs)
            labels= [r'wind tunnel flux']
        else:
            l = ax.errorbar(height,flux,xerr=yerr,fmt='o',color=dataColor,ms=dataSize, #dodgerblue
            ecolor="black",
            label=r'wind tunnel flux', **kwargs)
            labels= [r'wind tunnel flux']
        ret.append(l)

    if (cut[0]!=None and cut[1]!=None):
        if lat == False:
            for flux, height in zip(data_outside_lower, heights_outside_lower):
                l = ax.errorbar(flux,height,xerr=yerr,fmt='o',color='lightgrey',
                ecolor="black",capsize=3.0,ms=dataSize,
                **kwargs)
                ret.append(l)
            for flux, height in zip(data_outside_upper, heights_outside_upper):
                ax.errorbar(flux,height,xerr=yerr,fmt='o',color='lightgrey',
                ecolor="black",capsize=3.0,ms=dataSize,
                **kwargs)
                ret.append(l)
                

        
    ax.grid(True)
    if lat == False:
        sfc_layer = np.where(heights<sfc_height)
        xcen = np.mean(data_inside[sfc_layer])
        xrange = np.abs(0.1*xcen)
        ax.axvspan(xcen-xrange,xcen+xrange,facecolor='grey',
                   edgecolor='none', alpha=0.2,
                   label='10% range of low point mean')
        if(showLegend==True):
            if(Labels!=None):
                ax.legend([l],Labels,loc='best',fontsize=16)
            ax.legend([l],labels,loc='best',fontsize=16)
        ax.set_xlabel(xLabel if xLabel is not None else r'u'+ '\'' +component+'\'$\cdot U_{0}^{-2}\ (-)$')
        ax.set_ylabel(yLabel if yLabel is not None else 'z full-scale (m)') #fontsize=11
        ax.set_xlim(xAchse if xAchse is not None else (0.0,70.0))
        ax.set_ylim(yAchse if xAchse is not None else (0.0,0.050))

        """
        if np.nanmax(data)<0:
            ax.set_xlim([np.nanmin(data) * 1.1, 0])
        else:
            ax.set_xlim([np.nanmin(data) * 1.1, np.nanmax(data)*1.1])
        """

    else:
        if(showLegend==True):
            ax.legend([l],labels,loc='best',fontsize=16)
        ax.set_xlabel(xLabel if xLabel is not None else r'u' + '\'' + component + '\' $\cdot u_{ref}^{-2}$ $(-)$')
        ax.set_ylabel(yLabel if yLabel is not None else var_lat+' full-scale (m)') #fontsize=11
        ax.set_xlim(xAchse if xAchse is not None else (0.0,70.0))
        ax.set_ylim(yAchse if xAchse is not None else (0.0,0.050))

    
 
    return ret

def plot_fluxes_log(data, heights, yerr=0, component='v',
                    xLabel=None,yLabel=None,showLegend=False,
                    ax=None, sfc_height=60., **kwargs):
    """ Plots fluxes from data for their respective height on a log scale with
    a 10% range of the low point mean. yerr specifies the uncertainty. Its 
    default value is 0. WARNING: Data must be made dimensionless before 
    plotting!
    
    Parameters
    ----------
    

    data: list or np.array
    heights: list or np.array
    yerr: float
    component: string
    ax: axis passed to function
    sfc_height: float

    Returns
    ----------
    

    ret: list
    """

    if ax is None:
       ax = plt.gca()

    data = np.asarray(data)
    heights = np.asarray(heights)
    
    ret = []
    for flux, height in zip(data, heights):
        l = ax.errorbar(flux,height,xerr=yerr,fmt='o',color='red',
                        ecolor="black",capsize=3.0,
                        **kwargs)
        
        labels= [r'wind tunnel flux']
        
        ret.append(l)
    # xlim is user-defined
    # plt.xlim(-0.0025,0.)
    plt.yscale('log')
    ax.grid(True,'both','both')

    #Fill surface layer region
    sfc_layer = np.where(heights<sfc_height)
    xcen = np.mean(data[sfc_layer])
    xrange = np.abs(0.1*xcen)
    ax.axvspan(xcen-xrange,xcen+xrange,facecolor='grey', #'lightskyblue'
                edgecolor='none', alpha=0.2,
                label='10% range of low point mean')
    if(showLegend==True):
        ax.legend([l],labels,loc='best',fontsize=16, numpoints=1)
    ax.set_xlabel(xLabel if xLabel is not None else r'u' + '\'' + component + '\' $\cdot u_{ref}^{-2}$ $(-)$')
    ax.set_ylabel(yLabel if yLabel is not None else 'r $Z_{fs}$ [m]') #fontsize=11
    
 
    ax.set_ylim(4.,100.)
    if np.nanmax(data) < 0:
        ax.set_xlim([np.nanmin(data) * 1.1, 0])
    else:
        ax.set_xlim([np.nanmin(data) * 1.1, np.nanmax(data) * 1.1])
    return ret

def plot_winddata(heights,components=["u"],mean_magnitude=None, u_mean=None, v_mean=None, yerr=0, var_lat='Y', lat=False,
                  xLabel=None,yLabel=None,yAchse=None,xAchse=None,Labels=None,showLegend=True,
                  ax=None, **kwargs):

    
    """ Plots wind components and wind magnitude for their respective height.
    yerr specifies the uncertainty. Its default value is 0. If lat is True then
    a lateral profile is created.
    
    Parameters
    ----------
    

    mean_magnitude: array like
    u_mean: array like
    v_mean: array like
    heights: array like        
    yerr: float
    var_lat: string, integer, float 
    lat: boolean
    ax: axes object
    kwargs : arbitrary

    Returns
    ----------
    

    ret: list of axes objects
    lgd: axes object
    """
    if ax is None:
       ax = plt.gca()

    mean_magnitude = np.asarray(mean_magnitude)
    u_mean = np.asarray(u_mean)
    v_mean = np.asarray(v_mean)
    heights = np.asarray(heights)
    
    ret = []  
    for i in range(np.size(mean_magnitude)):
        #print("gets here")
        if lat == False:

            if Labels==None:
                 Labels = ['Magnitude','U-component',r'$2^{nd}-component$']

            
            if "u" in components:
                U = ax.errorbar(u_mean[i],heights[i],xerr=yerr,marker='o',
                             color='darkred',label=Labels[0],
                             ecolor="black",capsize=3.0,
                             **kwargs)
                ret.append(U)
            if "v" in components:
                V = ax.errorbar(v_mean[i],heights[i],yerr=yerr,marker='^',
                            color='darkred',label=Labels[1])
                ret.append(V)
            if "mag" in components:
                M = ax.errorbar(mean_magnitude[i],heights[i],yerr=yerr,marker='s',
                            color='darkred',label=Labels[2])
                ret.append(M)
            
            
            ax.grid(True)
            #lgd = ax.legend(Labels,bbox_to_anchor=(0.5,1.05),
            #          loc='lower center',borderaxespad=0.,ncol=3,fontsize=16)
            #ax.set_xlabel(r'velocity $(-)$')
            #ax.set_ylabel('z full-scale (m)')
            
            ax.set_xlabel(xLabel if xLabel is not None else r'velocity $(-)$')
            ax.set_ylabel(yLabel if yLabel is not None else 'z full-scale (m)') #fontsize=11
            #ret.append(M + U + V)

        
        else:
            i=0#?
            if Labels==None:
                 Labels = ['Magnitude','U-component',r'$2^{nd}-component$']

            
            if "u" in components:
                U = ax.errorbar(u_mean[i],heights[i],yerr=yerr,marker='o',
                             color='darkred',label=Labels[0])
                ret.append(U)
            if "v" in components:
                V = ax.errorbar(v_mean[i],heights[i],yerr=yerr,marker='^',
                            color='darkred',label=Labels[1])
                ret.append(V)
            if "mag" in components:
                M = ax.errorbar(mean_magnitude[i],heights[i],yerr=yerr,marker='s',
                            color='darkred',label=Labels[2])
                ret.append(M)

            """
            M = ax.errorbar(heights[i],mean_magnitude[i],yerr=yerr,marker='s',
                             color='aqua',label='Magnitude')
            U = ax.errorbar(heights[i],u_mean[i],yerr=yerr,marker='o',
                             color='navy',label='U-component')
            V = ax.errorbar(heights[i],v_mean[i],yerr=yerr,marker='^',
                             color='dodgerblue')
            
            labels = ['Magnitude','U-component',r'$2^{nd}-component$']
            """

            ax.grid(True)
            if(showLegend==True):
                ax.legend(loc="upper right")#,bbox_to_anchor=(0.5,1.05), lgd=
                      #loc='lower center',borderaxespad=0.,ncol=3,fontsize=16)
            ax.set_xlabel(xLabel if xLabel is not None else var_lat+' full-scale (m)')
            ax.set_ylabel(yLabel if yLabel is not None else r'velocity $(-)$') #fontsize=11
            ax.set_ylim(-0.1,0.7)
            #ret.append(M + U + V)

    if(showLegend==True):
        ax.legend(loc='upper right')#,fontsize=16, numpoints=1)

    #print(lgd)
    return ret#, lgd

def plot_winddata_log(mean_magnitude,u_mean,v_mean,heights,yerr=0,ax=None,
                      **kwargs):
    """Plots wind components and wind magnitude for their respective height on
    a log scale. yerr specifies the uncertainty. Its default value is 0.
    
    Parameters
    ----------
    
    mean_magnitude: array like
    u_mean: array like
    v_mean: array like
    heights: array like        
    yerr: float
    ax: axes object
    kwargs : arbitrary

    Returns
    ----------
    

    ret: list of axes objects
    lgd: axes object

    """
    if ax is None:
       ax = plt.gca()
    
    ret = []
    for i in range(np.size(mean_magnitude)):
        M = ax.errorbar(mean_magnitude[i],heights[i],yerr=yerr,fmt='s',
                         color='aqua')
        U = ax.errorbar(u_mean[i],heights[i],yerr=yerr,fmt='o',color='navy'),
        V = ax.errorbar(v_mean[i],heights[i],yerr=yerr,fmt='^',
                        color='dodgerblue')
        ret.append(M + U + V)
        
    labels = ['Magnitude','U-component',r'$2^{nd}-component$']
    
    plt.yscale('log')
    ax.grid(True,'both','both')
    lgd = ax.legend([M,U,V],labels,bbox_to_anchor=(0.5,1.05),loc='lower center',
              borderaxespad=0.,ncol=3,fontsize=16)
    ax.set_xlabel(r'wind magnitude $(-)$')
    ax.set_ylabel('z full-scale (m)')
    
    return ret, lgd

def plot_lux(Lux, heights, err=None, var_lat='Y', lat=False, ref_path=None,manuel_VDI=False,values_VDI={},
              ax=None,xLabel=None,yLabel=None,Labels=None,showLegend=True,xAchse=None,yAchse=None,dataColor="darkred",dataSize=5.0,
             new_ref=True, **kwargs):
    """Plots Lux data on a double logarithmic scale with reference data. yerr
    specifies the uncertainty. Its default value is 0. If lat
    is True then a lateral profile, without a loglog scale, is created.

    Parameters
    ----------
    

    Lux: array like
    heights: array like        
    err: float
    var_lat: string, integer, float 
    lat: boolean
    ref_path: string
    ax: axes object
    kwargs : arbitrary

    Returns
    ----------
    

    ret: list of axes objects
    """
	#edit 08/02/2019: moved labels to ax.plot, to ensure proper plotting of legend, and removing need for extra labels variable.
    if ax is None:
       ax = plt.gca()
       #ax.figure.set_size_inches(figSize[0], figSize[1])
    
    if Labels == None:
        Labels = ['wind tunnel',r'$z_0=10\ m$ (theory)',r'$z_0=1\ m$ (theory)',r'$z_0=0.1\ m$ (theory)',
                r'$z_0=0.01\ m$ (theory)',
                'observations smooth surface','observations rough surface'
                ]

    if lat == False:
        if new_ref==False:
            Lux_10,Lux_1,Lux_01,Lux_001, Lux_obs_smooth,Lux_obs_rough = \
            wt.get_lux_referencedata(ref_path)
        else:
            Lux_10,Lux_1,Lux_01,Lux_001 = wt.get_new_lux_referencedata(manuel_VDI=manuel_VDI,values_VDI=values_VDI)

    ret = []
    if lat == False:
        Lux = ax.errorbar(Lux,heights,xerr=err,fmt='o',color=dataColor,label=Labels[0],
                          ecolor="black", capsize=3.0, capthick=1.5, elinewidth=1.0,ms=dataSize)
        
        ref1 = ax.plot(Lux_10[1,:],Lux_10[0,:],'k-',linewidth=1,label=Labels[1])
        ref2 = ax.plot(Lux_1[1,:],Lux_1[0,:],'k--',linewidth=1,label=Labels[2])
        ref3 = ax.plot(Lux_01[1,:],Lux_01[0,:],'k-.',linewidth=1,label=Labels[3])
        ref4 = ax.plot(Lux_001[1,:],Lux_001[0,:],'k:',linewidth=1,label=Labels[4])
        
        if new_ref==False:
            ref5 = ax.plot(Lux_obs_smooth[1,:],Lux_obs_smooth[0,:],'k+',linewidth=1,label=Labels[5],color="lightgrey")
            ref6 = ax.plot(Lux_obs_rough[1,:],Lux_obs_rough[0,:],'kx',linewidth=1,label=Labels[6],color="grey")

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)#,'both')
        #ax.grid(True,'both','both')
    
        if(showLegend==True):
            ax.legend(loc='upper left')#,fontsize=1, numpoints=1)
        ax.set_xlabel(xLabel if xLabel is not None else r'$L_{u}^{x}$ full-scale (m)')
        ax.set_ylabel(yLabel if yLabel is not None else r'$z$ full-scale (m)') #fontsize=11
        ax.set_yticks([50, 100, 150, 200, 250, 300])
        ax.set_yticklabels(['50', '100','150', '200', '250', '300'])
        ax.set_xlim(xAchse if xAchse is not None else [10,1000])
        ax.set_ylim(yAchse if yAchse is not None else [min(heights),300]) #fontsize=11
        
    else:
        Lux = ax.errorbar(heights,Lux,yerr=err,fmt='o',color=dataColor)
        labels = ['wind tunnel']
        ax.grid(True)
        if(showLegend==True):
            ax.legend([Lux],labels,bbox_to_anchor=(0.5,1.05),loc='upper left',
                  borderaxespad=0.,ncol=2,fontsize=16)
        ax.set_xlabel(xLabel if xLabel is not None else r'$L_{u}^{x}$ full-scale (m)')
        ax.set_ylabel(yLabel if yLabel is not None else r'$z$ full-scale (m)') #fontsize=11 
        ax.set_yticks([50, 100, 150, 200, 250, 300])
        ax.set_yticklabels(['50', '100','150', '200', '250', '300'])
        ax.set_xlim(xAchse if xAchse is not None else [10,1000])
        ax.set_ylim(yAchse if yAchse is not None else [min(heights),300]) #fontsize=11 
        
    return ret

def plot_spectra(f_sm, S_uu_sm, S_vv_sm, u_aliasing, v_aliasing, 
                 wind_comps, height, ref_path=None, 
                 xLabel=None,yLabel=None,Labels=None,showLegend=True,LegendSize=1.0, xAchse=None,yAchse=None,dataColor=["darkred","darkgreen"],dataSize=3.0,
                 refRangeVis="Interval", comps2Plot=["u"],
                 ax=None, **kwargs):
    """Plots spectra using INPUT with reference data.

    Parameters
    ----------
    

    f_sm: array like
    S_uu_sm: array like        
    S_vv_sm: array like
    S_vv_sm: array like
    u_aliasing: integer
    v_aliasing: integer
    uv_aliasing: integer
    wind_comps: list
    height: float
    ref_path: string
    ax: axes object
    kwargs : arbitrary

    Returns
    ----------
    

    h1: axes object
    h2: axes object
    h3: axes object
    h4: axes object    
    
    """

    if Labels==None:
        Labels = []
        Labels.append("Windkanalmessung")
        for u in comps2Plot:
            Labels.append(f"max {u}")
            Labels.append(f"min {u}")

    if str(type(dataColor))=="<class 'str'>":
        dataColor = [dataColor]

    if ax is None:
        ax = plt.gca()
    
    #Create achses range for plotting
    xsmin = np.nanmin(np.nanmin(f_sm[np.where(f_sm>0)]))
    xsmax = np.nanmax(np.nanmax(f_sm[np.where(f_sm>0)]))
    # xsmin = np.nanmin(10**-4,np.nanmin(f_sm[np.where(f_sm>0)]))
    # xsmax = np.nanmax(100,np.nanmax(f_sm[np.where(f_sm>0)]))
    ref_x = np.logspace(np.log10(xsmin),np.log10(xsmax),50)
    #ref_specs = wt.get_reference_spectra(height,ref_path)


    hs=[]
    if 'u' in wind_comps and 'u' in comps2Plot:
        h1 = ax.loglog(f_sm[:u_aliasing],S_uu_sm[:u_aliasing],"o",color=dataColor[0],ms=dataSize)#,
               #label=r'Windkanalmessung $'+'{0}{0}'.format(wind_comps[0])+'$')
        h2 = ax.loglog(f_sm[u_aliasing:],S_uu_sm[u_aliasing:],"o", color=dataColor[0],ms=dataSize,
               fillstyle='none')
        hs.append(h1)
        hs.append(h2)
    
    if 'v' in wind_comps and 'v' in comps2Plot:
        h3 = ax.loglog(f_sm[:v_aliasing],S_vv_sm[:v_aliasing],'o',color=dataColor[1],ms=dataSize)#,
              #label='wind tunnel $'+'{0}{0}'.format(wind_comps[1])+'$')
        h4 = ax.loglog(f_sm[v_aliasing:],S_vv_sm[v_aliasing:],'bs',color=dataColor[1],ms=dataSize,
              fillstyle='none')
        hs.append(h3)
        hs.append(h4)

    #if 'w' in wind_comps and 'w' in comps2Plot:
    #    h3 = ax.loglog(f_sm[:w_aliasing],S_ww_sm[:w_aliasing],'bs',markersize=3,
    #          label='wind tunnel $'+'{0}{0}'.format(wind_comps[1])+'$')
    #    h4 = ax.loglog(f_sm[w_aliasing:],S_ww_sm[w_aliasing:],'bs',markersize=3,
    #          fillstyle='none')
    #    hs.append(h5)
    #    hs.append(h6)
     
    #Choose interval visualisation
    if refRangeVis=="Interval":
        if 'u' in wind_comps and 'u' in comps2Plot:
            ax.fill_between(ref_x,wt.calc_ref_spectra(ref_x)[0],wt.calc_ref_spectra(ref_x)[1],
                        facecolor=(0.6,0.6,0.6),edgecolor='none',alpha=0.2,
                        label=r'reference range $uu$')

        if 'v' in wind_comps and 'v' in comps2Plot:
            ax.fill_between(ref_x,wt.calc_ref_spectra(ref_x)[0],wt.calc_ref_spectra(ref_x)[1],
                            facecolor=(0.8,0.6,0.6),edgecolor='none',alpha=0.2,
                            label=r'reference range $vv$')

        if 'w' in wind_comps and 'w' in comps2Plot:
            ax.fill_between(ref_x,wt.calc_ref_spectra(ref_x)[0],wt.calc_ref_spectra(ref_x)[1],
                            facecolor=(0.6,0.6,0.6),edgecolor='none',alpha=0.2,
                            label=r'reference range $ww$')
    
    elif refRangeVis=="maxMinLines":
        if 'u' in wind_comps and 'u' in comps2Plot:
            ax.plot(ref_x,wt.calc_ref_spectra(ref_x)[0],"-", color="grey")
            ax.plot(ref_x,wt.calc_ref_spectra(ref_x)[1],"--", color="grey")
        if 'v' in wind_comps and 'v' in comps2Plot:
            ax.plot(ref_x,wt.calc_ref_spectra(ref_x)[0],"-", color="grey")
            ax.plot(ref_x,wt.calc_ref_spectra(ref_x)[1],"--", color="grey")
        if 'w' in wind_comps and 'w' in comps2Plot:
            ax.plot(ref_x,wt.calc_ref_spectra(ref_x)[0],"-", color="grey")
            ax.plot(ref_x,wt.calc_ref_spectra(ref_x)[1],"--", color="grey")

    # ax.set_xlim(xsmin,xsmax)
    #ax.set_xlim(10**-3.,2.*10.**2.)
    #ax.set_ylim([10**-6,10])
    #ax.set_xlabel("test x")
    #ax.set_ylabel("test y")
    ax.set_xlim(xAchse if xAchse is not None else (10**-3.,2.*10.**2.))
    ax.set_ylim(yAchse if yAchse is not None else (10**-6,10))
    ax.set_xlabel(xLabel if xLabel is not None else r"$f\cdot z\cdot U^{-1}$")
    ax.set_ylabel(yLabel if yLabel is not None else r"$f\cdot S_{ij}\cdot (\sigma_i\sigma_j)^{-1}$")
    if(showLegend==True):
        ax.legend(Labels,loc='lower right',fontsize=LegendSize)
    ax.grid(True)
    
   
    return hs


def plot_spectra_nc(f_comp1_sm,f_comp2_sm, S_comp1_sm,S_comp2_sm,
                 comp1_aliasing,comp2_aliasing,wind_comps, height, ref_path=None, set_limits=True):
    """Plots spectra using INPUT with reference data.
    
    Parameters
    ----------
    

    f_comp1_sm: array like
    f_comp2_sm: array like
    S_comp1_sm: array like        
    S_comp2_sm: array like
    comp1_aliasing: integer
    comp2_aliasing: integer
    wind_comps: list
    height: float
    ref_path: string
    set_limits: boolean

    Returns
    ----------
    

    h1: axes object
    h2: axes object
    """


    f_sm = [f_comp1_sm,f_comp2_sm][np.argmin([np.nanmax(f_comp1_sm),np.nanmax(f_comp2_sm)])]

    xsmin = np.nanmin(f_sm[np.where(f_sm > 0)])
    xsmax = np.nanmax(f_sm[np.where(f_sm > 0)])

    S_comp1_sm = S_comp1_sm[:np.min((len(S_comp1_sm),len(S_comp2_sm)))]
    S_comp2_sm = S_comp2_sm[:np.min((len(S_comp1_sm), len(S_comp2_sm)))]
    #    xsmin = np.nanmin(10**-4,np.nanmin(f_sm[np.where(f_sm>0)]))
    #    xsmax = np.nanmax(100,np.nanmax(f_sm[np.where(f_sm>0)]))
    ref_x = np.logspace(np.log10(xsmin), np.log10(xsmax), 50)
    ref_specs = wt.get_reference_spectra(height, ref_path)




    #f_sm = f_sm[:len(S_uu_sm)]
    ax = plt.gca()
    h1 = ax.loglog(f_sm[:comp1_aliasing], S_comp1_sm[:comp1_aliasing], 'ro', markersize=3,
                   label=r'wind tunnel $'+'{0}{0}'.format(wind_comps[0])+'$')
    h2 = ax.loglog(f_sm[comp1_aliasing:], S_comp1_sm[comp1_aliasing:], 'ro', markersize=3,
                   fillstyle='none')
    h3 =  ax.loglog(f_sm[:comp2_aliasing], S_comp2_sm[:comp2_aliasing], 'bo', markersize=3,
                   label=r'wind tunnel $'+'{0}{0}'.format(wind_comps[1])+'$')
    h4 = ax.loglog(f_sm[comp2_aliasing:], S_comp2_sm[comp2_aliasing:], 'bo', markersize=3,
                   fillstyle='none')
    if  'u' in wind_comps:
        ax.fill_between(ref_x, wt.calc_ref_spectra(ref_x, *ref_specs[0, :]),
                        wt.calc_ref_spectra(ref_x, *ref_specs[1, :]),
                        facecolor=(1., 0.6, 0.6), edgecolor='none', alpha=0.2,
                        label=r'reference range $uu$')

    if  'v' in wind_comps:
        ax.fill_between(ref_x, wt.calc_ref_spectra(ref_x, *ref_specs[2, :]),
                        wt.calc_ref_spectra(ref_x, *ref_specs[3, :]),
                        facecolor=(0.6, 0.6, 1.), edgecolor='none', alpha=0.2,
                        label=r'reference range $vv$')

    if  'w' in wind_comps:
        ax.fill_between(ref_x, wt.calc_ref_spectra(ref_x, *ref_specs[4, :]),
                        wt.calc_ref_spectra(ref_x, *ref_specs[5, :]),
                        facecolor=(0.6, 0.6, 1.), edgecolor='none', alpha=0.2,
                        label=r'reference range $ww$')

    if set_limits:
        ax.set_xlim([10**-3,10**1])
    else:
        ax.set_xlim(xsmin,xsmax)
    ax.set_ylim([10 ** -6, 10])
    #ax.set_xlabel(r"$f\cdot z\cdot U^{-1}$")
    #ax.set_ylabel(r"$f\cdot S_{ij}\cdot (\sigma_i\sigma_j)^{-1}$")
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True)

    return h1, h2

def turb_refernce_plot(z, Iu, Iv, Iw):
    z0 = np.array([0.005, 0.1, 0.5, 2])
    components = ['Iu', 'Iv', 'Iw']

    for j, I in enumerate([Iu, Iv, Iw]):

        fig, ax = plt.subplots()

        for i, I_z0 in enumerate(I):
            ax.plot(I_z0, z, linewidth=0.5, ls='-', label='$z_{0}$ = ' + str(z0[i]))

        ax.legend()
        ax.set_xlabel('Turbulence Intensity ' + components[j])
        ax.set_ylabel('z (m) ')

def plot_Re_independence(data,wtref,ymin=None,ymax=None,yerr=0,ax=None,**kwargs):
    """ Plots the results for a Reynolds Number Independence test from a non-
    dimensionalised timeseries. yerr specifies the uncertainty. Its default 
    value is 0.
    
    Parameters
    ----------
    

    data: array like
    wtref: array like
    ymin: float
    ymax: float
    yerr: float
    ax: axes object
    kwargs: arbitrary

    Returns
    ----------
    

    ret: list
    """
    if ax is None:
        ax=plt.gca()
    if ymin is None:
       ymin=np.min(data)
    if ymax is None:
       ymax=np.max(data)
    # Sort wtref and data to correspond to increasing wtref values
    data = [wtref for _,wtref in sorted(zip(wtref,data))]
    wtref = sorted(wtref)
    
    # Plot
    ret = []
    for i,value in enumerate(data):
        l = ax.errorbar(wtref[i],value,yerr=yerr,fmt='o',markersize=4,
                        ls='None',color='navy', label= 'Non Dimensionalised Velocity',**kwargs)
        ax.set_ylim((ymin,ymax))
        ret.append(l)
        
    ax.set_xlabel(r'$U_{0}$ ([ms$^{-1}]$)')
    ax.set_ylabel(r'$M\cdot U_{0}^{-1}$')
    ax.legend(loc='upper right',fontsize=14)
    ax.grid(True)
    
    return ret
       
def plot_repeat(mean_magnitude, heights, wtref,yerr=0,ax=None,**kwargs):
    """ Plots the results for a Repeatability test from a non-
    dimensionalised timeseries. yerr specifies the uncertainty. Its default 
    value is 0.
    
    Parameters
    ----------
    

    mean_magnitude: array like
    heights: array like
    wtref: array like
    yerr: float
    ax: axes object
    kwargs: arbitrary

    Returns
    ----------
    

    ret: list    
    """
    if ax is None:
        ax=plt.gca()

    # Plot
    ret = []
    for j in range(np.shape(mean_magnitude)[1]):
        for i,value in enumerate(mean_magnitude[:,j]):
            l = ax.errorbar(j,value/wtref[i,j],yerr=yerr,fmt='o',markersize=4,
                        ls='None',color='navy',**kwargs)
            ret.append(l)
            
    labels=[str(heights)]    
    ax.set_xlabel('Measurement Number')
    ax.set_ylabel(r'$M\cdot U_{0}^{-1}$')
    ax.legend(labels,loc='lower right',fontsize=14)
    ax.grid(True)
   
    return ret         

def plot_convergence_test(data,wtref=1,ref_length=1,scale=1,ylabel='',title='',Label="Dataset",ax=None,
                          **kwargs):
    """Plots results of convergence tests  from data. This is a very limited 
    function and is only intended to give a brief overview of the convergence
    rest results using dictionaries as input objects. wtref, ref_length and 
    scale are used to determine a dimensionless time unit on the x-axis. 
    Default values for each are 1.
    
    Parameters
    ----------
    

    data: array like
    wtref: float
    ref_length: float
    scale: float
    ylabel: string
    title: string
    ax: axes object
    kwargs: arbitrary

    Returns
    ----------
    

    handles: list    
    """

    if ax is None:
        ax = plt.gca()
    handles = []
    print(['ylabel ='+ylabel])
    for key in data.keys():
        l = ax.scatter(np.ones_like(data.get(key))*key,np.asarray(data.get(key)), s=15,
                       c='navy',marker='o')
    ax.grid(True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Interval Size')
    ax.legend()
    handles.append(l)
    
    return handles
    
def plot_convergence(data_dict,ncols=3,**kwargs):
    """ Plots results of convergence tests performed on any number of 
    quantities in one plot. ncols specifies the number of columns desired in
    the output plot. kwargs contains any parameters to be passed to
    plot_convergence_test, such as wtref, ref_length and scale. See doc_string
    of plot_convergence_test for more details.
    
    Parameters
    ----------
    

    data_dict: dictionary
    ncols: integer
    kwargs: arbitrary

    Returns
    ----------
    

    axes: axes object 
    """

    fig, axes = plt.subplots(ncols,int(np.ceil(len(data_dict.keys())/ncols)),
                             figsize=(24,14))
    for (key,data), ax in zip(data_dict.items(), axes.flat):
        plot_convergence_test(data,ylabel=key,ax=ax,**kwargs)

    return axes

def plot_JTFA_STFT(u1, v1, t_eq, height, second_comp = 'v', 
                   window_length = 3500, fixed_limits = (None, None), 
                   ymax = None):
    """ Plots the joint time frequency analysis using a short-time Fourier
    transform smoothed and raw for both wind components in one figure. Returns
    the figure. To change overlap.
    
    Parameters
    ----------
    

    u1: array like
    v1: array like
    height: array like
    t_eq: array like
    second_comp: string
    window_length: integer
    ncols: integer
    kwargs: arbitrary
    fixes_limits: array like
    ymax: float

    Returns
    ----------
    

    fig: figure object 
    """
    
    #set the window size to 3500 ms - this seems to caputure the relevant 
    #frequency range
    sampling_period = t_eq[1] - t_eq[0]
    pointsPerSegment = window_length / (sampling_period)
    
    # Use a much smaller window
    data_length = len(u1)
    pointsPerSegment = min(data_length // 2, 8)  # Use small window for short data
    noverlap = pointsPerSegment // 2

    # Analyze u - f is frequency, t is time, and Zxx is the Fourier transform
    f, t, Zxx = signal.stft(u1, fs = 1.0 / (sampling_period), \
                            window = 'parzen', 
                            padded = False,
                            noverlap = noverlap, #(pointsPerSegment/2), 
                            nperseg = pointsPerSegment) #ointsPerSegment) 
    #Placeholder:
    reduced_transform_u1=Zxx
    reduced_freqs_u1=f
    aliasing_u1=t
    #print(noverlap)
    #print(nperseg)
    # get nondimensionalized forms of f and Zxx - 
    # these are f*z/h and f*S/sigma^2, respectively
    #reduced_transform_u1, reduced_freqs_u1, aliasing_u1 = \
    # wt.calc_normalization_params(f, Zxx, t, height, np.nanmean(u1),
    #                             u1.std(dtype=float), len(t_eq))
   
    # Analyze second component
    f, t, Zxx = signal.stft(v1, fs = 1.0 / (sampling_period), \
                            window = 'parzen', padded = False, 
                            noverlap = (pointsPerSegment/2),
                            nperseg = pointsPerSegment)
    # and nondimensionalize    
    #reduced_transform_v1, reduced_freqs_v1, aliasing_v1 = \
    
    # wt.calc_normalization_params(f, Zxx, t, height, np.nanmean(u1),
    #                             u1.std(dtype=float), len(t_eq))
    #reduced_transform_u1 *= 10e17
    #reduced_transform_v1 *= 10e17
     # Analyze u - f is frequency, t is time, and Zxx is the Fourier transform
    #Placeolder
    reduced_transform_v1=Zxx
    reduced_freqs_v1=f
    aliasing_v1=t
    # Create figure - 2x2 subplots
    fig, axarr = plt.subplots(2, 2)
    
    if(fixed_limits[0] is None or fixed_limits[1] is None):
        levels_u1 = np.linspace(np.nanmin(reduced_transform_u1.real), 
                                np.nanmax(reduced_transform_u1.real)*0.05,30)
    else:
        levels_u1 = np.linspace(fixed_limits[0], fixed_limits[1], 30)
        
    # Upper left plot - u1 
    axarr[0][0].set_yscale('log')
    im1=axarr[0][0].pcolormesh(t, reduced_freqs_u1[f < 0.1], 
                               reduced_transform_u1.real[f < 0.1][:], 
                               cmap='winter', 
                               vmin=np.nanmin(reduced_transform_u1.real), 
                               vmax=np.nanmax(reduced_transform_u1.real)*0.05)
    axarr[0][0].set_xlabel('f*S/sigma^2')
    axarr[0][0].set_ylabel('Frequency (f*h/mean_u)')
    axarr[0][0].set_title('u\' STFT')
    if(not ymax is None):
        axarr[0][0].set_ylim((0, ymax))
    else:
        axarr[0][0].set_ylim(np.nanmean(reduced_freqs_u1[f < 0.1])/75)
    
    # Lower left plot - u1 smoothed
    axarr[1][0].set_yscale('log')
    im2 = axarr[1][0].contourf(t, reduced_freqs_u1[f < 0.1],
                               reduced_transform_u1.real[f < 0.1][:],
                               levels_u1, cmap = 'winter')
    axarr[1][0].set_xlabel('f*S/sigma^2')
    axarr[1][0].set_ylabel('Frequency (f*h/mean_u)')
    axarr[1][0].set_title('u\' STFT smoothed')
    if(not ymax is None):
        axarr[1][0].set_ylim((0, ymax))
    else:
        axarr[1][0].set_ylim(np.nanmean(reduced_freqs_u1[f < 0.1])/75)
   
    # Upper right plot - second comp
    axarr[0][1].set_yscale('log')
    im3=axarr[0][1].pcolormesh(t, reduced_freqs_v1[f < 0.1], 
                               reduced_transform_v1.real[f < 0.1][:],
                               cmap='winter',
                               vmin=np.nanmin(reduced_transform_v1.real),
                               vmax=np.nanmax(reduced_transform_v1.real)*0.05)
    axarr[0][1].set_xlabel('f*S/sigma^2')
    axarr[0][1].set_ylabel('Frequency (f*h/mean_v)')
    axarr[0][1].set_title(second_comp + '\' STFT')
    if(not ymax is None):
        axarr[0][1].set_ylim((0, ymax))
    else:
        axarr[0][1].set_ylim(np.nanmean(reduced_freqs_v1[f < 0.1])/75)
    
    # Lower right plot - second comp smoothed
    axarr[1][1].set_yscale('log')
    im4 = axarr[1][1].contourf(t, reduced_freqs_v1[f < 0.1],
                               reduced_transform_v1.real[f < 0.1][:],
                               levels_u1,
                               cmap = 'winter')
    axarr[1][1].set_xlabel('f*S/sigma^2')
    axarr[1][1].set_ylabel('Frequency (f*h/mean_v)')
    axarr[1][1].set_title(second_comp + '\' STFT smoothed')
    if(not ymax is None):
        axarr[1][1].set_ylim((0, ymax))
    else:
        axarr[1][1].set_ylim(np.nanmean(reduced_freqs_v1[f < 0.1])/75)


    print('reduced_transform u min '+str(np.nanmin(reduced_transform_u1.real))
     + '\n                     max '+str(np.nanmax(reduced_transform_u1.real))
     + '\n                     mean '+str(np.nanmean(reduced_transform_u1.real)
          ))
    print('reduced_freqs u     min '+str(np.nanmin(reduced_freqs_u1.real))
     + '\n                     max '+str(np.nanmax(reduced_freqs_u1.real))
     + '\n                     mean '+str(np.nanmean(reduced_freqs_u1.real)))
    print('reduced_transform v min '+str(np.nanmin(reduced_transform_v1.real))
     + '\n                     max '+str(np.nanmax(reduced_transform_v1.real))
     + '\n                     mean '+str(np.nanmean(reduced_transform_v1.real)
          ))
    print('reduced_freqs v     min '+str(np.nanmin(reduced_freqs_v1.real))
     + '\n                     max '+str(np.nanmax(reduced_freqs_v1.real))
     + '\n                     mean '+str(np.nanmean(reduced_freqs_v1.real)))
    
    cbar1 = fig.colorbar(im1, ax = axarr[0][0])
    cbar2 = fig.colorbar(im2, ax = axarr[1][0])
    cbar3 = fig.colorbar(im3, ax = axarr[0][1])
    cbar4 = fig.colorbar(im4, ax = axarr[1][1])

    if(fixed_limits != (None, None)):
        cbar1.set_clim(fixed_limits[0], fixed_limits[1])
        cbar2.set_clim(fixed_limits[0], fixed_limits[1])
        cbar3.set_clim(fixed_limits[0], fixed_limits[1])
        cbar4.set_clim(fixed_limits[0], fixed_limits[1])
    else:
        cbar1.set_clim(np.nanmin(reduced_transform_u1.real),
                       np.nanmax(reduced_transform_u1.real)*0.05)
        cbar3.set_clim(np.nanmin(reduced_transform_v1.real),
                       np.nanmax(reduced_transform_v1.real)*0.05)
     
    cbar1.update_normal(im1)
    cbar2.update_normal(im2)
    cbar3.update_normal(im3)
    cbar4.update_normal(im4)
        
    plt.tight_layout()
    
    return fig
 
def plot_stdevs(data, t_eq, tau, ax=None, **kwargs):
    """ This function plots the spread of an array based on how many standard 
    deviations each point is from the mean over each tau-long time period.
    
    ----------
    Parameters

    data: array like
    t_eq: array like
    tau: integer
    ax: axes object
    """
    # Get current axis
    if ax is None:
        ax = plt.gca()
    
    for i,value in enumerate(t_eq):
        if(value > t_eq[0] + tau):
            step_size = i
            break
    
    starts = np.arange(0,np.size(t_eq)-step_size,step_size)
    stops =  np.arange(step_size,np.size(t_eq),step_size)
    stds_from_mean = {}
    keys = [1,2,3,4,5,6,7]
    stds_from_mean.fromkeys(keys)
    for key in keys:
        stds_from_mean[key] = 0
                      
    for begin,end in zip(starts,stops):
        segment = data[begin : end]
        mean = np.nanmean(segment)
        std = np.std(segment)
        for value in segment:
            perturbation = value - mean
            if perturbation < 1 * std:
                stds_from_mean[1] += 1
            if perturbation > 1 * std:
                stds_from_mean[2] += 1
            if perturbation > 2 * std:
                stds_from_mean[3] += 1
            if perturbation > 3 * std:
                stds_from_mean[4] += 1
            if perturbation > 4 * std:
                stds_from_mean[5] += 1
            if perturbation > 5 * std:
                stds_from_mean[6] += 1
            if perturbation > 6 * std:
                stds_from_mean[7] += 1
     
    ax.bar(range(len(stds_from_mean)), list(stds_from_mean.values()),
           align='center')
    ax.set_xticks(range(len(stds_from_mean)), list(stds_from_mean.keys()))
  
def plot_perturbation_rose(u1, v1, total_mag, total_direction, 
                           bar_divider = 3000, second_comp = 'v'):
    """ Plots a detailed wind rose using only the perturbation component of
    the wind. Number of bars depends on bar_divider and length of u1.
    
    Parameters
    ----------
    

    u1: array like
    v1: array like
    total_mag: array like
    total_direction: array like
    bar_divider: float
    second_comp: string
    """
    
    u1 = np.asarray(u1)
    v1 = np.asarray(v1)
    total_mag = np.asarray(total_mag)
    total_direction = np.asarray(total_direction)
    
    fig, axarr = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
    # Calculate perturbation direction
    unit_WD = np.arctan2(v1,u1) * 180/np.pi
    directions = (360 + unit_WD) % 360
    
    # Calculate perturbation magnitude
    speeds = np.sqrt(np.power(u1, 2) + np.power(v1, 2))
    
    # Plot the wind rose. Method called can be found in tools.py
    wt.plots.plot_windrose(total_mag, total_direction, len(total_mag) / 
                          bar_divider, ax = axarr[0], left_legend = True)
    wt.plots.plot_windrose(speeds, directions, len(u1) / 
                          bar_divider, ax = axarr[1])

    fig.suptitle('u-' + second_comp + ' plane', y = 0.8, x = 0.55)
    axarr[0].set_title('Wind Rose', y = 1.2)
    axarr[1].set_title('Perturbations', y = 1.2)

    axarr[0].set_position([0.2, 0.125, 0.4, 0.4])
    axarr[1].set_position([0.6, 0.125, 0.4, 0.4])

def plot_arrival_law(delta_t_arr, arrival_law, binscenters, 
                        data_entries, popt, logplot = None, ax = None, **kwargs):
    """ 
    Plots particle arrival law and scale KDE-pdf to mean data rate before plotting.

    Parameters
    ----------
    

    delta_t_arr: array like
    arrival_law: array like
    binscenters: float
    data_entries: array like
    popt: float
    logplot: boolean
    ax: axes object
    kwargs: arbitrary

    Returns
    ----------
    

    ret: axes object
    lgd: legend object
    """
    if logplot == None:
        logplot = True
    if ax is None:
       ax = plt.gca()
    ret = []

    # zip-sort 
    arrival_law = [delta_t_arr for _,
                    delta_t_arr in sorted(zip(delta_t_arr, arrival_law))]
    delta_t_arr = sorted(delta_t_arr)

    # particle arrival law
    def fit_function(x, A):
        return (A * np.exp(-x * A) )

    # Generate enough x values to make the curves look smooth.
    xspace = np.linspace(0, max(binscenters), 10000)
    data_entries[ data_entries==0 ] = np.nan

    if logplot:
        # Plot the histogram,the fitted function and the expected law
        b = ax.semilogy(binscenters, data_entries, label=r'pdf($\delta t$)')
        f = ax.semilogy(xspace, fit_function(xspace, *popt), 
                    label=r'fit: $\frac{N}{T_{mes}}=$' + '{}'.format(np.around(popt[0],2)))
        a = ax.semilogy(delta_t_arr, arrival_law, 
                    label = 'particle arrival law', linestyle = ':')
        plt.xlim(0.,max(delta_t_arr))
        plt.ylim(10**(-3.),10**4.)
    else:
        # Plot the histogram,the fitted function and the expected law
        b = ax.plot(binscenters, data_entries, label=r'pdf($\delta t$)')
        f = ax.plot(xspace, fit_function(xspace, *popt), 
                    label=r'fit: $\frac{N}{T_{mes}}=$' + '{}'.format(np.around(popt[0],2)))            
        a = ax.plot(delta_t_arr, arrival_law, 
                    label = 'particle arrival law', linestyle = ':')
        plt.xlim(0.,max(delta_t_arr))
        plt.ylim(10**(-3.),10**4.)

    ret.append(b+f+a)

    ax.set_xlabel(r'$\delta t$ (ms)')
    ax.set_ylabel(r'$P(\delta t)$ (1/s)')
    ax.grid()
    lgd = plt.legend(loc='best')

    return ret, lgd

def plot_transit_time_distribution(transit_time, skew, ax=None):
    """ 
    Plots transit-time distribution.

    Parameters
    ----------
    

    transit time: array like
    skew: float
    ax: axes object

    Returns
    ----------
    

    ret: axes object
    """

    if ax is None:
       ax = plt.gca()

    ret = ax.hist(transit_time, density=False, 
            bins='auto')
    ax.set_ylabel('Number of Particles')
    ax.set_xlabel(r'$t_{transit}$ $(\mu s)$')
    ax.grid()
    ax.text(x=0.8, y=0.9, s=r'$\gamma = {}$'.format(np.around(skew,2)), 
                transform=ax.transAxes)

    return ret

def plot_wavelet_transform(wavelet, scale, u_eq, t_eq, z_val, ax=None):
    """ 
    Plots CWT-results as a contour-plot. 
    The Wavelet-Coefficients Wn(s,t) are plotted for each timestep in a defined range of scales. 

    ----------
    Parameters

    wavelet: array like
    scale: array-like
    u_eq: array-like
    t_eq: array-like
    z_val: float
    ax: axes object

    ----------
    Returns

    ret: axes object
    """

    if ax is None:
        ax = plt.gca()

    f_scale = z_val/(scale * np.mean(u_eq))

    im1 = ax.contour(t_eq,
                f_scale, 
                np.abs(wavelet)**2. * np.std(u_eq)**-2.,
                levels = 15,
                colors='gray')
    im2 = ax.contourf(t_eq,
                f_scale, 
                np.abs(wavelet)**2. * np.std(u_eq)**-2.,
                levels = 15,
                cmap='YlGnBu')
    # plot cone of incidence
    pl1 = ax.plot(scale*2.**0.25,
                f_scale,
                color='black',
                linestyle='dashed')
    pl2 = ax.plot(np.amax(t_eq)-(scale*2.**0.25),
                f_scale,
                color='black',
                linestyle='dashed')
    
    # if colorbar is wished:
    # plt.colorbar(im2, label=r'$|W_n(f,t)|^{2} \cdot \sigma_u^{-2}$ (-)')
    ax.grid(True)
    ax.set_ylabel(r'$f z \cdot \overline{u}^{-1}$ (-)', fontsize=18)
    ax.set_xlabel(r'$t$ (s)', fontsize=18)
    ax.set_yscale('log')
    ax.set_box_aspect(0.5)    
    ax.set_ylim(np.min(f_scale), 100.)
    ax.set_xlim(0., np.amax(t_eq))

    return im1, im2, pl1, pl2
