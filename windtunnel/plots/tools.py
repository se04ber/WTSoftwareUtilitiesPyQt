#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Plotting utilities.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections import get_projection_class


__all__ = [
    'Windrose',
    'plot_windrose',
    'plot_DWD_windrose',
    'plot_rose',
    'plot_rose_map',
    'plot_pdfs',
    'plot_pdfs_err',
    'plot_cdfs',
]

class Windrose:
     def __init__(self,dd,ff):
        val = np.where(np.logical_and(np.logical_and(dd<=360.,dd>=0.),ff<100.))
        sorted = np.argsort(dd[val])
        self.dd = dd[val][sorted]
        self.ff = ff[val][sorted]
        self.description = 'plot a windrose from wind direction (dd) and wind speed (ff) data'
     def pack(self,incr,bins):
        self.wdir=np.arange(0,360,incr)
        self.ws = []
        dir = 0.
        while dir<360:
             ind = np.where(np.logical_and(self.dd<=dir+incr,self.dd>=dir))
             self.ws.append(np.histogram(self.ff[ind], bins = bins)[0]/self.ff.size*100)
             dir+=incr
        return self.wdir,np.array(self.ws)
    
    
def plot_windrose(inFF,inDD, num_bars = 10, ax = None, left_legend = False):
    """ Plots windrose with dynamic velocity classes of each 10% percentile and
    10 degree classes for directional data. The representation of the windrose 
    in this function is more detailed than in plot_DWD_windrose().
   
    Parameters
    ----------
    
    
    inFF: np.array
    inDD: np.array
    num_bars: integer
    ax: pyplot axes object, must be polar
    left_legend: bool
    
    """

    ffs = np.array([])
    percs = np.arange(0,100,10)
    for perc in percs:
        ffs = np.append(ffs,np.percentile(inFF,perc))
        
    factors_of_360 = np.array([1., 2., 3., 4., 5., 6., 8., 10., 12.,
                               18., 20., 36., 40., 120., 360.])
    # find appropriate degree width for each bar
    dd_range = min(factors_of_360[factors_of_360 > 360/num_bars]) 
    labels = []
    for i,f in enumerate(ffs[:-2]):
       labels.append(r'$'+'{0:.2f}-{1:.2f}'.format(f,ffs[i+1])+'\ ms^{-1}$')
    labels.append(r'$'+'>{0:.2f}'.format(ffs[-2])+'\ ms^{-1}$')
    
    ##  DATA PROCESSING
    dd,ff = Windrose(np.asarray(inDD),np.asarray(inFF)).pack(dd_range,ffs)
    dd = dd*np.pi/180.
    
    ##  PLOT
    width = dd_range*np.pi/180.
    cmap = plt.cm.jet
    
    if(ax is None):
        ax = plt.subplot(111,polar=True)
        
    ax.bar(dd, ff[:,0],
           width=width,
           bottom=0.,
           facecolor=cmap(0),
           label=labels[0],
           align='edge',
           edgecolor='none')
    
    for i in range(ff[0].size-1):
       ax.bar(dd, ff[:,i+1],
              width=width,
              bottom=ff[:,i],
              facecolor=cmap(np.linspace(0, 1, ff[0].size)[i+1]),
              label=labels[i+1],
              align='edge',
              edgecolor='none')
       ff[:,i+1]=ff[:,i+1]+ff[:,i]
    ax.set_yticklabels([])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    
    if(left_legend):
        bbox = (-1.15, 0.5)
    else:
        bbox = (1.25, 0.5)
    ax.legend(bbox_to_anchor=bbox, loc='center left',
                     borderaxespad=0.,fontsize=8, handlelength=1)
    
    # copied from: 
    # https://stackoverflow.com/questions/9651092/
    # my-matplotlib-pyplot-legend-is-being-cut-off/14246266
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

def plot_DWD_windrose(inFF,inDD):
    """ Plots windrose according to DWD classes of 1 m/s for velocity data and
    30 degree classes for directional data. The representation of the windrose 
    in this function is less detailed than in plotwindrose().

    Parameters
    ----------
    
    inFF: np.array
    inDD: np.array

    """
    ffs = np.arange(np.max(inFF))
    dd_range = 30.
    labels = []
    for i,f in enumerate(ffs[:-2]):
       labels.append(r'$'+'{0:.2f}-{1:.2f}'.format(f,ffs[i+1])+'\ ms^{-1}$')
    labels.append(r'$'+'>{0:.2f}'.format(ffs[-2])+'\ ms^{-1}$')
    
    ##  DATA PROCESSING
    dd,ff = Windrose(inDD,inFF).pack(dd_range,ffs)
    dd = dd*np.pi/180.
    
    ##  PLOT
    width = dd_range*np.pi/180.
    cmap = plt.cm.jet
    ax = plt.subplot(111,polar=True)
    ax.bar(dd, ff[:,0],
           width=width,
           bottom=0.,
           facecolor=cmap(0),
           label=labels[0],
           align='edge',
           edgecolor='none')
    
    for i in range(ff[0].size-1):
       ax.bar(dd, ff[:,i+1],
              width=width,
              bottom=ff[:,i],
              facecolor=cmap(np.linspace(0, 1, ff[0].size)[i+1]),
              label=labels[i+1],
              align='edge',
              edgecolor='none')
       ff[:,i+1]=ff[:,i+1]+ff[:,i]
    ax.set_yticklabels([])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.legend(bbox_to_anchor=(1.14, 0.5), loc='center left',
              borderaxespad=0.,fontsize=12)
    
def plot_rose(inFF,inDD,ff_steps,dd_range):
    """ Plots windrose according to user specified input from ff_steps and 
    dd_Range.

    Parameters
    ----------
    
    inFF:  np.array
    inDD:  np.array
    ff_steps: list or np.array
    dd_range: int or float

    """
    
    labels = []
    for i,f in enumerate(ff_steps[:-2]):
       labels.append(r'$'+'{0:.2f}-{1:.2f}'.format(f,ff_steps[i+1])+'\ ms^{-1}$')
    labels.append(r'$'+'>{0:.2f}'.format(ff_steps[-2])+'\ ms^{-1}$')
    
    ##  DATA PROCESSING
    dd,ff = Windrose(inDD,inFF).pack(dd_range,ff_steps)
    #dd = dd*np.pi/180.
    
    ##  PLOT
    width = dd_range#*np.pi/180.
    cmap = plt.cm.jet
    ax = plt.subplot(111,polar=True)
    ax.bar(dd, ff[:,0],
           width=width,
           bottom=0.,
           facecolor=cmap(0),
           label=labels[0],
           align='edge',
           edgecolor='none')
    
    for i in range(ff[0].size-1):
       ax.bar(dd, ff[:,i+1],
              width=width,
              bottom=ff[:,i],
              facecolor=cmap(np.linspace(0, 1, ff[0].size)[i+1]),
              label=labels[i+1],
              align='edge',
              edgecolor='none')
       ff[:,i+1]=ff[:,i+1]+ff[:,i]
    ax.set_yticklabels([])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.legend(bbox_to_anchor=(1.14, 0.5), loc='center left',
              borderaxespad=0.,fontsize=12)
    plt.tight_layout()
    plt.show()
    

def plot_rose_map(inFF, inDD, x_coor, y_coor, ff_steps, dd_range, ax, alpha,cmap=plt.cm.viridis):
    """ Plots windrose according to user specified input from ff_steps and
    dd_Range.

    Parameters
    ----------

    inFF: np.array, contains the windspeeds
    inDD:np.array, contains the winddirections
    x_coor:  np.array, contains the x coordinates of the measurements
    y_coor:  np.array, contains the x coordinates of the measurements
    ff_steps:  list or np.array, specifies the steps of the windspeeds for the windrose
    dd_range: int or float, specifies the direction ranges
    ax: pyplot axes object
    alpha: float
    cmap: `~matplotlib.colors.Colormap

    Returns
    ----------
    

    ax:  axes object
    cbar: matplotlib object

    """


    # this dummy image will only be generated to create a colorbar
    dummy_img = ax.imshow(np.meshgrid(np.asarray(ff_steps)),extent= (10000,10000,10010,10010),
                          cmap=cmap)
    cbar = plt.colorbar(dummy_img,fraction=0.046,pad=0.04)
    cbar.set_label('Windspeed in (-)')

    ax.set_xlim([np.min(x_coor)-(np.abs(np.min(x_coor))+np.max(x_coor))/10,
                 np.max(x_coor)+(np.abs(np.min(x_coor))+np.max(x_coor))/10])
    ax.set_ylim([np.min(y_coor)-(np.abs(np.min(y_coor))+np.max(y_coor))/10,
                 np.max(y_coor)+(np.abs(np.min(y_coor))+np.max(y_coor))/10])
    
    for rose in range(inFF.shape[1]):
        ##  DATA PROCESSING
        dd, ff = Windrose(inDD[rose], inFF[rose]).pack(dd_range, ff_steps)
        dd = dd * np.pi / 180.

        ##  PLOT
        width = (np.pi) * dd_range / 180  # (2*np.pi)/dd_range
        ax_sub= inset_axes(ax, width=0.55, height=0.55, loc=10,
                       bbox_to_anchor=(x_coor[rose],y_coor[rose]),
                       bbox_transform=ax.transData,
                       borderpad=0.0, axes_class=get_projection_class("polar"))
        ax_sub.bar(dd, ff[:, 0],
                width=width,
                bottom=0.,
                facecolor=cmap(0),
                align='edge',
                edgecolor='none', alpha=alpha, linewidth=0.05)

        for i in range(ff[0].size - 1):
            ax_sub.bar(dd, ff[:, i + 1],
                width=width,
                bottom=ff[:, i],
                facecolor=cmap(np.linspace(0, 1, ff[0].size)[i + 1]),
                align='edge',
                edgecolor='none', alpha=alpha, linewidth=0.05)
            ff[:, i + 1] = ff[:, i + 1] + ff[:, i]

        ax_sub.set_theta_zero_location("W")
        ax_sub.set_theta_direction(1)
        ax_sub.set_xticklabels([])
        ax_sub.set_yticklabels([])
        ax_sub.grid(False)
        ax_sub.set_yticks([])
        ax_sub.set_xticks([])
        ax_sub.axis('off')
    return ax, cbar

def plot_pdfs(sets,lablist,ax=None, **kwargs):
    """Plots PDFs of data in sets using the respective labels from lablist.

    Parameters
    ----------
    
    sets: iterable set of data
    lablist: list of strings
    ax: axis passed to function
    kwargs : additional keyword arguments passed to plt.plot()
    
    Returns
    ----------
    

    ret: axes object
    
    """
    if ax is None:
       ax = plt.gca()
        
    ret = []
    for data, label in zip(sets, lablist):
        heights,bins = np.histogram(data[~np.isnan(data)],bins='auto')
        # Normalize
        heights = heights/float(sum(heights))
        binMids=bins[:-1]+np.diff(bins)/2.
        l = ax.plot(binMids,heights,label=label, **kwargs)
        ret.append(l)
        
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.grid('on')
    
    return ret

def plot_pdfs_err(sets,lablist,error,ax=None, **kwargs):
    """Plots PDFs of data in sets using the respective labels from lablist with
    a given margin of error.

    Parameters
    ----------
    
    sets: array-like
    lablist: list of strings
    error: int or float
    ax: axis passed to function
    kwargs : additional keyword arguments passed to plt.plot()
    
    Returns
    ----------
    
    ret: list of axes object
    
    """
    if ax is None:
       ax = plt.gca()
        
    ret = []
    for data, label in zip(sets, lablist):
        heights,bins = np.histogram(data[~np.isnan(data)],bins='auto')
        # Normalize
        heights = heights/float(sum(heights))
        binMids=bins[:-1]+np.diff(bins)/2.
        l = ax.plot(binMids,heights,label=label, **kwargs)
        plt.fill_between(binMids, heights-heights*error,
                             heights+heights*error,alpha=0.5,
                             edgecolor='lightsteelblue',
                             facecolor='lightsteelblue', label='Error',
                             **kwargs)
        ret.append(l)
        
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.grid('on')
    
    return ret

def plot_cdfs(sets, lablist, ax=None, **kwargs):
    """Plots CDFs of data in sets using the respective labels from lablist

    Parameters
    ----------
     
    sets: array like
    lablist: list of strings
    ax: axis passed to function
    kwargs : additional keyword arguments passed to plt.plot()
    
    Returns
    ----------
    

    ret: list of axes object
    
    """
    if ax is None:
        ax = plt.gca()
        
    ret = []
    for data, label in zip(sets, lablist):
        # Cumulative distributions:
        l = ax.plot(np.sort(data), np.linspace(0, 1, data.size),
                    label=label, **kwargs)
        ret.append(l)
        
    ax.set_ylabel('Count')
    ax.grid('on')
    ax.legend()
    
    return ret
