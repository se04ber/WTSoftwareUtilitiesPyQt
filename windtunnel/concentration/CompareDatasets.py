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
from sklearn.metrics import mean_squared_error
   


#File for all functions comparing two Objects 


def compare_point_concentrations(PointConcTsArray, labels=None, max_lag=50,functionsForOverview=["all"],dimensionless="False",errorConc=5.0, errorType="absolute"):
    """
    Compare two point concentration datasets with autocorrelation.
    
    Parameters:
    -----------
    pc1 : array-like
        First point concentration dataset
    pc2 : array-like
        Second point concentration dataset
    labels : tuple, optional
        Labels for the two datasets (default: ['Dataset 1', 'Dataset 2'])
    max_lag : int, optional
        Maximum lag for autocorrelation (default: 50)

    Further Options:
    -----------------
    -Choose which plots to show in overview
    functionsForOverview=["all"]

    -Add Errorbars:
        If errorConc != None -> overgive errorvalue for concentration to visualise uncertainty as errorbars in plots
        If errorType !=None /"percentage"/"absolute" type of overgiving error
        
    """
    pc1=PointConcTsArray[0]
    pc2=PointConcTsArray[1]

    wtref1 = pc1.wtref
    wtref2 = pc2.wtref

    if(dimensionless=="True"):
        pc1 = pc1.c_star
        pc2 = pc2.c_star
    else:
        pc1 = pc1.net_concentration
        pc2 = pc2.net_concentration


    # Set defacomapreult labels if not provided
    if labels is None:
        labels = ['Dataset 1', 'Dataset 2']

    #Choose which plots to show:
    #if(functionsForOverview!=["all"]):
        

    pc1_values=pc1
    pc2_values=pc2

    # Calculate correlation statistics
    pearson_r, pearson_p = sc.pearsonr(pc1_values, pc2_values)
    spearman_r, spearman_p = sc.spearmanr(pc1_values, pc2_values)
    rmse = np.sqrt(mean_squared_error(pc1_values, pc2_values))
    max_val = max(np.max(pc1_values), np.max(pc2_values))
    min_val = min(np.min(pc1_values), np.min(pc2_values))
    
    # Basic statistics subplot
    stats_data = [
        ('Mean', np.mean(pc1), np.mean(pc2)),
        ('Std Dev', np.std(pc1), np.std(pc2)),
        ('Skewness', sc.skew(pc1), sc.skew(pc2)),
        ('Percentile 95', np.percentile(pc1, 95), np.percentile(pc1, 95)),
        ('Turbulence Intensity v', np.std(wtref1)/np.mean(wtref1), np.std(wtref2)/np.mean(wtref2)),
        #('Turbulence Intensity c', np.std(pc1)/np.mean(pc1), np.std(pc2)/np.mean(pc2)),
        #('RMSE', rmse, '-')
    ]


    #Calculate array of correct absolute error value if overgiven(errorConc!=None) and percentage or absolute
    if(errorConc!=None):
        if(errorType=="percentage"):
            errorConc * np.mean(pc1)
            y_values = [stats_data[0][1], stats_data[0][2]]
            
        # Vectorized one-line approach for error bar calculation and plotting
            errorConc = np.array([stats_data[0][1], stats_data[0][2]]) * (errorConc/100)
        
        else:
            errorConc = np.full(len(PointConcTsArray), errorConc)


    plt.figure(figsize=(12, 6))
    plt.plot(pc1_values, label=labels[0])
    plt.plot(pc2_values, label=labels[1])
    plt.legend()
    plt.title("Concentration time series comparison")
    plt.xlabel("Time Index")
    plt.ylabel("Concentration")
    plt.tight_layout()
    plt.show()
    
  
    fig, axs = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle('Point Concentration Comparison', fontsize=16)
    
    # Row 1, Col 1: Histogram comparison
    axs[0, 0].hist(pc1_values, bins='auto', alpha=0.5, label=labels[0])
    axs[0, 0].hist(pc2_values, bins='auto', alpha=0.5, label=labels[1])
    axs[0, 0].set_title('Distribution Comparison')
    axs[0, 0].set_xlabel('Concentration')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].legend()
    

    # NEW: Row 3, Col 2: Probability Density Functions
    from scipy.stats import gaussian_kde
    density1 = gaussian_kde(pc1_values)
    density2 = gaussian_kde(pc2_values)
    x = np.linspace(min_val, max_val, 200)
    axs[0, 1].plot(x, density1(x), label=labels[0])
    axs[0, 1].plot(x, density2(x), label=labels[1])
    axs[0, 1].set_title('Probability Density Functions')
    axs[0, 1].set_xlabel('Concentration')
    axs[0, 1].set_ylabel('Density')
    axs[0, 1].legend()
    
    # NEW: Row 3, Col 3: Cumulative Distribution Functions
    sorted1 = np.sort(pc1_values)
    sorted2 = np.sort(pc2_values)
    yvals1 = np.arange(1, len(sorted1) + 1) / len(sorted1)
    yvals2 = np.arange(1, len(sorted2) + 1) / len(sorted2)
    
    axs[0, 2].plot(sorted1, yvals1, label=labels[0])
    axs[0, 2].plot(sorted2, yvals2, label=labels[1])
    axs[0, 2].set_title('Cumulative Distribution Functions')
    axs[0, 2].set_xlabel('Concentration')
    axs[0, 2].set_ylabel('Cumulative Probability')
    axs[0, 2].legend()
    
    # Row 1, Col 2: Mean comparison
    #axs[1,0].bar([f'{labels[0]} Mean', f'{labels[1]} Mean'],
    #             [stats_data[0][1], stats_data[0][2]])
    
    axs[1, 0].errorbar(x=[1, 2],  y=[stats_data[0][1], stats_data[0][2]], yerr=errorConc,  fmt='o', capsize=5 )
    #axs[1, 0].plot([stats_data[0][1],stats_data[0][2]],".",labels=labels)
    axs[1, 0].set_title('Mean Comparison')
    axs[1, 0].set_ylabel('Value')
    
    # Row 1, Col 3: Box plot
    axs[1, 1].boxplot([pc1_values, pc2_values], labels=labels)
    axs[1, 1].set_title('Box Plot Comparison')
    axs[1, 1].set_ylabel('Concentration')
    

    # Row 2, Col 3: Q-Q plot for second dataset
    sc.probplot(pc2_values, dist="norm", plot=axs[1, 2])
    axs[1, 2].set_title(f'Q-Q Plot - {labels[1]}')
    

    # Row 2, Col 2: Scatter plot with 1:1 line and correlation statistics
    axs[2, 0].scatter(pc1_values, pc2_values, alpha=0.5)
    axs[2, 0].plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')
    axs[2, 0].set_title(f'Scatter Plot\nPearson r: {pearson_r:.3f}')#, Spearman ρ: {spearman_r:.3f}, RMSE: {rmse:.3f}')
    axs[2, 0].set_xlabel(labels[0])
    axs[2, 0].set_ylabel(labels[1])
    axs[2, 0].legend()
    

    # NEW: Row 3, Col 1: Residual plot
    residuals = pc2_values - pc1_values  # Assuming pc1 is the reference
    x = np.arange(0,len(residuals),1)
    axs[2, 1].scatter(x, residuals, alpha=0.5)
    axs[2, 1].axhline(y=0, color='r', linestyle='--')
    axs[2, 1].set_title('Residual Plot')
    axs[2, 1].set_xlabel(f'Reference ({labels[0]})')
    axs[2, 1].set_ylabel(f'Residuals ({labels[1]} - {labels[0]})')
    

    # Row 2, Col 1: Autocorrelation
    autocorr1 = scSignal.correlate(pc1_values - np.mean(pc1_values), pc1_values - np.mean(pc1_values), mode='full')
    autocorr1 = autocorr1[len(autocorr1)//2:]
    autocorr1 /= autocorr1[0]
    
    autocorr2 = scSignal.correlate(pc2_values - np.mean(pc2_values), pc2_values - np.mean(pc2_values), mode='full')
    autocorr2 = autocorr2[len(autocorr2)//2:]
    autocorr2 /= autocorr2[0]

    corr3  = scSignal.correlate(pc1_values - np.mean(pc1_values), pc2_values - np.mean(pc2_values), mode='full')
    
    axs[2, 2].plot(range(min(max_lag, len(autocorr1))), autocorr1[:max_lag], label=labels[0])
    axs[2, 2].plot(range(min(max_lag, len(autocorr2))), autocorr2[:max_lag], label=labels[1])
    axs[2, 2].plot(range(min(max_lag, len(corr3))), corr3[:max_lag], label="Cross correlation")
    axs[2, 2].set_title('Autocorrelation and cross-correlation')
    axs[2, 2].set_xlabel('Lag')
    axs[2, 2].set_ylabel('Autocorrelation')
    axs[2, 2].legend()

    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    plt.show()
    
    # Print out detailed statistics
    #print("\nDetailed Comparison:")
    #for stat, val1, val2 in stats_data:
    #    if val2 == '-':
    #        print(f"{stat}: {val1:.4f}")
    #    else:
    #        print(f"{stat}: {labels[0]} = {val1:.4f}, {labels[1]} = {val2:.4f}")
    

    # Print out detailed statistics
    print("\nDetailed Comparison:")
    for stat, val1, val2 in stats_data:
        print(f"{stat}: {labels[0]} = {val1:.4f}, {labels[1]} = {val2:.4f}")


    # Create and print a correlation summary
    #print("\nCorrelation Summary:")
    #print(f"Pearson correlation coefficient: {pearson_r:.4f} (p-value: {pearson_p:.4e})")
    #print(f"Spearman rank correlation: {spearman_r:.4f} (p-value: {spearman_p:.4e})")
    #print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    return


def compare_point_concentrations_2(PointConcTsArray, functionsForOverview=["all"],labels=None, max_lag=50, dimensionless="False", errorConc=5.0, errorType="absolute"):
    """
    Compare multiple point concentration datasets with autocorrelation.
    
    Parameters:
    -----------
    PointConcTsArray : list of arrays
        List of point concentration datasets to compare
    labels : list, optional
        Labels for the datasets (default: ['Dataset 1', 'Dataset 2', ...])
    max_lag : int, optional
        Maximum lag for autocorrelation (default: 50)

    Further Options:
    -----------------
    -Choose which plots to show in overview
    functionsForOverview=["all"]

    -Add Errorbars:
        If errorConc != None -> overgive errorvalue for concentration to visualise uncertainty as errorbars in plots
        If errorType !=None /"percentage"/"absolute" type of overgiving error
        
    """
    # Extract datasets and prepare containers
    num_datasets = len(PointConcTsArray) 
    pc_values = []
    wtref_values = []
    
    # Process each dataset
    for i in range(num_datasets):
        wtref_values.append(PointConcTsArray[i].wtref)
        
        if dimensionless == "True":
            pc_values.append(PointConcTsArray[i].c_star)
        else:
            pc_values.append(PointConcTsArray[i].net_concentration)

    # Set default labels if not provided
    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(num_datasets)]
    
    # Find global min and max for plotting
    max_val = max([np.max(pc) for pc in pc_values])
    min_val = min([np.min(pc) for pc in pc_values])
    
    # Calculate basic stats for all datasets
    stats_rows = []
    
    # Mean
    means = [np.mean(pc) for pc in pc_values]
    stats_rows.append(('Mean', *means))
    
    # Standard deviation
    stds = [np.std(pc) for pc in pc_values]
    stats_rows.append(('Std Dev', *stds))
    
    # Skewness
    skews = [sc.skew(pc) for pc in pc_values]
    stats_rows.append(('Skewness', *skews))
    
    # 95th percentile
    p95s = [np.percentile(pc, 95) for pc in pc_values]
    stats_rows.append(('Percentile 95', *p95s))
    
    # Turbulence intensity v
    ti_vs = [np.std(wtref)/np.mean(wtref) for wtref in wtref_values]
    stats_rows.append(('Turbulence Intensity v', *ti_vs))
    
    # Turbulence intensity c
    ti_cs = [np.std(pc)/np.mean(pc) for pc in pc_values]
    stats_rows.append(('Turbulence Intensity c', *ti_cs))
    
    # Calculate correlation statistics (comparing all to the first dataset)
    if num_datasets > 1:
        for i in range(1, num_datasets):
            pearson_r, pearson_p = sc.pearsonr(pc_values[0], pc_values[i])
            spearman_r, spearman_p = sc.spearmanr(pc_values[0], pc_values[i])
            rmse = np.sqrt(mean_squared_error(pc_values[0], pc_values[i]))
            
            # Add to stats with placeholder dashes for datasets not involved in comparison
            pearson_row = ['Pearson r (vs Dataset 1)'] + ['-'] * num_datasets
            pearson_row[i+1] = pearson_r
            stats_rows.append(tuple(pearson_row))
            
            spearman_row = ['Spearman rho (vs Dataset 1)'] + ['-'] * num_datasets
            spearman_row[i+1] = spearman_r
            stats_rows.append(tuple(spearman_row))
            
            rmse_row = ['RMSE (vs Dataset 1)'] + ['-'] * num_datasets
            rmse_row[i+1] = rmse
            stats_rows.append(tuple(rmse_row))
    
    # Calculate error bars if provided
    if errorConc is not None:
        if errorType == "percentage":
            # Calculate percentage-based error
            error_values = np.array(means) * (errorConc/100)
        else:
            # Use absolute error
            error_values = np.full(num_datasets, errorConc)
    else:
        error_values = None

    # Time series plot (separate figure)
    #plt.figure(figsize=(12, 6))
    #for i in range(num_datasets):
    #    plt.plot(pc_values[i], label=labels[i])
    #plt.legend()
    #plt.title("Concentration time series comparison")
    #plt.xlabel("Time Index")
    #plt.ylabel("Concentration")
    #plt.tight_layout()
    #plt.show()
    num_datasets = len(pc_values)
    # Create subplots, one for each dataset
    fig, axes = plt.subplots(num_datasets, 1, figsize=(12, 2*num_datasets), sharex=True)
    # Handle the case when there's only one dataset
    if num_datasets == 1:
        axes = [axes]
    # Plot each time series in its own subplot
    for i in range(num_datasets):
        axes[i].plot(pc_values[i])
        axes[i].set_ylabel("Concentration")
        axes[i].set_title(labels[i])
    # Add x-label only to the bottom subplot
    axes[-1].set_xlabel("Time Index")
    # Add an overall title
    plt.suptitle("Concentration time series comparison", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    
    # Create figure with subplots (3x3 grid)
    fig, axs = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle('Point Concentration Comparison', fontsize=16)
    
    # Row 1, Col 1: Histogram comparison
    for i in range(num_datasets):
        axs[0, 0].hist(pc_values[i], bins='auto', alpha=0.7/num_datasets, label=labels[i])
    axs[0, 0].set_title('Distribution Comparison')
    axs[0, 0].set_xlabel('Concentration')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].legend()
    
    # Row 1, Col 2: Probability Density Functions
    from scipy.stats import gaussian_kde
    x = np.linspace(min_val, max_val, 200)
    for i in range(num_datasets):
        density = gaussian_kde(pc_values[i])
        axs[0, 1].plot(x, density(x), label=labels[i])
    axs[0, 1].set_title('Probability Density Functions')
    axs[0, 1].set_xlabel('Concentration')
    axs[0, 1].set_ylabel('Density')
    axs[0, 1].legend()
    
    # Row 1, Col 3: Cumulative Distribution Functions
    for i in range(num_datasets):
        sorted_values = np.sort(pc_values[i])
        yvals = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        axs[0, 2].plot(sorted_values, yvals, label=labels[i])
    axs[0, 2].set_title('Cumulative Distribution Functions')
    axs[0, 2].set_xlabel('Concentration')
    axs[0, 2].set_ylabel('Cumulative Probability')
    axs[0, 2].legend()
    
    # Row 2, Col 1: Mean comparison
    x_positions = np.arange(1, num_datasets + 1)
    axs[1, 0].errorbar(x=x_positions, y=means, yerr=error_values, fmt='o', capsize=5)
    axs[1, 0].set_xticks(x_positions)
    axs[1, 0].set_xticklabels(labels, rotation=45 if num_datasets > 3 else 0)
    axs[1, 0].set_title('Mean Comparison')
    axs[1, 0].set_ylabel('Value')
    
    # Row 2, Col 2: Box plot
    axs[1, 1].boxplot(pc_values, labels=labels)
    axs[1, 1].set_title('Box Plot Comparison')
    axs[1, 1].set_ylabel('Concentration')
    if num_datasets > 3:
        plt.setp(axs[1, 1].get_xticklabels(), rotation=45)
    
    # Row 2, Col 3: Q-Q plot for all datasets vs first dataset (reference)
    # For multiple datasets, we'll plot Q-Q plot for the second dataset only to maintain compatibility
    if num_datasets > 1:
        sc.probplot(pc_values[1], dist="norm", plot=axs[1, 2])
        axs[1, 2].set_title(f'Q-Q Plot - {labels[1]}')
    else:
        axs[1, 2].text(0.5, 0.5, "Q-Q Plot requires at least two datasets", 
                    horizontalalignment='center', verticalalignment='center')
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])
    
    # Row 3, Col 1: Scatter plots (compared to first dataset)
    if num_datasets > 1:
        for i in range(1, min(4, num_datasets)):  # Limit to 3 comparisons to avoid overcrowding
            axs[2, 0].scatter(pc_values[0], pc_values[i], alpha=0.5/num_datasets, 
                            label=f"{labels[0]} vs {labels[i]}")
        
        axs[2, 0].plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')
        axs[2, 0].set_title(f'Scatter Plot vs {labels[0]}')
        axs[2, 0].set_xlabel(labels[0])
        axs[2, 0].set_ylabel('Other Datasets')
        axs[2, 0].legend()
    else:
        axs[2, 0].text(0.5, 0.5, "Scatter plot requires at least two datasets", 
                     horizontalalignment='center', verticalalignment='center')
        axs[2, 0].set_xticks([])
        axs[2, 0].set_yticks([])
    
    # Row 3, Col 2: Residual plot (compared to first dataset)
    if num_datasets > 1:
        x = np.arange(len(pc_values[0]))
        for i in range(1, min(4, num_datasets)):  # Limit to 3 comparisons
            residuals = pc_values[i] - pc_values[0]
            axs[2, 1].scatter(x, residuals, alpha=0.5/num_datasets, 
                             label=f"{labels[i]} - {labels[0]}")
        
        axs[2, 1].axhline(y=0, color='r', linestyle='--')
        axs[2, 1].set_title('Residual Plot')
        axs[2, 1].set_xlabel('Index')
        axs[2, 1].set_ylabel('Residuals')
        axs[2, 1].legend()
    else:
        axs[2, 1].text(0.5, 0.5, "Residual plot requires at least two datasets", 
                     horizontalalignment='center', verticalalignment='center')
        axs[2, 1].set_xticks([])
        axs[2, 1].set_yticks([])
    
    # Row 3, Col 3: Autocorrelation and cross-correlation
    autocorrs = []
    for pc in pc_values:
        autocorr = scSignal.correlate(pc - np.mean(pc), pc - np.mean(pc), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr /= autocorr[0]
        autocorrs.append(autocorr)
    
    for i in range(num_datasets):
        axs[2, 2].plot(range(min(max_lag, len(autocorrs[i]))), 
                     autocorrs[i][:max_lag], label=f"Auto: {labels[i]}")
    
    # Add cross-correlations with first dataset
    if num_datasets > 1:
        for i in range(1, min(3, num_datasets)):  # Limit to 2 cross-correlations
            cross_corr = scSignal.correlate(
                pc_values[0] - np.mean(pc_values[0]), 
                pc_values[i] - np.mean(pc_values[i]), 
                mode='full'
            )
            # Normalize by the geometric mean of autocorrelations at lag 0
            norm_factor = np.sqrt(autocorrs[0][0] * autocorrs[i][0])
            cross_corr /= norm_factor
            
            mid_point = len(cross_corr) // 2
            axs[2, 2].plot(
                range(mid_point, min(mid_point + max_lag, len(cross_corr))), 
                cross_corr[mid_point:mid_point + max_lag], 
                linestyle='--',
                label=f"Cross: {labels[0]} vs {labels[i]}"
            )
    
    axs[2, 2].set_title('Autocorrelation and Cross-correlation')
    axs[2, 2].set_xlabel('Lag')
    axs[2, 2].set_ylabel('Correlation')
    axs[2, 2].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    plt.show()
    
    # Print out detailed statistics
    print("\nDetailed Comparison:")
    for stat_row in stats_rows:
        stat_name = stat_row[0]
        values = stat_row[1:]
        
        stats_line = f"{stat_name}: "
        for i, val in enumerate(values):
            if val == '-':
                stats_line += f"{labels[i]} = N/A, "
            else:
                stats_line += f"{labels[i]} = {val:.4f}, "
        
        # Remove trailing comma and space
        stats_line = stats_line[:-2]
        print(stats_line)
        
    return


def get_arrays_for_plotting(PointConcTsArray,labels,dimensionless="True"):
    num_datasets = len(PointConcTsArray) 
    pc_values = []
    wtref_values = []
    
    for i in range(num_datasets):
        wtref_values.append(PointConcTsArray[i].wtref)
        
        if dimensionless == "True":
            pc_values.append(PointConcTsArray[i].c_star)
        else:
            pc_values.append(PointConcTsArray[i].net_concentration)

    #Default labels if not provided
    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(num_datasets)]
    
    # Find global min and max for plotting
    max_val = max([np.max(pc) for pc in pc_values])
    min_val = min([np.min(pc) for pc in pc_values])

    return pc_values, wtref_values, labels,max_val,min_val

def get_stats_for_plotting(pc_values,wtref_values, num_datasets):
    stats_rows = []
    
    # Mean
    means = [np.mean(pc) for pc in pc_values]
    stats_rows.append(('Mean', *means))
    # Standard deviation
    stds = [np.std(pc) for pc in pc_values]
    stats_rows.append(('Std Dev', *stds))
    # Skewness
    skews = [sc.skew(pc) for pc in pc_values]
    stats_rows.append(('Skewness', *skews))
    # 95th percentile
    p95s = [np.percentile(pc, 95) for pc in pc_values]
    stats_rows.append(('Percentile 95', *p95s))
    # Turbulence intensity v
    ti_vs = [np.std(wtref)/np.mean(wtref) for wtref in wtref_values]
    stats_rows.append(('Turbulence Intensity v', *ti_vs))
    # Turbulence intensity c
    ti_cs = [np.std(pc)/np.mean(pc) for pc in pc_values]
    stats_rows.append(('Turbulence Intensity c', *ti_cs))
    
    # Calculate correlation statistics (comparing all to the first dataset)
    if num_datasets > 1:
        for i in range(1, num_datasets):
            pearson_r, pearson_p = sc.pearsonr(pc_values[0], pc_values[i])
            spearman_r, spearman_p = sc.spearmanr(pc_values[0], pc_values[i])
            rmse = np.sqrt(mean_squared_error(pc_values[0], pc_values[i]))
            
            # Add to stats with placeholder dashes for datasets not involved in comparison
            pearson_row = ['Pearson r (vs Dataset 1)'] + ['-'] * num_datasets
            pearson_row[i+1] = pearson_r
            stats_rows.append(tuple(pearson_row))
            
            spearman_row = ['Spearman rho (vs Dataset 1)'] + ['-'] * num_datasets
            spearman_row[i+1] = spearman_r
            stats_rows.append(tuple(spearman_row))
            
            rmse_row = ['RMSE (vs Dataset 1)'] + ['-'] * num_datasets
            rmse_row[i+1] = rmse
            stats_rows.append(tuple(rmse_row))

    return stats_rows
    


def compare_point_concentrations_3(PointConcTsArray, functionsForOverview=None, labels=None, max_lag=50, dimensionless="False", errorConc=5.0, errorType="absolute"):
    """
    Compare multiple point concentration datasets with autocorrelation.
    
    Parameters:
    -----------
    PointConcTsArray : list of arrays
        List of point concentration datasets to compare
    functionsForOverview : dict or list, optional
        Dictionary with plot types as keys and "true"/"false" as values,
        or list of plot types to include.
        Available plot types:
            - "Histogram" - Distribution histogram
            - "Pdf" - Probability Density Function
            - "Cdf" - Cumulative Distribution Function
            - "Means" - Mean comparison
            - "BoxPlot" - Box plot comparison
            - "QuantilPlot" - Q-Q plot (normality test)
            - "ScatterPlot" - Scatter plot vs first dataset
            - "ResidualPlot" - Residual plot vs first dataset
            - "Autocorrelation" - Autocorrelation and cross-correlation
        If None or ["all"], all plots will be shown
    labels : list, optional
        Labels for the datasets (default: ['Dataset 1', 'Dataset 2', ...])
    max_lag : int, optional
        Maximum lag for autocorrelation (default: 50)
    dimensionless : str, optional
        Use dimensionless concentration values (default: "False")
    errorConc : float, optional
        Error value for concentration to visualize uncertainty (default: 5.0)
    errorType : str, optional
        Type of error: "percentage" or "absolute" (default: "absolute")
    """
    pc_values,wtref_values,labels,max_val,min_val = get_arrays_for_plotting(PointConcTsArray,labels)
    num_datasets = len(PointConcTsArray)
    
    # Calculate basic stats for all datasets
    stats_rows = get_stats_for_plotting(pc_values,wtref_values,num_datasets)
   
    # Calculate error bars if provided
    if errorConc is not None:
        if errorType == "percentage":
            # Calculate percentage-based error
            error_values = np.array(stats_rows[0]) * (errorConc/100)
        else:
            # Use absolute error
            error_values = np.full(num_datasets, errorConc)
    else:
        error_values = None

    num_datasets = len(pc_values)
    # Create subplots, one for each dataset
    fig, axes = plt.subplots(num_datasets, 1, figsize=(12, 2*num_datasets), sharex=True)
    # Handle the case when there's only one dataset
    if num_datasets == 1:
        axes = [axes]
    # Plot each time series in its own subplot
    for i in range(num_datasets):
        axes[i].plot(pc_values[i])
        axes[i].set_ylabel("Concentration")
        axes[i].set_title(labels[i])
    # Add x-label only to the bottom subplot
    axes[-1].set_xlabel("Time Index")
    # Add an overall title
    plt.suptitle("Concentration time series comparison", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Determine which plots to show based on functionsForOverview
    all_plot_types = [
        "Histogram", "Pdf", "Cdf", "Means", "BoxPlot", 
        "PowerDensity", "QuantilPlot", "ScatterPlot", "ResidualPlot", "Autocorrelation",
    ]
    
    # Process functionsForOverview to determine which plots to show
    plots_to_show = []
    
    if functionsForOverview is None or functionsForOverview == ["all"]:
        # Show all plots if None or ["all"]
        plots_to_show = all_plot_types
    elif isinstance(functionsForOverview, dict):
        # Dictionary format: {"pdf": "true", "cdf": "false", ...}
        for plot_type in all_plot_types:
            key = plot_type.lower()
            if key in functionsForOverview and functionsForOverview[key].lower() == "true":
                plots_to_show.append(plot_type)
    elif isinstance(functionsForOverview, list):
        # List format: ["Histogram", "Pdf", ...]
        plots_to_show = [p for p in functionsForOverview if p in all_plot_types]
    
    # If no valid plots were specified, use all plots
    if not plots_to_show:
        plots_to_show = all_plot_types
    
    # Calculate number of rows needed (3 plots per row)
    num_plots = len(plots_to_show)
    num_rows = (num_plots + 2) // 3  # Ceiling division to get number of rows (3 plots per row)
    
    # Create figure with appropriate number of subplots
    fig, axs = plt.subplots(num_rows, 3, figsize=(18, 5*num_rows))
    fig.suptitle('Point Concentration Comparison', fontsize=16)
    
    # Handle case when there's only one row
    if num_rows == 1:
        axs = axs.reshape(1, -1)
    
    # Dictionary mapping plot types to functions that will create those plots
    plot_functions = {}
    
    # Histogram plot function
    def create_histogram(ax):
        for i in range(num_datasets):
            ax.hist(pc_values[i], bins='auto', alpha=0.7/num_datasets, label=labels[i])
        ax.set_title('Distribution Comparison')
        ax.set_xlabel('Concentration')
        ax.set_ylabel('Frequency')
        ax.grid(True)
        ax.legend()
    plot_functions["Histogram"] = create_histogram
    
    # PDF plot function
    def create_pdf(ax):
        from scipy.stats import gaussian_kde
        x = np.linspace(min_val, max_val, 200)
        for i in range(num_datasets):
            density = gaussian_kde(pc_values[i])
            ax.plot(x, density(x), label=labels[i])
        ax.set_title('Probability Density Functions')
        ax.set_xlabel('Concentration')
        ax.set_ylabel('Density')
        ax.grid(True)
        ax.legend()
    plot_functions["Pdf"] = create_pdf
    
    # CDF plot function
    def create_cdf(ax):
        for i in range(num_datasets):
            sorted_values = np.sort(pc_values[i])
            yvals = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            ax.plot(sorted_values, yvals, label=labels[i])
        ax.set_title('Cumulative Distribution Functions')
        ax.set_xlabel('Concentration')
        ax.set_ylabel('Cumulative Probability')
        ax.grid(True)
        ax.legend()
    plot_functions["Cdf"] = create_cdf
    
    # Mean comparison plot function
    def create_means(ax):
        x_positions = np.arange(1, num_datasets + 1)
        means = stats_rows[0][1:]
        print(x_positions)
        print(len(x_positions))
        print(len(stats_rows[0]))
        ax.errorbar(x=x_positions, y=means, yerr=error_values, fmt='o', capsize=5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45 if num_datasets > 3 else 0)
        ax.set_title('Mean Comparison')
        ax.set_ylabel('Value')
        ax.grid(True)
    plot_functions["Means"] = create_means
    
    # Box plot function
    def create_boxplot(ax):
        ax.boxplot(pc_values, labels=labels)
        ax.set_title('Box Plot Comparison')
        ax.set_ylabel('Concentration')
        ax.grid(True)
        if num_datasets > 3:
            plt.setp(ax.get_xticklabels(), rotation=45)
    plot_functions["BoxPlot"] = create_boxplot
    
    # Q-Q plot function
    def create_quantilplot(ax):
        if num_datasets > 1:
            sc.probplot(pc_values[1], dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot - {labels[1]}')
        else:
            ax.text(0.5, 0.5, "Q-Q Plot requires at least two datasets", 
                    horizontalalignment='center', verticalalignment='center')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True)
    plot_functions["QuantilPlot"] = create_quantilplot
    
    # Scatter plot function
    def create_scatterplot(ax):
        if num_datasets > 1:
            for i in range(1, min(4, num_datasets)):  # Limit to 3 comparisons to avoid overcrowding
                ax.scatter(pc_values[0], pc_values[i], alpha=0.5/num_datasets, 
                          label=f"{labels[0]} vs {labels[i]}")
            
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')
            ax.set_title(f'Scatter Plot vs {labels[0]}')
            ax.set_xlabel(labels[0])
            ax.set_ylabel('Other Datasets')
            ax.grid(True)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Scatter plot requires at least two datasets", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True)
    plot_functions["ScatterPlot"] = create_scatterplot
    
    # Residual plot function
    def create_residualplot(ax):
        if num_datasets > 1:
            x = np.arange(len(pc_values[0]))
            for i in range(1, min(4, num_datasets)):  # Limit to 3 comparisons
                residuals = pc_values[i] - pc_values[0]
                ax.scatter(x, residuals, alpha=0.5/num_datasets, 
                         label=f"{labels[i]} - {labels[0]}")
            
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_title('Residual Plot')
            ax.set_xlabel('Index')
            ax.set_ylabel('Residuals')
            ax.grid(True)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Residual plot requires at least two datasets", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True)
    plot_functions["ResidualPlot"] = create_residualplot
    
    # Autocorrelation plot function
    def create_autocorrelation(ax):
        autocorrs = []
        for pc in pc_values:
            autocorr = scSignal.correlate(pc - np.mean(pc), pc - np.mean(pc), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr /= autocorr[0]
            autocorrs.append(autocorr)
        
        for i in range(num_datasets):
            ax.plot(range(min(max_lag, len(autocorrs[i]))), 
                     autocorrs[i][:max_lag], label=f"Auto: {labels[i]}")
            ax.grid(True)
        
        # Add cross-correlations with first dataset
        if num_datasets > 1:
            for i in range(1, min(3, num_datasets)):  # Limit to 2 cross-correlations
                cross_corr = scSignal.correlate(
                    pc_values[0] - np.mean(pc_values[0]), 
                    pc_values[i] - np.mean(pc_values[i]), 
                    mode='full'
                )
                # Normalize by the geometric mean of autocorrelations at lag 0
                norm_factor = np.sqrt(autocorrs[0][0] * autocorrs[i][0])
                cross_corr /= norm_factor
                
                mid_point = len(cross_corr) // 2
                ax.plot(
                    range(mid_point, min(mid_point + max_lag, len(cross_corr))), 
                    cross_corr[mid_point:mid_point + max_lag], 
                    linestyle='--',
                    label=f"Cross: {labels[0]} vs {labels[i]}"
                )
        
        ax.set_title('Autocorrelation and Cross-correlation')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Correlation')
        ax.legend()
        ax.grid(True)
    plot_functions["Autocorrelation"] = create_autocorrelation


    def create_powerDensityPlot(ax): 
        times = []
        plot=True
        for i in range(num_datasets):
            times.append(PointConcTsArray[i].time)

        # Sampling frequency
        fs = [1.0 / np.mean(np.diff(time)) for time in times]
        # Spectral analysis, using Welch's PSD
        freqs=[]
        psd=[]

        for i in range(num_datasets):
            freq_i, psd_i = scSignal.welch(pc_values[i], fs=fs[i])
            freqs.append(freq_i)
            psd.append(psd_i)

        print(freqs)
        print(psd)
    
        # Key metrics
        peak_freq = []
        high_freq_power = []
        high_freq_power = []
        total_power = [] 
        for i in range(num_datasets):
            peak_freq.append( freqs[i][np.argmax(psd[i])] )
            high_freq_power.append(np.sum(psd[i][freqs[i] >= 1.0]))
            total_power.append(np.sum(psd[i]))

        fluctuation_intensity = []
        for i in range(num_datasets):
            fluctuation_intensity.append ( high_freq_power[i] / total_power[i] )

            
        # Optional plotting
        if plot:
            for i in range(num_datasets):
                ax.semilogy(freqs[i], psd[i])
            ax.set_title('Power Spectral Density')
            #ax.tight_layout()
            ax.grid(True)
            ax.set_xlabel("frequency[Hz]")
            ax.set_ylabel("Power Spectral Density[(ppm)²/Hz]")
            ax.legend()
    plot_functions["PowerDensity"] = create_powerDensityPlot

    
    
    # Create plots based on the plots_to_show list
    plot_idx = 0
    for row in range(num_rows):
        for col in range(3):
            if plot_idx < len(plots_to_show):
                # Get the plot type and create it
                plot_type = plots_to_show[plot_idx]
                plot_functions[plot_type](axs[row, col])
                plot_idx += 1
            else:
                # Hide unused subplots
                axs[row, col].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    plt.show()
    
    # Print out detailed statistics
    print("\nDetailed Comparison:")
    for stat_row in stats_rows:
        stat_name = stat_row[0]
        values = stat_row[1:]
        
        stats_line = f"{stat_name}: "
        for i, val in enumerate(values):
            if val == '-':
                stats_line += f"{labels[i]} = N/A, "
            else:
                stats_line += f"{labels[i]} = {val:.4f}, "
        
        # Remove trailing comma and space
        stats_line = stats_line[:-2]
        print(stats_line)
        
    return


 # Histogram plot function
def create_histogram(PointConcTsArray,dimensionless="False",labels=None,xLabel=None,yLabel=None,xAchse=None,yAchse=None):
    num_datasets = len(PointConcTsArray)
    pc_values,wtref_values,labels,max_val,min_val = get_arrays_for_plotting(PointConcTsArray,labels=labels, dimensionless=dimensionless)

    for i in range(num_datasets):
        plt.hist(pc_values[i], bins='auto', alpha=0.7/num_datasets, label=labels[i])
    plt.title('Distribution Comparison')
    plt.xlabel(xLabel if xLabel is not None else 'Concentration')
    plt.ylabel(yLabel if yLabel is not None else 'Frequency')
    plt.grid(True)
    plt.legend()
    plt.show()
    return
  
    
    # PDF plot function
def create_pdf(PointConcTsArray,dimensionless="False",labels=None,xLabel=None,yLabel=None,xAchse=None,yAchse=None):
    num_datasets = len(PointConcTsArray)
    pc_values,wtref_values,labels,max_val,min_val = get_arrays_for_plotting(PointConcTsArray,labels=labels, dimensionless=dimensionless)
    

    from scipy.stats import gaussian_kde
    x = np.linspace(min_val, max_val, 200)
    for i in range(num_datasets):
        density = gaussian_kde(pc_values[i])
        plt.plot(x, density(x), label=labels[i])
    plt.title('Probability Density Functions')
    plt.xlabel(xLabel if xLabel is not None else 'Concentration')
    plt.ylabel(yLabel if yLabel is not None else 'Density')
    plt.legend()
    plt.grid()
    plt.show()
    return
    
# CDF plot function
def create_cdf(PointConcTsArray,dimensionless="False",labels=None,xLabel=None,yLabel=None,xAchse=None,yAchse=None):
    num_datasets = len(PointConcTsArray)
    pc_values,wtref_values,labels,max_val,min_val = get_arrays_for_plotting(PointConcTsArray,labels=labels, dimensionless=dimensionless)

    plt.figure(figsize=(13, 9))
    for i in range(num_datasets):
        sorted_values = np.sort(pc_values[i])
        yvals = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        plt.plot(sorted_values, yvals, label=labels[i])
    plt.title('Cumulative Distribution Functions')
    plt.xlabel(xLabel if xLabel is not None else 'Concentration')
    plt.ylabel(yLabel if yLabel is not None else 'Cumulative Probability')
    plt.grid()
    plt.legend()
    return

# Mean comparison plot function
def create_means(PointConcTsArray,error_values=None,errorType=None,dimensionless="False",labels=None,xLabel=None,yLabel=None,xAchse=None,yAchse=None):
    num_datasets = len(PointConcTsArray)

    pc_values,wtref_value,labels,max_val,min_val = get_arrays_for_plotting(PointConcTsArray,labels=labels, dimensionless=dimensionless)
    means = [np.mean(pc) for pc in pc_values]

    # Calculate error bars if provided errorConc and errorType
    if error_values is not None:
        if errorType == "percentage":
            # Calculate percentage-based error
            error_values = np.array(means) * (error_values/100)
        elif errorType == "absolute":
            # Use absolute error
            error_values = np.full(num_datasets, error_values)
    else:
        if errorType == "std":
            error_value = [np.std(pc) for pc in pc_values]
    
    x_positions = np.arange(1, num_datasets + 1)
    plt.figure(figsize=(13, 9))
    if error_values is not None:
        plt.errorbar(x=x_positions, y=means, yerr=error_values, fmt='o', capsize=5)
    else:
        plt.plot(x_positions, means, 'o')
    plt.xticks(x_positions,labels) #, rotation=45 if num_datasets > 3 else 0)
    plt.title('Mean Comparison')
    plt.xlabel(xLabel if xLabel is not None else 'Value')
    plt.ylabel(yLabel if yLabel is not None else 'Datasets')
    if(yAchse!=None):
        plt.ylim(yAchse)
    plt.grid()
    return
   
    
# Box plot function
def create_boxplot(PointConcTsArray,dimensionless="False",labels=None,xLabel=None,yLabel=None,xAchse=None,yAchse=None):
    num_datasets = len(PointConcTsArray)
    pc_values,wtref_values,labels,max_val,min_val = get_arrays_for_plotting(PointConcTsArray,labels=labels, dimensionless=dimensionless)
    
    plt.figure(figsize=(13, 9))
    plt.boxplot(pc_values, labels=labels)
    plt.title('Box Plot Comparison')
    plt.xlabel(xLabel if xLabel is not None else 'Datasets')
    plt.ylabel(yLabel if yLabel is not None else 'Concentration')
    plt.grid(True)
    if num_datasets > 3:
        plt.setp(plt.get_xticklabels(), rotation=45)
    return




# Q-Q plot function
def create_quantilplot(PointConcTsArray,dimensionless="False",labels=None,xLabel=None,yLabel=None,xAchse=None,yAchse=None):
    num_datasets = len(PointConcTsArray)
    pc_values,wtref_values,labels,max_val,min_val = get_arrays_for_plotting(PointConcTsArray,labels=labels, dimensionless=dimensionless)
    

    if num_datasets > 1:
        sc.probplot(pc_values[1], dist="norm", plot=plt)
        plt.figure(figsize=(13, 9))
        #plt.title(f'Q-Q Plot - {labels[1]}')
    else:
        plt.text(0.5, 0.5, "Q-Q Plot requires at least two datasets", 
                horizontalalignment='center', verticalalignment='center')
        plt.xticks([])
        plt.yticks([])
        plt.grid()
    return


# Scatter plot function
def create_scatterplot(PointConcTsArray,dimensionless="False",labels=None,xLabel=None,yLabel=None,xAchse=None,yAchse=None):
    num_datasets = len(PointConcTsArray)
    pc_values,wtref_values,labels,max_val,min_val = get_arrays_for_plotting(PointConcTsArray,labels=labels, dimensionless=dimensionless)
    

    if num_datasets > 1:
        for i in range(1, min(4, num_datasets)):  # Limit to 3 comparisons to avoid overcrowding
            plt.scatter(pc_values[0], pc_values[i], alpha=0.5/num_datasets, 
                        label=f"{labels[0]} vs {labels[i]}")
            
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')
        plt.title(f'Scatter Plot vs {labels[0]}')
        plt.xlabel(xLabel if xLabel is not None else labels[0])
        plt.ylabel(yLabel if yLabel is not None else 'Other Datasets')
        plt.legend()
        plt.show()
    else:
        plt.text(0.5, 0.5, "Scatter plot requires at least two datasets", 
                horizontalalignment='center', verticalalignment='center')
        plt.set_xticks([])
        plt.set_yticks([])
        plt.show()

    return
   
    # Residual plot function
def create_residualplot(PointConcTsArray,dimensionless="False",labels=None,xLabel=None,yLabel=None,xAchse=None,yAchse=None):
    num_datasets = len(PointConcTsArray)
    pc_values,wtref_values,labels,max_val,min_val = get_arrays_for_plotting(PointConcTsArray,labels=labels, dimensionless=dimensionless)
    

    if num_datasets > 1:
        x = np.arange(len(pc_values[0]))
        for i in range(1, min(4, num_datasets)):  # Limit to 3 comparisons
            residuals = pc_values[i] - pc_values[0]
            plt.scatter(x, residuals, alpha=0.5/num_datasets, 
                        label=f"{labels[i]} - {labels[0]}")
        
        plt.axhline(y=0, color='r', linestyle='--')
        plt.set_title('Residual Plot')
        plt.set_xlabel('Index')
        plt.set_ylabel('Residuals')
        plt.legend()
    else:
        plt.text(0.5, 0.5, "Residual plot requires at least two datasets", 
                horizontalalignment='center', verticalalignment='center')
        plt.set_xticks([])
        plt.set_yticks([])
    return


    # Autocorrelation plot function
def create_autocorrelation(max_lag,PointConcTsArray,dimensionless="False",labels=None,xLabel=None,yLabel=None,xAchse=None,yAchse=None):
    num_datasets = len(PointConcTsArray)
    pc_values,wtref_values,labels,max_val,min_val = get_arrays_for_plotting(PointConcTsArray,labels=labels, dimensionless=dimensionless)
    

    autocorrs = []
    for pc in pc_values:
        autocorr = scSignal.correlate(pc - np.mean(pc), pc - np.mean(pc), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr /= autocorr[0]
        autocorrs.append(autocorr)
    
    for i in range(num_datasets):
        plt.plot(range(min(max_lag, len(autocorrs[i]))), 
                    autocorrs[i][:max_lag], label=f"Auto: {labels[i]}")
    
    # Add cross-correlations with first dataset
    if num_datasets > 1:
        for i in range(1, min(3, num_datasets)):  # Limit to 2 cross-correlations
            cross_corr = scSignal.correlate(
                pc_values[0] - np.mean(pc_values[0]), 
                pc_values[i] - np.mean(pc_values[i]), 
                mode='full'
            )
            # Normalize by the geometric mean of autocorrelations at lag 0
            norm_factor = np.sqrt(autocorrs[0][0] * autocorrs[i][0])
            cross_corr /= norm_factor
            
            mid_point = len(cross_corr) // 2
            plt.plot(
                range(mid_point, min(mid_point + max_lag, len(cross_corr))), 
                cross_corr[mid_point:mid_point + max_lag], 
                linestyle='--',
                label=f"Cross: {labels[0]} vs {labels[i]}"
            )
    
    plt.set_title('Autocorrelation and Cross-correlation')
    plt.set_xlabel('Lag')
    plt.set_ylabel('Correlation')
    plt.legend()


def powerDensityPlot(PointConcTsArray,dimensionless="False",plot=True,labels=None,xLabel=None,yLabel=None,xAchse=None,yAchse=None):
    num_datasets = len(PointConcTsArray)
    pc_values,wtref_values,labels,max_val,min_val = get_arrays_for_plotting(PointConcTsArray,labels=labels, dimensionless=dimensionless)
    
    times = []
    for i in range(num_datasets):
        times.append(PointConcTsArray[i].time)

    # Sampling frequency
    fs = [1.0 / np.mean(np.diff(time)) for time in times]
    # Spectral analysis, using Welch's PSD
    freqs=[]
    psd=[]

    for i in range(num_datasets):
        freq_i, psd_i = scSignal.welch(pc_values[i], fs=fs[i])
        freqs.append(freq_i)
        psd.append(psd_i)

    print(freqs)
    print(psd)
 
    # Key metrics
    peak_freq = []
    high_freq_power = []
    high_freq_power = []
    total_power = [] 
    for i in range(num_datasets):
        peak_freq.append( freqs[i][np.argmax(psd[i])] )
        high_freq_power.append(np.sum(psd[i][freqs[i] >= 1.0]))
        total_power.append(np.sum(psd[i]))

    fluctuation_intensity = []
    for i in range(num_datasets):
        fluctuation_intensity.append ( high_freq_power[i] / total_power[i] )

    # ... existing code ...
    # (no change to plotting logic)
    # ... existing code ...
    return plt.gcf()
   
"""
    # Histogram comparison
    axs[0, 0].hist(pc1, bins='auto', alpha=0.5, label=labels[0])
    axs[0, 0].hist(pc2, bins='auto', alpha=0.5, label=labels[1])
    axs[0, 0].set_title('Distribution Comparison')
    axs[0, 0].set_xlabel('Concentration')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].legend()

    axs[0, 1].bar([f'{labels[0]} Mean', f'{labels[1]} Mean'], 
                  [stats_data[0][1], stats_data[0][2]])
    axs[0, 1].set_title('Mean Comparison')
    axs[0, 1].set_ylabel('Value')
    
    # Box plot
    axs[0, 2].boxplot([pc1, pc2], labels=labels)
    axs[0, 2].set_title('Box Plot Comparison')
    axs[0, 2].set_ylabel('Concentration')
    
    # Autocorrelation for first dataset
    autocorr1 = scSignal.correlate(pc1 - np.mean(pc1), pc1 - np.mean(pc1), mode='full')
    autocorr1 = autocorr1[len(autocorr1)//2:]
    autocorr1 /= autocorr1[0]
    
    # Autocorrelation for second dataset
    autocorr2 = scSignal.correlate(pc2 - np.mean(pc2), pc2 - np.mean(pc2), mode='full')
    autocorr2 = autocorr2[len(autocorr2)//2:]
    autocorr2 /= autocorr2[0]
    
    # Autocorrelation plot
    axs[1, 0].plot(range(min(max_lag, len(autocorr1))), autocorr1[:max_lag], label=labels[0])
    axs[1, 0].plot(range(min(max_lag, len(autocorr2))), autocorr2[:max_lag], label=labels[1])
    axs[1, 0].set_title('Autocorrelation')
    axs[1, 0].set_xlabel('Lag')
    axs[1, 0].set_ylabel('Autocorrelation')
    axs[1, 0].legend()


    axs[1,1].scatter(pc1,pc2)#,#labels=labels)
    #axs[1,1].scatter(pc2)
    axs[1, 1].set_title(f'Scatter Plot - {labels[1]}')
    # Q-Q plot
    #stats.probplot(pc1, dist="norm", plot=axs[1, 1])
    #axs[1, 1].set_title(f'Q-Q Plot - {labels[0]}')
    
    #Two in one Q-Q plot
    sc.probplot([pc1,pc2], dist="norm", plot=axs[1, 2])
    #stats.probplot(pc2, dist="norm", plot=axs[1, 2])
    
    # Q-Q plot
    #sc.probplot(pc1, dist="norm", plot=axs[1, 1])
    #axs[1, 1].set_title(f'Q-Q Plot - {labels[0]}')
    
    # Second Q-Q plot
    sc.probplot(pc2, dist="norm", plot=axs[1, 2])
    axs[1, 2].set_title(f'Q-Q Plot - {labels[1]}')
    
    plt.tight_layout()
    plt.show()
    
    # Print out detailed statistics
    print("\nDetailed Comparison:")
    for stat, val1, val2 in stats_data:
        print(f"{stat}: {labels[0]} = {val1:.4f}, {labels[1]} = {val2:.4f}")

    return
"""
