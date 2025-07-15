"""
logic/plot_manager.py
Plotting logic for the application.
"""
from typing import Any, List, Dict, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from logic.plot_config import PlotConfigManager, PlotType, PlotConfiguration

class PlotManager:
    """
    Handles creation of plots from data objects using configurable parameters.
    """
    
    def __init__(self, config_manager: Optional[PlotConfigManager] = None):
        self.config_manager = config_manager or PlotConfigManager()
    
    def create_plot(self, data: List[Any], plot_type: str, 
                   config_overrides: Optional[Dict[str, Any]] = None) -> plt.Figure:
        """
        Create a plot of the specified type from the data.
        Args:
            data: List of data objects to plot.
            plot_type: Type of plot to create (e.g., 'histogram', 'pdf').
            config_overrides: Optional configuration overrides.
        Returns:
            A matplotlib Figure object.
        """
        try:
            plot_type_enum = PlotType(plot_type)
        except ValueError:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        config = self.config_manager.get_configuration(plot_type_enum)
        if not config:
            raise ValueError(f"No configuration found for plot type: {plot_type}")
        
        # Apply configuration overrides
        if config_overrides:
            config = self._apply_config_overrides(config, config_overrides)
        
        # Convert data to DataFrame if needed
        df = self._prepare_data(data)
        
        # Create the plot based on type
        fig = self._create_plot_by_type(plot_type_enum, df, config)
        
        return fig
    
    def _apply_config_overrides(self, config: PlotConfiguration, 
                               overrides: Dict[str, Any]) -> PlotConfiguration:
        """Apply configuration overrides to the base configuration."""
        # Create a copy to avoid modifying the original
        import copy
        config_copy = copy.deepcopy(config)
        
        # Apply basic overrides
        if "title" in overrides:
            config_copy.title = overrides["title"]
        if "x_label" in overrides:
            config_copy.x_label = overrides["x_label"]
        if "y_label" in overrides:
            config_copy.y_label = overrides["y_label"]
        
        # Apply parameter overrides
        if "parameters" in overrides:
            for param_name, value in overrides["parameters"].items():
                for param in config_copy.parameters:
                    if param.name == param_name:
                        param.default_value = value
                        break
        
        return config_copy
    
    def _prepare_data(self, data: List[Any]) -> pd.DataFrame:
        """Convert data objects to a pandas DataFrame."""
        print(f"[DEBUG] _prepare_data called with {len(data)} items")
        print(f"[DEBUG] Data type: {type(data[0]) if data else 'No data'}")
        
        if not data:
            print("[DEBUG] No data provided")
            return pd.DataFrame()
        
        # Handle concentration objects (from windtunnel data) - check this FIRST
        # Check if these are concentration objects with time series data
        if hasattr(data[0], 'net_concentration'):
            print("[DEBUG] Data appears to be concentration objects")
            all_data = []
            for i, conc_obj in enumerate(data):
                # Get the time series data - convert pandas Series to numpy arrays
                net_conc = getattr(conc_obj, 'net_concentration', [])
                full_scale_conc = getattr(conc_obj, 'full_scale_concentration', [])
                c_star_data = getattr(conc_obj, 'c_star', [])
                
                # Convert pandas Series to numpy arrays if needed
                if hasattr(net_conc, 'values'):
                    net_conc = net_conc.values
                if hasattr(full_scale_conc, 'values'):
                    full_scale_conc = full_scale_conc.values
                if hasattr(c_star_data, 'values'):
                    c_star_data = c_star_data.values
                
                # Get coordinate data (these are single values, not arrays)
                x_fs = getattr(conc_obj, 'x', 0.0)
                y_fs = getattr(conc_obj, 'y', 0.0)
                z_fs = getattr(conc_obj, 'z', 0.0)
                
                # Use filename if available, otherwise use index
                filename = getattr(conc_obj, 'filename', f'dataset_{i}')
                
                print(f"[DEBUG] Processing {filename}: net_conc length={len(net_conc)}, c_star length={len(c_star_data)}")
                print(f"[DEBUG] net_conc type: {type(net_conc)}, sample: {net_conc[:3] if len(net_conc) > 0 else 'No data'}")
                
                # Create a data entry for each time point
                if len(net_conc) > 0:
                    for j in range(len(net_conc)):
                        data_dict = {
                            'filename': filename,
                            'x_fs': x_fs,
                            'y_fs': y_fs,
                            'z_fs': z_fs,
                            'net_concentration': float(net_conc[j]) if j < len(net_conc) else 0.0,
                            'full_scale_concentration': float(full_scale_conc[j]) if j < len(full_scale_conc) else 0.0,
                            'c_star': float(c_star_data[j]) if j < len(c_star_data) else 0.0
                        }
                        all_data.append(data_dict)
                else:
                    # Fallback if no time series data
                    data_dict = {
                        'filename': filename,
                        'x_fs': x_fs,
                        'y_fs': y_fs,
                        'z_fs': z_fs,
                        'net_concentration': 0.0,
                        'full_scale_concentration': 0.0,
                        'c_star': 0.0
                    }
                    all_data.append(data_dict)
            
            # Create DataFrame from all data points
            print(f"[DEBUG] Created {len(all_data)} data entries")
            if all_data:
                df = pd.DataFrame(all_data)
                print(f"[DEBUG] DataFrame shape: {df.shape}")
                print(f"[DEBUG] DataFrame columns: {list(df.columns)}")
                print(f"[DEBUG] DataFrame sample:\n{df.head()}")
                return df
            else:
                print("[DEBUG] No data entries created")
                return pd.DataFrame()
        
        # If data is already a DataFrame (and not a concentration object), return it
        if isinstance(data[0], pd.DataFrame):
            print("[DEBUG] Data is already DataFrame")
            return pd.concat(data, ignore_index=True)
        
        # If data is a list of dictionaries, convert to DataFrame
        if isinstance(data[0], dict):
            print("[DEBUG] Data is list of dictionaries")
            # Handle list of dictionaries (each dict represents a dataset)
            all_data = []
            for i, data_dict in enumerate(data):
                # Convert numpy arrays to lists for DataFrame
                processed_dict = {}
                for key, value in data_dict.items():
                    if isinstance(value, np.ndarray):
                        processed_dict[key] = value.tolist()
                    else:
                        processed_dict[key] = value
                
                # Create DataFrame from this dataset with an index
                df_part = pd.DataFrame([processed_dict], index=[i])
                all_data.append(df_part)
            
            # Concatenate all datasets
            if all_data:
                return pd.concat(all_data, ignore_index=True)
            else:
                return pd.DataFrame()
        
        # For other data types, try to convert
        try:
            print("[DEBUG] Trying generic DataFrame conversion")
            return pd.DataFrame(data)
        except Exception as e:
            print(f"[DEBUG] Generic conversion failed: {e}")
            # Fallback: create empty DataFrame
            return pd.DataFrame()
    
    def _create_plot_by_type(self, plot_type: PlotType, df: pd.DataFrame, 
                           config: PlotConfiguration) -> plt.Figure:
        """Create a specific type of plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == PlotType.HISTOGRAM:
            self._create_histogram(ax, df, config)
        elif plot_type == PlotType.SCATTER_PLOT:
            self._create_scatter_plot(ax, df, config)
        elif plot_type == PlotType.BOX_PLOT:
            self._create_box_plot(ax, df, config)
        elif plot_type == PlotType.PDF:
            self._create_pdf_plot(ax, df, config)
        elif plot_type == PlotType.CDF:
            self._create_cdf_plot(ax, df, config)
        elif plot_type == PlotType.MEANS:
            self._create_means_plot(ax, df, config)
        elif plot_type == PlotType.POWER_DENSITY:
            self._create_power_density_plot(ax, df, config)
        else:
            # Default: create a simple line plot
            self._create_default_plot(ax, df, config)
        
        # Apply common formatting
        ax.set_title(config.title)
        ax.set_xlabel(config.x_label)
        ax.set_ylabel(config.y_label)
        ax.grid(True, alpha=0.3)
        
        # Apply legend font size if specified
        params = self._get_parameter_dict(config)
        legend_fontsize = params.get("legend_fontsize", 10)
        if ax.get_legend():
            ax.legend(fontsize=legend_fontsize)
        
        plt.tight_layout()
        return fig
    
    def _create_histogram(self, ax, df: pd.DataFrame, config: PlotConfiguration):
        """Create a histogram plot with support for multiple datasets."""
        data_col = self._get_available_column(df, config.data_columns)
        if data_col is None:
            return
        
        # Get parameters
        params = self._get_parameter_dict(config)
        bins = params.get("bins", 20)
        density = params.get("density", True)
        color = params.get("color", "blue")
        alpha = params.get("alpha", 0.7)
        dimensionless = params.get("dimensionless", "False")
        xAchse = params.get("xAchse", "")
        yAchse = params.get("yAchse", "")
        
        # Use dimensionless data if requested
        if dimensionless == "True" and "c_star" in df.columns:
            data_col = "c_star"
        
        # Group by filename to get individual datasets
        if 'filename' in df.columns:
            grouped_data = df.groupby('filename')
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
            
            for i, (filename, group) in enumerate(grouped_data):
                dataset_data = group[data_col].dropna()
                if len(dataset_data) == 0:
                    continue
                
                # Choose color for this dataset
                hist_color = colors[i % len(colors)]
                
                # Create histogram
                ax.hist(dataset_data, bins=bins, density=density, alpha=alpha, 
                       color=hist_color, label=filename, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Concentration [ppmV]')
            ax.set_ylabel('Density' if density else 'Count')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            # Fallback to single dataset approach
            data = df[data_col].dropna()
            if len(data) == 0:
                return
            
            # Create histogram
            ax.hist(data, bins=bins, density=density, alpha=alpha, 
                   color=color, label='Histogram', edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Concentration [ppmV]')
            ax.set_ylabel('Density' if density else 'Count')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Set axis ranges if specified
        if xAchse and xAchse.strip():
            try:
                x_range = eval(xAchse)
                if isinstance(x_range, tuple) and len(x_range) == 2:
                    ax.set_xlim(x_range)
            except:
                pass
        
        if yAchse and yAchse.strip():
            try:
                y_range = eval(yAchse)
                if isinstance(y_range, tuple) and len(y_range) == 2:
                    ax.set_ylim(y_range)
            except:
                pass
    
    def _create_scatter_plot(self, ax, df: pd.DataFrame, config: PlotConfiguration):
        """Create a scatter plot with support for multiple datasets."""
        # Get parameters
        params = self._get_parameter_dict(config)
        x_column = params.get("x_column", "x_fs")
        y_column = params.get("y_column", "net_concentration")
        point_size = params.get("point_size", 50)
        color_by = params.get("color_by", "none")
        dimensionless = params.get("dimensionless", "False")
        alpha = params.get("alpha", 0.7)
        marker_style = params.get("marker_style", "o")
        xAchse = params.get("xAchse", "")
        yAchse = params.get("yAchse", "")
        
        # Use dimensionless data if requested
        if dimensionless == "True" and "c_star" in df.columns:
            y_column = "c_star"
        
        # Check if required columns exist
        if x_column not in df.columns or y_column not in df.columns:
            print(f"[WARNING] Required columns {x_column} or {y_column} not found in data")
            return
        
        # Group by filename to get individual datasets
        if 'filename' in df.columns:
            grouped_data = df.groupby('filename')
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
            markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'H', '+', 'x']
            
            for i, (filename, group) in enumerate(grouped_data):
                dataset_data = group[[x_column, y_column]].dropna()
                if len(dataset_data) == 0:
                    continue
                
                # Choose color and marker for this dataset
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                
                # Create scatter plot
                ax.scatter(dataset_data[x_column], dataset_data[y_column], 
                          s=point_size, c=color, alpha=alpha, marker=marker, 
                          label=filename, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel(x_column.replace('_', ' ').title())
            ax.set_ylabel(y_column.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            # Fallback to single dataset approach
            data = df[[x_column, y_column]].dropna()
            if len(data) == 0:
                return
            
            # Create scatter plot
            ax.scatter(data[x_column], data[y_column], s=point_size, 
                      alpha=alpha, marker=marker_style, label='Data', 
                      edgecolors='black', linewidth=0.5)
            ax.set_xlabel(x_column.replace('_', ' ').title())
            ax.set_ylabel(y_column.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Set axis ranges if specified
        if xAchse and xAchse.strip():
            try:
                x_range = eval(xAchse)
                if isinstance(x_range, tuple) and len(x_range) == 2:
                    ax.set_xlim(x_range)
            except:
                pass
        
        if yAchse and yAchse.strip():
            try:
                y_range = eval(yAchse)
                if isinstance(y_range, tuple) and len(y_range) == 2:
                    ax.set_ylim(y_range)
            except:
                pass
    
    def _create_box_plot(self, ax, df: pd.DataFrame, config: PlotConfiguration):
        """Create a box plot."""
        params = self._get_parameter_dict(config)
        
        group_by = params.get("group_by", "filename")
        show_outliers = params.get("show_outliers", True)
        notch = params.get("notch", False)
        color_scheme = params.get("color_scheme", "default")
        dimensionless = params.get("dimensionless", "False")
        box_width = params.get("box_width", 0.8)
        flier_size = params.get("flier_size", 3.0)
        xAchse = params.get("xAchse", "")
        yAchse = params.get("yAchse", "")
        
        # Use dimensionless data if requested
        if dimensionless == "True" and "c_star" in df.columns:
            data_col = "c_star"
        else:
            data_col = self._get_available_column(df, config.data_columns)
        
        # Ensure boolean parameters are properly handled
        if isinstance(show_outliers, (list, np.ndarray)):
            show_outliers = bool(show_outliers[0]) if len(show_outliers) > 0 else True
        if isinstance(notch, (list, np.ndarray)):
            notch = bool(notch[0]) if len(notch) > 0 else False
        
        if data_col is None or group_by not in df.columns:
            return
        
        # Group data
        grouped_data = [group[data_col].dropna() for name, group in df.groupby(group_by)]
        labels = list(df[group_by].unique())
        
        if len(grouped_data) == 0:
            return
        
        if color_scheme == "default":
            colors = plt.cm.Set3(np.linspace(0, 1, len(grouped_data)))
        else:
            colors = plt.cm.get_cmap(color_scheme)(np.linspace(0, 1, len(grouped_data)))
        
        bp = ax.boxplot(grouped_data, labels=labels, patch_artist=True, 
                       showfliers=show_outliers, notch=notch, widths=box_width, flierprops={'markersize': flier_size})
        
        # Apply colors
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Set axis ranges if specified
        if xAchse and xAchse.strip():
            try:
                x_range = eval(xAchse)
                if isinstance(x_range, tuple) and len(x_range) == 2:
                    ax.set_xlim(x_range)
            except:
                pass
        
        if yAchse and yAchse.strip():
            try:
                y_range = eval(yAchse)
                if isinstance(y_range, tuple) and len(y_range) == 2:
                    ax.set_ylim(y_range)
            except:
                pass
    
    def _create_pdf_plot(self, ax, df: pd.DataFrame, config: PlotConfiguration):
        """Create a PDF plot with support for multiple datasets."""
        data_col = self._get_available_column(df, config.data_columns)
        if data_col is None:
            return
        
        # Get parameters
        params = self._get_parameter_dict(config)
        dimensionless = params.get("dimensionless", "False")
        line_color = params.get("color", "blue")
        line_style = params.get("line_style", "solid")
        line_width = params.get("line_width", 2.0)
        xAchse = params.get("xAchse", "")
        yAchse = params.get("yAchse", "")
        
        # Use dimensionless data if requested
        if dimensionless == "True" and "c_star" in df.columns:
            data_col = "c_star"
        
        # Group by filename to get individual datasets
        if 'filename' in df.columns:
            grouped_data = df.groupby('filename')
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
            
            for i, (filename, group) in enumerate(grouped_data):
                dataset_data = group[data_col].dropna()
                if len(dataset_data) == 0:
                    continue
                
                # Choose color for this dataset
                color = colors[i % len(colors)]
                
                # Calculate PDF
                try:
                    from scipy import stats
                    # Fit a kernel density estimate
                    kde = stats.gaussian_kde(dataset_data)
                    x_range = np.linspace(dataset_data.min(), dataset_data.max(), 100)
                    pdf_values = kde(x_range)
                    ax.plot(x_range, pdf_values, color=color, linestyle=line_style, 
                           linewidth=line_width, label=filename)
                except ImportError:
                    # Fallback: histogram-based PDF
                    hist, bin_edges = np.histogram(dataset_data, bins=30, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    ax.plot(bin_centers, hist, color=color, linestyle=line_style, 
                           linewidth=line_width, label=filename)
            
            ax.set_xlabel('Concentration [ppmV]')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True)
        else:
            # Fallback to single dataset approach
            data = df[data_col].dropna()
            if len(data) == 0:
                return
            
            # Calculate PDF
            try:
                from scipy import stats
                # Fit a kernel density estimate
                kde = stats.gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 100)
                pdf_values = kde(x_range)
                ax.plot(x_range, pdf_values, color=line_color, linestyle=line_style, 
                       linewidth=line_width, label='PDF')
            except ImportError:
                # Fallback: histogram-based PDF
                hist, bin_edges = np.histogram(data, bins=30, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax.plot(bin_centers, hist, color=line_color, linestyle=line_style, 
                       linewidth=line_width, label='PDF')
            
            ax.set_xlabel('Concentration [ppmV]')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True)
        
        # Set axis ranges if specified
        if xAchse and xAchse.strip():
            try:
                x_range = eval(xAchse)
                if isinstance(x_range, tuple) and len(x_range) == 2:
                    ax.set_xlim(x_range)
            except:
                pass
        
        if yAchse and yAchse.strip():
            try:
                y_range = eval(yAchse)
                if isinstance(y_range, tuple) and len(y_range) == 2:
                    ax.set_ylim(y_range)
            except:
                pass
    
    def _create_cdf_plot(self, ax, df: pd.DataFrame, config: PlotConfiguration):
        """Create a CDF plot with support for multiple datasets."""
        data_col = self._get_available_column(df, config.data_columns)
        if data_col is None:
            return
        
        # Get parameters
        params = self._get_parameter_dict(config)
        dimensionless = params.get("dimensionless", "False")
        line_color = params.get("color", "blue")
        line_style = params.get("line_style", "solid")
        line_width = params.get("line_width", 2.0)
        xAchse = params.get("xAchse", "")
        yAchse = params.get("yAchse", "")
        
        # Use dimensionless data if requested
        if dimensionless == "True" and "c_star" in df.columns:
            data_col = "c_star"
        
        # Group by filename to get individual datasets
        if 'filename' in df.columns:
            grouped_data = df.groupby('filename')
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
            
            for i, (filename, group) in enumerate(grouped_data):
                dataset_data = group[data_col].dropna()
                if len(dataset_data) == 0:
                    continue
                
                # Choose color for this dataset
                color = colors[i % len(colors)]
                
                # Calculate CDF
                sorted_data = np.sort(dataset_data)
                y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                ax.plot(sorted_data, y, color=color, linewidth=line_width, 
                       linestyle=line_style, label=filename)
            
            ax.set_xlabel('Concentration [ppmV]')
            ax.set_ylabel('Cumulative Probability')
            ax.legend()
            ax.grid(True)
        else:
            # Fallback to single dataset approach
            data = df[data_col].dropna()
            if len(data) == 0:
                return
            
            # Calculate CDF
            sorted_data = np.sort(data)
            y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, y, color=line_color, linewidth=line_width, 
                   linestyle=line_style, label='CDF')
            ax.set_xlabel('Concentration [ppmV]')
            ax.set_ylabel('Cumulative Probability')
            ax.legend()
            ax.grid(True)
        
        # Set axis ranges if specified
        if xAchse and xAchse.strip():
            try:
                x_range = eval(xAchse)
                if isinstance(x_range, tuple) and len(x_range) == 2:
                    ax.set_xlim(x_range)
            except:
                pass
        
        if yAchse and yAchse.strip():
            try:
                y_range = eval(yAchse)
                if isinstance(y_range, tuple) and len(y_range) == 2:
                    ax.set_ylim(y_range)
            except:
                pass
    
    def _create_means_plot(self, ax, df: pd.DataFrame, config: PlotConfiguration):
        """Create a means plot with individual points for each dataset."""
        data_col = self._get_available_column(df, config.data_columns)
        if data_col is None:
            return
        
        # Get parameters
        params = self._get_parameter_dict(config)
        error_values = params.get("error_values", None)
        error_type = params.get("error_type", "absolute")
        dimensionless = params.get("dimensionless", "False")
        bar_color = params.get("bar_color", "blue")
        error_color = params.get("error_color", "black")
        error_capsize = params.get("error_capsize", 5.0)
        xAchse = params.get("xAchse", "")
        yAchse = params.get("yAchse", "")
        
        # Use dimensionless data if requested
        if dimensionless == "True" and "c_star" in df.columns:
            data_col = "c_star"
        
        # Group by filename to get individual datasets
        if 'filename' in df.columns:
            grouped_data = df.groupby('filename')
            
            # Calculate means for each dataset
            means = []
            labels = []
            errors = []
            
            for filename, group in grouped_data:
                dataset_data = group[data_col].dropna()
                if len(dataset_data) > 0:
                    mean_val = dataset_data.mean()
                    means.append(mean_val)
                    labels.append(filename)
                    
                    # Calculate error based on type
                    if error_values is not None:
                        if error_type == "percentage":
                            errors.append(mean_val * (error_values / 100))
                        elif error_type == "absolute":
                            errors.append(error_values)
                        elif error_type == "std":
                            errors.append(dataset_data.std())
                    else:
                        errors.append(dataset_data.std())
            
            if len(means) > 0:
                x_positions = np.arange(1, len(means) + 1)
                
                # Plot with error bars
                ax.errorbar(x=x_positions, y=means, yerr=errors, fmt='o', 
                           capsize=error_capsize, color=bar_color, ecolor=error_color)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(labels, rotation=45 if len(labels) > 3 else 0)
                ax.set_title('Mean Comparison')
                ax.set_ylabel('Value')
                ax.grid(True)
                
                # Set axis ranges if specified
                if xAchse and xAchse.strip():
                    try:
                        x_range = eval(xAchse)
                        if isinstance(x_range, tuple) and len(x_range) == 2:
                            ax.set_xlim(x_range)
                    except:
                        pass
                
                if yAchse and yAchse.strip():
                    try:
                        y_range = eval(yAchse)
                        if isinstance(y_range, tuple) and len(y_range) == 2:
                            ax.set_ylim(y_range)
                    except:
                        pass
        else:
            # Fallback to single dataset approach
            data = df[data_col].dropna()
            if len(data) == 0:
                return
            
            # Calculate mean and standard error
            mean_val = data.mean()
            std_val = data.std()
            sem_val = data.sem()
            
            ax.axhline(y=mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axhline(y=mean_val + std_val, color='red', linestyle='--', alpha=0.7, label=f'±1σ: {std_val:.3f}')
            ax.axhline(y=mean_val - std_val, color='red', linestyle='--', alpha=0.7)
            ax.axhline(y=mean_val + sem_val, color='blue', linestyle='--', alpha=0.7, label=f'SEM: {sem_val:.3f}')
            ax.axhline(y=mean_val - sem_val, color='blue', linestyle='--', alpha=0.7)
            
            ax.legend()
    
    def _create_power_density_plot(self, ax, df: pd.DataFrame, config: PlotConfiguration):
        """Create a power density plot."""
        data_col = self._get_available_column(df, config.data_columns)
        if data_col is None:
            return
        
        # Get parameters
        params = self._get_parameter_dict(config)
        dimensionless = params.get("dimensionless", "False")
        line_color = params.get("line_color", "blue")
        line_style = params.get("line_style", "solid")
        line_width = params.get("line_width", 1.5)
        xAchse = params.get("xAchse", "")
        yAchse = params.get("yAchse", "")
        
        # Use dimensionless data if requested
        if dimensionless == "True" and "c_star" in df.columns:
            data_col = "c_star"
        
        # Group by filename to get individual datasets
        if 'filename' in df.columns:
            grouped_data = df.groupby('filename')
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for i, (filename, group) in enumerate(grouped_data):
                dataset_data = group[data_col].dropna()
                if len(dataset_data) == 0:
                    continue
                
                # Choose color for this dataset
                color = colors[i % len(colors)]
                
                # Calculate power spectrum
                try:
                    from scipy import signal
                    f, Pxx = signal.welch(dataset_data, fs=1.0)
                    ax.semilogy(f, Pxx, color=color, linestyle=line_style, 
                               linewidth=line_width, label=filename)
                except ImportError:
                    # Fallback: simple FFT
                    fft_vals = np.abs(np.fft.fft(dataset_data))
                    freqs = np.fft.fftfreq(len(dataset_data))
                    ax.semilogy(freqs[1:len(freqs)//2], fft_vals[1:len(fft_vals)//2], 
                               color=color, linestyle=line_style, linewidth=line_width, 
                               label=filename)
            
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Power Spectral Density')
            ax.legend()
            ax.grid(True)
        else:
            # Fallback to single dataset approach
            data = df[data_col].dropna()
            if len(data) == 0:
                return
            
            # Simple power spectrum
            try:
                from scipy import signal
                f, Pxx = signal.welch(data, fs=1.0)
                ax.semilogy(f, Pxx, color=line_color, linestyle=line_style, linewidth=line_width)
                ax.set_xlabel('Frequency')
                ax.set_ylabel('Power Spectral Density')
            except ImportError:
                # Fallback: simple FFT
                fft_vals = np.abs(np.fft.fft(data))
                freqs = np.fft.fftfreq(len(data))
                ax.semilogy(freqs[1:len(freqs)//2], fft_vals[1:len(fft_vals)//2], 
                           color=line_color, linestyle=line_style, linewidth=line_width)
                ax.set_xlabel('Frequency')
                ax.set_ylabel('FFT Magnitude')
        
        # Set axis ranges if specified
        if xAchse and xAchse.strip():
            try:
                x_range = eval(xAchse)
                if isinstance(x_range, tuple) and len(x_range) == 2:
                    ax.set_xlim(x_range)
            except:
                pass
        
        if yAchse and yAchse.strip():
            try:
                y_range = eval(yAchse)
                if isinstance(y_range, tuple) and len(y_range) == 2:
                    ax.set_ylim(y_range)
            except:
                pass
    
    def _create_default_plot(self, ax, df: pd.DataFrame, config: PlotConfiguration):
        """Create a default line plot."""
        data_col = self._get_available_column(df, config.data_columns)
        if data_col is None:
            return
        
        data = df[data_col].dropna()
        if len(data) == 0:
            return
        
        ax.plot(data.index, data.values, 'b-', linewidth=1)
    
    def _get_parameter_dict(self, config: PlotConfiguration) -> Dict[str, Any]:
        """Convert configuration parameters to a dictionary."""
        params = {}
        for param in config.parameters:
            params[param.name] = param.default_value
        return params
    
    def _get_available_column(self, df: pd.DataFrame, columns: List[str]) -> Optional[str]:
        """Get the first available column from the list."""
        for col in columns:
            if col in df.columns:
                return col
        return None
