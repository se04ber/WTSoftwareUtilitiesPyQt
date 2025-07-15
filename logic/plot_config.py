"""
logic/plot_config.py
Plot configuration system for the application.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import os

class PlotType(Enum):
    """Enumeration of available plot types."""
    HISTOGRAM = "Histogram"
    PDF = "Pdf"
    CDF = "Cdf"
    MEANS = "Means"
    SCATTER_PLOT = "ScatterPlot"
    POWER_DENSITY = "PowerDensity"
    BOX_PLOT = "BoxPlot"

@dataclass
class PlotParameter:
    """Represents a configurable plot parameter."""
    name: str
    display_name: str
    parameter_type: str  # 'text', 'number', 'boolean', 'choice'
    default_value: Any
    description: str = ""
    choices: Optional[List[str]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    required: bool = True

@dataclass
class PlotConfiguration:
    """Configuration for a specific plot type."""
    plot_type: PlotType
    title: str
    x_label: str
    y_label: str
    parameters: List[PlotParameter] = field(default_factory=list)
    data_columns: List[str] = field(default_factory=list)
    enabled: bool = True

class PlotConfigManager:
    """
    Manages plot configurations and provides methods to create, save, and load them.
    """
    
    def __init__(self):
        self.configurations: Dict[PlotType, PlotConfiguration] = {}
        self._initialize_default_configurations()
    
    def _initialize_default_configurations(self):
        """Initialize default plot configurations."""
        
        # Histogram Configuration
        histogram_config = PlotConfiguration(
            plot_type=PlotType.HISTOGRAM,
            title="Concentration Distribution",
            x_label="Concentration [ppmV]",
            y_label="Frequency",
            data_columns=["net_concentration", "full_scale_concentration"],
            parameters=[
                PlotParameter("bins", "Number of Bins", "number", 20, 
                            "Number of histogram bins", min_value=5, max_value=100),
                PlotParameter("density", "Show Density", "boolean", True,
                            "Show density instead of counts"),
                PlotParameter("color", "Bar Color", "choice", "blue",
                            "Color of histogram bars", 
                            choices=["blue", "red", "green", "orange", "purple"]),
                PlotParameter("alpha", "Transparency", "number", 0.7,
                            "Transparency level (0-1)", min_value=0.0, max_value=1.0),
                PlotParameter("dimensionless", "Dimensionless", "choice", "False",
                            "Use dimensionless concentration (C*)",
                            choices=["True", "False"]),
                PlotParameter("labels", "Custom Labels", "text", "",
                            "Custom labels for datasets (comma-separated)"),
                PlotParameter("xAchse", "X-Axis Range", "text", "",
                            "X-axis range as tuple (min, max) or leave empty for auto"),
                PlotParameter("yAchse", "Y-Axis Range", "text", "",
                            "Y-axis range as tuple (min, max) or leave empty for auto"),
                PlotParameter("legend_fontsize", "Legend Font Size", "number", 10,
                            "Font size for legend text", min_value=6, max_value=20)
            ]
        )
        self.configurations[PlotType.HISTOGRAM] = histogram_config
        
        # Scatter Plot Configuration
        scatter_config = PlotConfiguration(
            plot_type=PlotType.SCATTER_PLOT,
            title="Concentration vs Distance",
            x_label="Distance [m]",
            y_label="Concentration [ppmV]",
            data_columns=["x_fs", "y_fs", "z_fs", "net_concentration"],
            parameters=[
                PlotParameter("x_column", "X-Axis Data", "choice", "x_fs",
                            "Data column for X-axis", 
                            choices=["x_fs", "y_fs", "z_fs", "net_concentration"]),
                PlotParameter("y_column", "Y-Axis Data", "choice", "net_concentration",
                            "Data column for Y-axis",
                            choices=["x_fs", "y_fs", "z_fs", "net_concentration", "c_star"]),
                PlotParameter("point_size", "Point Size", "number", 50,
                            "Size of scatter points", min_value=10, max_value=200),
                PlotParameter("color_by", "Color By", "choice", "none",
                            "Color points by another variable",
                            choices=["none", "z_fs", "c_star", "net_concentration"]),
                PlotParameter("dimensionless", "Dimensionless", "choice", "False",
                            "Use dimensionless concentration (C*)",
                            choices=["True", "False"]),
                PlotParameter("labels", "Custom Labels", "text", "",
                            "Custom labels for datasets (comma-separated)"),
                PlotParameter("xAchse", "X-Axis Range", "text", "",
                            "X-axis range as tuple (min, max) or leave empty for auto"),
                PlotParameter("yAchse", "Y-Axis Range", "text", "",
                            "Y-axis range as tuple (min, max) or leave empty for auto"),
                PlotParameter("alpha", "Transparency", "number", 0.7,
                            "Transparency level (0-1)", min_value=0.0, max_value=1.0),
                PlotParameter("marker_style", "Marker Style", "choice", "o",
                            "Marker style for scatter points",
                            choices=["o", "s", "^", "v", "D", "p", "*", "h", "H", "+", "x"]),
                PlotParameter("legend_fontsize", "Legend Font Size", "number", 10,
                            "Font size for legend text", min_value=6, max_value=20)
            ]
        )
        self.configurations[PlotType.SCATTER_PLOT] = scatter_config
        
        # Box Plot Configuration
        box_config = PlotConfiguration(
            plot_type=PlotType.BOX_PLOT,
            title="Concentration Statistics by Location",
            x_label="Location",
            y_label="Concentration [ppmV]",
            data_columns=["net_concentration", "full_scale_concentration"],
            parameters=[
                PlotParameter("group_by", "Group By", "choice", "filename",
                            "Group data by this column",
                            choices=["filename", "x_fs", "y_fs", "z_fs"]),
                PlotParameter("show_outliers", "Show Outliers", "boolean", True,
                            "Show outlier points"),
                PlotParameter("notch", "Show Notch", "boolean", False,
                            "Show confidence interval notch"),
                PlotParameter("color_scheme", "Color Scheme", "choice", "default",
                            "Color scheme for boxes",
                            choices=["default", "viridis", "plasma", "coolwarm"]),
                PlotParameter("dimensionless", "Dimensionless", "choice", "False",
                            "Use dimensionless concentration (C*)",
                            choices=["True", "False"]),
                PlotParameter("labels", "Custom Labels", "text", "",
                            "Custom labels for datasets (comma-separated)"),
                PlotParameter("xAchse", "X-Axis Range", "text", "",
                            "X-axis range as tuple (min, max) or leave empty for auto"),
                PlotParameter("yAchse", "Y-Axis Range", "text", "",
                            "Y-axis range as tuple (min, max) or leave empty for auto"),
                PlotParameter("box_width", "Box Width", "number", 0.8,
                            "Width of boxes", min_value=0.1, max_value=2.0),
                PlotParameter("flier_size", "Outlier Size", "number", 3.0,
                            "Size of outlier points", min_value=1.0, max_value=10.0),
                PlotParameter("legend_fontsize", "Legend Font Size", "number", 10,
                            "Font size for legend text", min_value=6, max_value=20)
            ]
        )
        self.configurations[PlotType.BOX_PLOT] = box_config
        
        # PDF Configuration
        pdf_config = PlotConfiguration(
            plot_type=PlotType.PDF,
            title="Probability Density Functions",
            x_label="Concentration [ppmV]",
            y_label="Density",
            data_columns=["net_concentration", "full_scale_concentration"],
            parameters=[
                PlotParameter("color", "Line Color", "choice", "blue",
                            "Color of PDF lines", 
                            choices=["blue", "red", "green", "orange", "purple", "black"]),
                PlotParameter("dimensionless", "Dimensionless", "choice", "False",
                            "Use dimensionless concentration (C*)",
                            choices=["True", "False"]),
                PlotParameter("labels", "Custom Labels", "text", "",
                            "Custom labels for datasets (comma-separated)"),
                PlotParameter("xAchse", "X-Axis Range", "text", "",
                            "X-axis range as tuple (min, max) or leave empty for auto"),
                PlotParameter("yAchse", "Y-Axis Range", "text", "",
                            "Y-axis range as tuple (min, max) or leave empty for auto"),
                PlotParameter("line_style", "Line Style", "choice", "solid",
                            "Line style for PDF curves",
                            choices=["solid", "dashed", "dotted", "dashdot"]),
                PlotParameter("line_width", "Line Width", "number", 2.0,
                            "Width of PDF lines", min_value=0.5, max_value=5.0),
                PlotParameter("legend_fontsize", "Legend Font Size", "number", 10,
                            "Font size for legend text", min_value=6, max_value=20)
            ]
        )
        self.configurations[PlotType.PDF] = pdf_config
        
        # CDF Configuration
        cdf_config = PlotConfiguration(
            plot_type=PlotType.CDF,
            title="Cumulative Distribution Function",
            x_label="Concentration [ppmV]",
            y_label="Cumulative Probability",
            data_columns=["net_concentration", "full_scale_concentration"],
            parameters=[
                PlotParameter("color", "Line Color", "choice", "blue",
                            "Color of CDF line", 
                            choices=["blue", "red", "green", "orange", "purple", "black"]),
                PlotParameter("dimensionless", "Dimensionless", "choice", "False",
                            "Use dimensionless concentration (C*)",
                            choices=["True", "False"]),
                PlotParameter("labels", "Custom Labels", "text", "",
                            "Custom labels for datasets (comma-separated)"),
                PlotParameter("xAchse", "X-Axis Range", "text", "",
                            "X-axis range as tuple (min, max) or leave empty for auto"),
                PlotParameter("yAchse", "Y-Axis Range", "text", "",
                            "Y-axis range as tuple (min, max) or leave empty for auto"),
                PlotParameter("line_style", "Line Style", "choice", "solid",
                            "Line style for CDF curves",
                            choices=["solid", "dashed", "dotted", "dashdot"]),
                PlotParameter("line_width", "Line Width", "number", 2.0,
                            "Width of CDF lines", min_value=0.5, max_value=5.0),
                PlotParameter("legend_fontsize", "Legend Font Size", "number", 10,
                            "Font size for legend text", min_value=6, max_value=20)
            ]
        )
        self.configurations[PlotType.CDF] = cdf_config
        
        # Means Configuration
        means_config = PlotConfiguration(
            plot_type=PlotType.MEANS,
            title="Mean Comparison",
            x_label="Datasets",
            y_label="Concentration [ppmV]",
            data_columns=["net_concentration", "full_scale_concentration"],
            parameters=[
                PlotParameter("error_values", "Error Values", "number", 0.5,
                            "Error bar values (absolute or percentage)", min_value=0.0),
                PlotParameter("error_type", "Error Type", "choice", "absolute",
                            "Type of error calculation",
                            choices=["absolute", "percentage", "std"]),
                PlotParameter("dimensionless", "Dimensionless", "choice", "False",
                            "Use dimensionless concentration (C*)",
                            choices=["True", "False"]),
                PlotParameter("labels", "Custom Labels", "text", "",
                            "Custom labels for datasets (comma-separated)"),
                PlotParameter("xAchse", "X-Axis Range", "text", "",
                            "X-axis range as tuple (min, max) or leave empty for auto"),
                PlotParameter("yAchse", "Y-Axis Range", "text", "",
                            "Y-axis range as tuple (min, max) or leave empty for auto"),
                PlotParameter("bar_color", "Bar Color", "choice", "blue",
                            "Color of mean bars",
                            choices=["blue", "red", "green", "orange", "purple", "gray"]),
                PlotParameter("error_color", "Error Bar Color", "choice", "black",
                            "Color of error bars",
                            choices=["black", "red", "blue", "green", "orange"]),
                PlotParameter("error_capsize", "Error Cap Size", "number", 5.0,
                            "Size of error bar caps", min_value=0.0, max_value=20.0),
                PlotParameter("legend_fontsize", "Legend Font Size", "number", 10,
                            "Font size for legend text", min_value=6, max_value=20)
            ]
        )
        self.configurations[PlotType.MEANS] = means_config
        
        # Power Density Configuration
        power_density_config = PlotConfiguration(
            plot_type=PlotType.POWER_DENSITY,
            title="Power Spectral Density",
            x_label="Frequency",
            y_label="Power Density",
            data_columns=["net_concentration", "full_scale_concentration"],
            parameters=[
                PlotParameter("dimensionless", "Dimensionless", "choice", "False",
                            "Use dimensionless concentration (C*)",
                            choices=["True", "False"]),
                PlotParameter("labels", "Custom Labels", "text", "",
                            "Custom labels for datasets (comma-separated)"),
                PlotParameter("xAchse", "X-Axis Range", "text", "",
                            "X-axis range as tuple (min, max) or leave empty for auto"),
                PlotParameter("yAchse", "Y-Axis Range", "text", "",
                            "Y-axis range as tuple (min, max) or leave empty for auto"),
                PlotParameter("plot", "Show Plot", "boolean", True,
                            "Display the plot (vs. just return figure)"),
                PlotParameter("line_color", "Line Color", "choice", "blue",
                            "Color of power density lines",
                            choices=["blue", "red", "green", "orange", "purple", "black"]),
                PlotParameter("line_style", "Line Style", "choice", "solid",
                            "Line style for power density curves",
                            choices=["solid", "dashed", "dotted", "dashdot"]),
                PlotParameter("line_width", "Line Width", "number", 1.5,
                            "Width of power density lines", min_value=0.5, max_value=5.0),
                PlotParameter("legend_fontsize", "Legend Font Size", "number", 10,
                            "Font size for legend text", min_value=6, max_value=20)
            ]
        )
        self.configurations[PlotType.POWER_DENSITY] = power_density_config
    
    def get_configuration(self, plot_type: PlotType) -> Optional[PlotConfiguration]:
        """Get configuration for a specific plot type."""
        return self.configurations.get(plot_type)
    
    def get_all_configurations(self) -> Dict[PlotType, PlotConfiguration]:
        """Get all plot configurations."""
        return self.configurations.copy()
    
    def update_configuration(self, plot_type: PlotType, config: PlotConfiguration):
        """Update a plot configuration."""
        self.configurations[plot_type] = config
    
    def save_configurations(self, file_path: str):
        """Save all configurations to a JSON file."""
        config_data = {}
        for plot_type, config in self.configurations.items():
            config_data[plot_type.value] = {
                "title": config.title,
                "x_label": config.x_label,
                "y_label": config.y_label,
                "data_columns": config.data_columns,
                "enabled": config.enabled,
                "parameters": [
                    {
                        "name": param.name,
                        "display_name": param.display_name,
                        "parameter_type": param.parameter_type,
                        "default_value": param.default_value,
                        "description": param.description,
                        "choices": param.choices,
                        "min_value": param.min_value,
                        "max_value": param.max_value,
                        "required": param.required
                    }
                    for param in config.parameters
                ]
            }
        
        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def load_configurations(self, file_path: str):
        """Load configurations from a JSON file."""
        if not os.path.exists(file_path):
            return
        
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        
        for plot_type_str, config_dict in config_data.items():
            try:
                plot_type = PlotType(plot_type_str)
                parameters = []
                for param_dict in config_dict.get("parameters", []):
                    param = PlotParameter(
                        name=param_dict["name"],
                        display_name=param_dict["display_name"],
                        parameter_type=param_dict["parameter_type"],
                        default_value=param_dict["default_value"],
                        description=param_dict.get("description", ""),
                        choices=param_dict.get("choices"),
                        min_value=param_dict.get("min_value"),
                        max_value=param_dict.get("max_value"),
                        required=param_dict.get("required", True)
                    )
                    parameters.append(param)
                
                config = PlotConfiguration(
                    plot_type=plot_type,
                    title=config_dict["title"],
                    x_label=config_dict["x_label"],
                    y_label=config_dict["y_label"],
                    data_columns=config_dict.get("data_columns", []),
                    enabled=config_dict.get("enabled", True),
                    parameters=parameters
                )
                self.configurations[plot_type] = config
            except ValueError:
                # Skip unknown plot types
                continue 