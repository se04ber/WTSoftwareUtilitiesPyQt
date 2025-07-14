import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap, Normalize

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from stl import mesh
import csv
import re
import glob
import pandas as pd
import os


def load_avg_file(filepath):
    """
    Load and parse an avg file to extract metadata and concentration data.
    
    Args:
        filepath (str): Path to the avg file
        
    Returns:
        dict: Parsed data containing metadata and concentration values
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Extract filename from path and clean it
        full_filename = Path(filepath).name
        # Extract only the part starting with _avg_ and ending with .txt.ts#0
        filename_match = re.search(r'(_avg_.*?\.txt\.ts#\d+)', full_filename)
        filename = filename_match.group(1) if filename_match else full_filename
        
        # Parse metadata from header
        metadata = {}
        
        # Extract geometric scale
        scale_match = re.search(r'geometric scale: 1:(\d+\.?\d*)', content)
        if scale_match:
            metadata['geometric_scale'] = float(scale_match.group(1))
        
        # Extract measurement positions (in mm)
        x_match = re.search(r'x \(measurement relativ to source\): ([\d\.-]+) \[mm\]', content)
        y_match = re.search(r'y \(measurement relativ to source\): ([\d\.-]+) \[mm\]', content)
        z_match = re.search(r'z \(measurement relativ to source\): ([\d\.-]+) \[mm\]', content)
        
        if x_match and y_match and z_match:
            # Convert to full scale (mm to m, then scale up)
            scale = metadata.get('geometric_scale', 200.0)
            metadata['x_fs'] = float(x_match.group(1)) / 1000.0 * scale  # mm to m, then scale
            metadata['y_fs'] = float(y_match.group(1)) / 1000.0 * scale
            metadata['z_fs'] = float(z_match.group(1)) / 1000.0 * scale
        
        # Find the data line (last non-comment line)
        lines = content.strip().split('\n')
        data_line = None
        for line in reversed(lines):
            if line.strip() and not line.strip().startswith('#'):
                data_line = line.strip()
                break
        
        if data_line:
            # Parse concentration data
            values = data_line.split()
            if len(values) >= 3:
                metadata['c_star'] = float(values[0])
                metadata['net_concentration'] = float(values[1])
                metadata['full_scale_concentration'] = float(values[2])
        
        metadata['filename'] = filename
        return metadata
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def load_stats_file(filepath):
    """
    Load and parse a stats file to extract metadata and statistical concentration data.
    
    Args:
        filepath (str): Path to the stats file
        
    Returns:
        dict: Parsed data containing metadata and statistical values
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Extract filename from path and clean it
        full_filename = Path(filepath).name
        # Extract only the part starting with _stats_ and ending with .txt.ts#0
        filename_match = re.search(r'(_stats_.*?\.txt\.ts#\d+)', full_filename)
        filename = filename_match.group(1) if filename_match else full_filename
        
        # Parse metadata from header (same as avg file)
        metadata = {}
        
        # Extract geometric scale
        scale_match = re.search(r'geometric scale: 1:(\d+\.?\d*)', content)
        if scale_match:
            metadata['geometric_scale'] = float(scale_match.group(1))
        
        # Extract measurement positions (in mm)
        x_match = re.search(r'x \(measurement relativ to source\): ([\d\.-]+) \[mm\]', content)
        y_match = re.search(r'y \(measurement relativ to source\): ([\d\.-]+) \[mm\]', content)
        z_match = re.search(r'z \(measurement relativ to source\): ([\d\.-]+) \[mm\]', content)
        
        if x_match and y_match and z_match:
            # Convert to full scale (mm to m, then scale up)
            scale = metadata.get('geometric_scale', 200.0)
            metadata['x_fs'] = float(x_match.group(1)) / 1000.0 * scale  # mm to m, then scale
            metadata['y_fs'] = float(y_match.group(1)) / 1000.0 * scale
            metadata['z_fs'] = float(z_match.group(1)) / 1000.0 * scale
        
        # Find the statistical data line (should be the first non-comment line after header)
        lines = content.strip().split('\n')
        data_line = None
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):
                data_line = line.strip()
                break
        
        if data_line:
            # Parse statistical data - should have 12 values
            # Order: means(3), percentiles95(3), percentiles5(3), peak2mean(3)
            values = data_line.split()
            if len(values) >= 12:
                # Means
                metadata['c_star'] = float(values[0])
                metadata['net_concentration'] = float(values[1])
                metadata['full_scale_concentration'] = float(values[2])
                
                # 95th percentiles
                metadata['percentiles95_c_star'] = float(values[3])
                metadata['percentiles95_net_concentration'] = float(values[4])
                metadata['percentiles95_full_scale_concentration'] = float(values[5])
                
                # 5th percentiles
                metadata['percentiles5_c_star'] = float(values[6])
                metadata['percentiles5_net_concentration'] = float(values[7])
                metadata['percentiles5_full_scale_concentration'] = float(values[8])
                
                # Peak2Mean ratios
                metadata['peak2mean_c_star'] = float(values[9])
                metadata['peak2mean_net_concentration'] = float(values[10])
                metadata['peak2mean_full_scale_concentration'] = float(values[11])
        
        metadata['filename'] = filename
        return metadata
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def combine_to_csv(file_names, base_path, file_type='avg', output_filename='combined_data.csv'):
    """
    Combine specific avg or stats files into a single CSV file.
    
    Args:
        file_names (list): Array of specific filenames to process
        base_path (str): Base directory path containing the files
        file_type (str): Type of files to process ('avg' or 'stats')
        output_filename (str): Name of the output CSV file
        
    Returns:
        pd.DataFrame: Combined data as a pandas DataFrame
    """
    
    # Define subdirectory and load function based on file type
    if file_type.lower() == 'avg':
        #subdirectory = 'Point_Data_avg'
        load_function = load_avg_file
    elif file_type.lower() == 'stats':
        #subdirectory = 'Point_Data_stats'
        load_function = load_stats_file
    else:
        raise ValueError("file_type must be 'avg' or 'stats'")
    
    # Build full file paths from the provided filenames
    file_paths = []
    for filename in file_names:
        # Search for the file in the subdirectory structure
        search_pattern = os.path.join(base_path, '**', filename)
        matches = glob.glob(search_pattern, recursive=True)
        
        if matches:
            file_paths.extend(matches)
        else:
            print(f"Warning: File '{filename}' not found in {base_path}{filename}")
    
    if not file_paths:
        print(f"No valid {file_type} files found from the provided list")
        return None
    
    print(f"Processing {len(file_paths)} {file_type} files")
    
    # Process all files
    all_data = []
    for filepath in file_paths:
        print(f"Processing: {filepath}")
        data = load_function(filepath)
        if data:
            all_data.append(data)
    
    if not all_data:
        print("No valid data found in any files")
        return None
    
    # Create DataFrame (flatten nested dicts, add filename)
    flat_data = []
    for entry, filepath in zip(all_data, file_paths):
        flat_entry = {}
        # Add filename from the actual file processed
        flat_entry['filename'] = os.path.basename(filepath)
        # Flatten metadata, extracting only the value for dicts with a 'value' key
        if 'metadata' in entry and isinstance(entry['metadata'], dict):
            for k, v in entry['metadata'].items():
                if isinstance(v, dict) and 'value' in v:
                    flat_entry[k] = v['value']
                else:
                    flat_entry[k] = v
        # Flatten data (first row only)
        if 'data' in entry and isinstance(entry['data'], list) and entry['data']:
            for k, v in entry['data'][0].items():
                flat_entry[k] = v
        flat_data.append(flat_entry)
    df = pd.DataFrame(flat_data)
    print("[DEBUG] DataFrame columns:", df.columns)
    print("[DEBUG] DataFrame head:\n", df.head())
    
    # Map actual column names to expected ones
    column_mapping = {
        'x (measurement relativ to source)': 'x_fs',
        'y (measurement relativ to source)': 'y_fs',
        'z (measurement relativ to source)': 'z_fs',
        'Means: c_star [-]': 'c_star',
        'net_concentration [ppmV]': 'net_concentration',
        'full_scale_concentration [ppmV]': 'full_scale_concentration',
        'Percentiles 95: c_star [-]': 'percentiles95_c_star',
        'Percentiles 5: c_star [-]': 'percentiles5_c_star',
        'Peak2MeanRatio: c_star [-]': 'peak2mean_c_star',
        # Add more if your data has them
    }
    df = df.rename(columns=column_mapping)
    print("[DEBUG] DataFrame columns after mapping:", df.columns)
    
    # Define columns to export (only those that exist)
    columns = [
        'filename', 'x_fs', 'y_fs', 'z_fs',
        'c_star', 'net_concentration', 'full_scale_concentration',
        'percentiles95_c_star', 'percentiles5_c_star', 'peak2mean_c_star'
    ]
    available_columns = [col for col in columns if col in df.columns]
    print("[DEBUG] Actually exporting columns:", available_columns)
    df = df[available_columns]
    
    # Define column order based on file type
    if file_type.lower() == 'avg':
        columns = [
            'filename', 'x_fs', 'y_fs', 'z_fs',
            'c_star', 'net_concentration', 'full_scale_concentration'
        ]
        # Rename columns to match desired format
        column_mapping_out = {
            'filename': 'Filename',
            'x_fs': 'X_fs [m]',
            'y_fs': 'Y_fs [m]',
            'z_fs': 'Z_fs [m]',
            'c_star': 'Avg_c_star [-]',
            'net_concentration': 'Avg_net_concentration [ppmV]',
            'full_scale_concentration': 'Avg_full_scale_concentration [ppmV]'
        }
    else:  # stats
        columns = [
            'filename', 'x_fs', 'y_fs', 'z_fs',
            'c_star', 'net_concentration', 'full_scale_concentration',
            'percentiles95_c_star', 'percentiles5_c_star', 'peak2mean_c_star'
        ]
        # Rename columns to match desired format
        column_mapping_out = {
            'filename': 'Filename',
            'x_fs': 'X_fs [m]',
            'y_fs': 'Y_fs [m]',
            'z_fs': 'Z_fs [m]',
            'c_star': 'Avg_c_star [-]',
            'net_concentration': 'Avg_net_concentration [ppmV]',
            'full_scale_concentration': 'Avg_full_scale_concentration [ppmV]',
            'percentiles95_c_star': 'Percentiles 95_cstar',
            'percentiles5_c_star': 'Percentiles 5_cstar',
            'peak2mean_c_star': 'Peak2MeanRatio_cstar'
        }
    # Print missing columns before selection
    missing = [col for col in columns if col not in df.columns]
    print("[DEBUG] Missing columns before selection:", missing)
    
    # Select and rename columns
    df = df[columns].rename(columns=column_mapping_out)
    
    # Fill NaN values with 0.0
    df = df.fillna(0.0)
    
    # Save to CSV
    df.to_csv(output_filename, index=False)
    print(f"Data saved to {output_filename}")
    
    return df



import csv

def load_csv(file_path):
    """
    Load data from a CSV file returns points and values lists for printing/or further analysis.
    
    Args:
        filename (str): Path to the CSV file
        
    Returns:
        tuple: (points, values) where:
            - points is a list of (x,y,z) tuples
            - values is a list of concentration values
    """
    points = []
    values = []
    
    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Check if we have enough elements in the row
            if len(row) == 4:
                x = float(row[0])
                y = float(row[1])
                z = float(row[2])
                c_star = float(row[3])
                
            elif len(row) > 4:
                #try:
                # Parse data, assuming the format matches X_fs [m],Y_fs [m],Z_fs [m],C* [-]
                file = row[0]
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
                c_star = float(row[4])
                if len(row) >=6:
                    c_net = float(row[5])
                    if len(row) >=7:
                        c_fs = float(row[6])
                
            # Add to our lists
            points.append((x, y, z))
            values.append(c_star)
                #except ValueError:
                #    # Skip rows that can't be converted to float
                #    print(f"Skipping invalid row: {row}")
    
    return points, values
    
    
    
#Functions for plotting locations of measurements in a CAD loaded background  

def stl_to_2d_plot(stl_file, projection='xy', alpha=0.7, color='lightgrey',
                 edgecolor='grey', figsize=(10, 8), toFullScale="False", scaling=1, 
                 toLatLon="False", ref_coords=None, ref_pos=None):
    """
    Convert an STL file to a 2D shaded plot, with option to plot in lat/lon coordinates.
    
    Parameters:
    -----------
    stl_file : str
        Path to the STL file
    projection : str
        Which projection to use: 'xy', 'xz', 'yz', or 'diagonal'
    alpha : float
        Transparency of the shaded regions
    color : str
        Color for the shaded regions
    edgecolor : str
        Color for the edges
    figsize : tuple
        Figure size for the plot
    toFullScale : str
        "True" to scale the mesh by scaling factor / 1000
    scaling : float
        Scaling factor when toFullScale is "True"
    toLatLon : str
        "True" to convert coordinates to lat/lon
    ref_coords : list
        [lat, lon] reference coordinates corresponding to ref_pos position in the mesh
    ref_pos : list
        [x, y] position in the mesh corresponding to ref_coords. Defaults to [0, 0] if not specified.
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
   
    print(f"Loading STL file: {stl_file}")
    your_mesh = mesh.Mesh.from_file(stl_file)
    
    # Scale mesh to meters if FS
    if toFullScale == "True" and scaling is not None:
        your_mesh.vectors = your_mesh.vectors * scaling / 1000  # [mm] to [m]
    
    # Get min/max values for the mesh (in meters)
    min_coords = your_mesh.min_ * scaling / 1000 if toFullScale != "True" else your_mesh.min_
    max_coords = your_mesh.max_ * scaling / 1000 if toFullScale != "True" else your_mesh.max_
    
    # Convert to lat/lon
    lat_lon_bounds = None
    if toLatLon == "True" and ref_coords is not None:
        # Earth radius in meters
        r_earth = 6371000  # m
        
        #Origin if not provided
        if ref_pos is None:
            ref_pos = [0, 0]     
      
        ref_pos_m = [ref_pos[0] * scaling /1000, ref_pos[1] * scaling /1000] if toFullScale != "True" else ref_pos
            
        x_range = [min_coords[0], max_coords[0]]
        y_range = [min_coords[1], max_coords[1]]
        
        # Calculate lat/lon bounds considering reference position offset
        y_min_offset = y_range[0] - ref_pos_m[1]
        y_max_offset = y_range[1] - ref_pos_m[1]
        
        lat_range = [
            ref_coords[0] + (y_min_offset / r_earth) * (180 / np.pi),  # min lat
            ref_coords[0] + (y_max_offset / r_earth) * (180 / np.pi)   # max lat
        ]
        cos_lat = np.cos(ref_coords[0] * np.pi / 180)
        x_min_offset = x_range[0] - ref_pos_m[0]
        x_max_offset = x_range[1] - ref_pos_m[0]
        
        lon_range = [
            ref_coords[1] + (x_min_offset / (r_earth * cos_lat)) * (180 / np.pi),  # min lon
            ref_coords[1] + (x_max_offset / (r_earth * cos_lat)) * (180 / np.pi)   # max lon
        ]
        
        lat_lon_bounds = {
            'lat_range': lat_range,
            'lon_range': lon_range
        }
        print(f"Lat range: {lat_range}")
        print(f"Lon range: {lon_range}")
        print(f"Reference position [x,y]: {ref_pos}")
        print(f"Reference coordinates [lat,lon]: {ref_coords}")
    
   
    fig, ax = plt.subplots(figsize=figsize)
    
    #Model vertices
    for i, face in enumerate(your_mesh.vectors):
        
        if projection == 'xy':
            # XY projection (top view)
            if toLatLon == "True" and ref_coords is not None:
               
                points = []
                for vertex in face:
                    
                    vert_x = vertex[0]    #/1000?
                    vert_y = vertex[1]   
                   
                    x_offset = vert_x - ref_pos_m[0]
                    y_offset = vert_y - ref_pos_m[1]
                    
                    lat = ref_coords[0] + (y_offset / r_earth) * (180 / np.pi)
                    lon = ref_coords[1] + (x_offset / (r_earth * cos_lat)) * (180 / np.pi)
                    
                    points.append((lon, lat))  # Longitude is x, Latitude is y
            else:
                points = [(vertex[0], vertex[1]) for vertex in face]
        #XZ projection (front view)
        elif projection == 'xz':
            points = [(vertex[0], vertex[2]) for vertex in face]
        # YZ projection (side view)
        elif projection == 'yz':
            points = [(vertex[1], vertex[2]) for vertex in face]
        # Simple diagonal view (custom angle)
        elif projection == 'diagonal':
            angle = np.pi/6
            points = [(vertex[1]*np.cos(angle) + vertex[0]*np.sin(angle),
                     vertex[2] - 0.3*vertex[0]) for vertex in face]
        else:
            raise ValueError("Projections: 'xy', 'xz', 'yz', 'diagonal'")
            
        poly = Polygon(points, closed=True, alpha=alpha,
                      facecolor=color, edgecolor=edgecolor)
        ax.add_patch(poly)
    
    if projection == 'xy':
        if toLatLon == "True" and ref_coords is not None:
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            title = 'Geographic View (Lat/Lon Projection)'
            
            if lat_lon_bounds:
                print("test")
                #ax.set_xlim(lat_lon_bounds['lon_range'])
                #ax.set_ylim(lat_lon_bounds['lat_range'])
        else:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            title = 'Top View (XY Projection)'
    elif projection == 'xz':
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        title = 'Front View (XZ Projection)'
    elif projection == 'yz':
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        title = 'Side View (YZ Projection)'
    elif projection == 'diagonal':
        ax.set_xlabel('Custom X')
        ax.set_ylabel('Custom Z')
        title = 'Diagonal View'
        
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal')
    ax.autoscale()
    
    return fig, ax

def meters2latlon(points, ref_coords, scaling=1.0, toFullScale="False", ref_pos=None):
    """
    Convert 2D points from meters to latitude/longitude coordinates.
    
    Parameters:
    -----------
    points : array-like
        Array of [x, y] coordinates in meters to convert to lat/lon
        Can be a single point [x, y] or a list of points [[x1, y1], [x2, y2], ...]
    ref_coords : list
        [lat, lon] reference coordinates corresponding to ref_pos position
    scaling : float
        Scaling factor to convert input units to meters (default: 1.0)
    toFullScale : str
        "True" to apply the scaling factor to convert units to meters
    ref_pos : list
        [x, y] position in the input coordinates corresponding to ref_coords
        Defaults to [0, 0] if not specified
        
    Returns:
    --------
    points_latlon : array-like
        Array of [lat, lon] coordinates corresponding to the input points
        Format matches the input format (single point or list of points)
    """
    import numpy as np
    
    # Check if points is a single point or a list of points
    is_single_point = not isinstance(points[0], (list, tuple, np.ndarray))
    # Convert to numpy array for easier processing
    points_array = np.array([points]) if is_single_point else np.array(points)
  
    r_earth = 6371000# m
    
    #Default ref position to origin
    if ref_pos is None:
        ref_pos = [0, 0]
    
    scale_factor = scaling/1000 if toFullScale == "True" else 1.0
    ref_pos_m = np.array(ref_pos) * scale_factor
    
 
    points_latlon = []
    
   
    cos_lat = np.cos(ref_coords[0] * np.pi / 180)
    for i, point in enumerate(points_array):
        # Scale point coordinates to meters if needed
        point_m = point * scale_factor
        
        x_offset = point_m[0] - ref_pos_m[0]
        y_offset = point_m[1] - ref_pos_m[1]
        
        lat = ref_coords[0] + (y_offset / r_earth) * (180 / np.pi)
        lon = ref_coords[1] + (x_offset / (r_earth * cos_lat)) * (180 / np.pi)
        
        points_latlon.append((lat, lon))
    
    if is_single_point:
        return points_latlon[0]
    else:
        return points_latlon
    

def add_crosses(ax, points, values=None, thresholds=None, colors=None, 
                marker='x', size=100, linewidth=2, cmap=None, add_colorbar=True,
                is_latlon=False):
    """
    Add color-coded crosses to the plot based on threshold values,
    with optional handling for geographic lat/lon data.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to add crosses to
    points : list or array
        List of (x, y) coordinates or (lat, lon) coordinates for crosses
    values : list or array, optional
        Values corresponding to each point (for coloring)
    thresholds : list, optional
        Threshold values for color changes
    colors : list, optional
        Colors corresponding to threshold ranges
    marker : str, optional
        Marker style ('x', '+', etc.)
    size : int, optional
        Size of the markers
    linewidth : int, optional
        Width of the marker lines
    cmap : str or colormap, optional
        Custom colormap (if not using thresholds/colors)
    add_colorbar : bool, optional
        Whether to add a colorbar to the plot
    is_latlon : bool, optional
        Whether the points are in (lat, lon) format, which requires special handling
        
    Returns:
    --------
    scatter : matplotlib scatter plot
        The scatter plot object for the crosses
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    
    if points is None or len(points) == 0:
        print("No points provided for crosses")
        return None
    #Lat/Lon conversion
    points_array = np.array(points)
    if is_latlon:
        x_coords = points_array[:, 1]  # Longitude as x
        y_coords = points_array[:, 0]  # Latitude as y
    else:
        # Standard x/y coordinates
        x_coords, y_coords = zip(*points)
    
    
    if values is not None:
        if thresholds is not None and colors is not None:
            # Use threshold-based coloring
            point_colors = []
            for value in values:
                for i, threshold in enumerate(thresholds):
                    if i == 0 and value < threshold:
                        point_colors.append(colors[0])
                        break
                    elif i == len(thresholds) - 1 or value < threshold:
                        point_colors.append(colors[i])
                        break
                    
            scatter = ax.scatter(x_coords, y_coords, c=point_colors, marker=marker, 
                                s=size, linewidth=linewidth)
            
            if add_colorbar:
                legend_elements = []
                
                # Add first range
                legend_elements.append(
                    Line2D([0], [0], marker=marker, color=colors[0], markerfacecolor=colors[0],
                          markersize=10, label=f'< {thresholds[0]}')
                )
                
                # Add middle ranges
                for i in range(len(thresholds) - 1):
                    legend_elements.append(
                        Line2D([0], [0], marker=marker, color=colors[i+1], markerfacecolor=colors[i+1],
                              markersize=10, label=f'{thresholds[i]} - {thresholds[i+1]}')
                    )
                
                # Add last range
                legend_elements.append(
                    Line2D([0], [0], marker=marker, color=colors[-1], markerfacecolor=colors[-1],
                          markersize=10, label=f'> {thresholds[-1]}')
                )
                
                ax.legend(handles=legend_elements, title="Value Ranges", 
                         loc='best', framealpha=0.7)
        else:
            # Use continuous coloring with a colormap
            if cmap is None:
                cmap = 'viridis'
                
            norm = Normalize(vmin=min(values), vmax=max(values))
            scatter = ax.scatter(x_coords, y_coords, c=values, marker=marker, 
                                s=size, linewidth=linewidth, cmap=cmap, norm=norm)
            
            if add_colorbar:
                plt.colorbar(scatter, ax=ax, label='Value')
    else:
        # Just add crosses without color coding
        scatter = ax.scatter(x_coords, y_coords, marker=marker, s=size, 
                            linewidth=linewidth, color='red')
    
    # For geographic plots, equal aspect ratio can distort the map at non-equatorial latitudes
    if is_latlon:
        # For lat/lon data, we typically don't want an equal aspect ratio
        # because 1 degree lat doesn't equal 1 degree lon except at the equator
        ax.set_aspect('auto')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    else:
        # For Cartesian data, equal aspect ratio makes sense
        ax.set_aspect('equal')
        
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.autoscale()
    
    return scatter


def add_crosses_geo(ax, points, values=None, thresholds=None, colors=None, 
                marker='x', size=100, linewidth=2, cmap=None, add_colorbar=True,
                is_latlon=False):
    """
    Add color-coded crosses to the plot based on threshold values,
    with special handling for geographic lat/lon data.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to add crosses to
    points : list or array
        List of (x, y) coordinates or (lat, lon) coordinates for crosses
    values : list or array, optional
        Values corresponding to each point (for coloring)
    thresholds : list, optional
        Threshold values for color changes
    colors : list, optional
        Colors corresponding to threshold ranges
    marker : str, optional
        Marker style ('x', '+', etc.)
    size : int, optional
        Size of the markers
    linewidth : int, optional
        Width of the marker lines
    cmap : str or colormap, optional
        Custom colormap (if not using thresholds/colors)
    add_colorbar : bool, optional
        Whether to add a colorbar to the plot
    is_latlon : bool, optional
        Whether the points are in (lat, lon) format, which requires special handling
        
    Returns:
    --------
    scatter : matplotlib scatter plot
        The scatter plot object for the crosses
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    
    if points is None or len(points) == 0:
        print("No points provided for crosses")
        return None
    
    # Handle points format based on whether it's lat/lon
    points_array = np.array(points)
    if is_latlon:
        # For lat/lon data, we need to make sure the order is correct
        # Most GIS uses (lat, lon) format but matplotlib expects (x=lon, y=lat)
        x_coords = points_array[:, 1]  # Longitude as x
        y_coords = points_array[:, 0]  # Latitude as y
    else:
        # Standard x/y coordinates
        x_coords, y_coords = zip(*points)
    
    # Different coloring approaches based on inputs
    if values is not None:
        if thresholds is not None and colors is not None:
            # Use threshold-based coloring
            point_colors = []
            for value in values:
                color_assigned = False
                for i, threshold in enumerate(thresholds):
                    if i == 0 and value < threshold:
                        point_colors.append(colors[0])
                        color_assigned = True
                        break
                    elif i < len(thresholds) - 1 and value >= thresholds[i] and value < thresholds[i+1]:
                        point_colors.append(colors[i+1])
                        color_assigned = True
                        break
                
                # Handle the last range (above the highest threshold)
                if not color_assigned:
                    point_colors.append(colors[-1])
                    
            scatter = ax.scatter(x_coords, y_coords, c=values, marker=marker,  #point_colors
                                s=size, linewidth=linewidth)
            
            if add_colorbar:
                legend_elements = []
                
                # Add first range
                legend_elements.append(
                    Line2D([0], [0], marker=marker, color=colors[0], markerfacecolor=colors[0],
                          markersize=10, label=f'< {thresholds[0]}')
                )
                
                # Add middle ranges
                for i in range(len(thresholds) - 1):
                    legend_elements.append(
                        Line2D([0], [0], marker=marker, color=colors[i+1], markerfacecolor=colors[i+1],
                              markersize=10, label=f'{thresholds[i]} - {thresholds[i+1]}')
                    )
                
                # Add last range
                legend_elements.append(
                    Line2D([0], [0], marker=marker, color=colors[-1], markerfacecolor=colors[-1],
                          markersize=10, label=f'> {thresholds[-1]}')
                )
                
                ax.legend(handles=legend_elements, title="Value Ranges", 
                         loc='best', framealpha=0.7)
        else:
            # Use continuous coloring with a colormap
            if cmap is None:
                cmap = 'viridis'
                
            norm = Normalize(vmin=min(values), vmax=max(values))
            scatter = ax.scatter(x_coords, y_coords, c=values, marker=marker, 
                                s=size, linewidth=linewidth, cmap=cmap, norm=norm)
            
            if add_colorbar:
                plt.colorbar(scatter, ax=ax, label='Value')
    else:
        # Just add crosses without color coding
        scatter = ax.scatter(x_coords, y_coords, marker=marker, s=size, 
                            linewidth=linewidth, color='red')
    
    # For geographic plots, equal aspect ratio can distort the map at non-equatorial latitudes
    if is_latlon:
        # For lat/lon data, we typically don't want an equal aspect ratio
        # because 1 degree lat doesn't equal 1 degree lon except at the equator
        ax.set_aspect('auto')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    else:
        # For Cartesian data, equal aspect ratio makes sense
        ax.set_aspect('equal')
        
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return scatter

"""
def add_crosses(ax, points, values=None, thresholds=None, colors=None, 
                marker='x', size=100, linewidth=2, cmap=None, add_colorbar=True):
    
    Add color-coded crosses to the plot based on threshold values.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to add crosses to
    points : list or array
        List of (x, y) coordinates for crosses
    values : list or array, optional
        Values corresponding to each point (for coloring)
    thresholds : list, optional
        Threshold values for color changes
    colors : list, optional
        Colors corresponding to threshold ranges
    marker : str, optional
        Marker style ('x', '+', etc.)
    size : int, optional
        Size of the markers
    linewidth : int, optional
        Width of the marker lines
    cmap : str or colormap, optional
        Custom colormap (if not using thresholds/colors)
    add_colorbar : bool, optional
        Whether to add a colorbar to the plot
        
    Returns:
    --------
    scatter : matplotlib scatter plot
        The scatter plot object for the crosses
    
    if points is None or len(points) == 0:
        print("No points provided for crosses")
        return None
    
    x_coords, y_coords = zip(*points)
    
    # Different coloring approaches based on inputs
    if values is not None:
        if thresholds is not None and colors is not None:
            # Use threshold-based coloring
            point_colors = []
            for value in values:
                for i, threshold in enumerate(thresholds):
                    if i == 0 and value < threshold:
                        point_colors.append(colors[0])
                        break
                    elif i == len(thresholds) - 1 or value < threshold:
                        point_colors.append(colors[i])
                        break
                    
            scatter = ax.scatter(x_coords, y_coords, c=point_colors, marker=marker, 
                                s=size, linewidth=linewidth)
            
            # Add a custom legend for threshold colors
            if add_colorbar:
                from matplotlib.lines import Line2D
                legend_elements = []
                
                # Add first range
                legend_elements.append(
                    Line2D([0], [0], marker=marker, color=colors[0], markerfacecolor=colors[0],
                          markersize=10, label=f'< {thresholds[0]}')
                )
                
                # Add middle ranges
                for i in range(len(thresholds) - 1):
                    legend_elements.append(
                        Line2D([0], [0], marker=marker, color=colors[i+1], markerfacecolor=colors[i+1],
                              markersize=10, label=f'{thresholds[i]} - {thresholds[i+1]}')
                    )
                
                # Add last range
                legend_elements.append(
                    Line2D([0], [0], marker=marker, color=colors[-1], markerfacecolor=colors[-1],
                          markersize=10, label=f'> {thresholds[-1]}')
                )
                
                ax.legend(handles=legend_elements, title="Value Ranges", 
                         loc='best', framealpha=0.7)
        else:
            # Use continuous coloring with a colormap
            if cmap is None:
                cmap = 'viridis'
                
            norm = Normalize(vmin=min(values), vmax=max(values))
            scatter = ax.scatter(x_coords, y_coords, c=values, marker=marker, 
                                s=size, linewidth=linewidth, cmap=cmap, norm=norm)
            
            if add_colorbar:
                plt.colorbar(scatter, ax=ax, label='Value')
    else:
        # Just add crosses without color coding
        scatter = ax.scatter(x_coords, y_coords, marker=marker, s=size, 
                            linewidth=linewidth, color='red')
        
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal')
    ax.autoscale()
    
    return scatter
"""


def add_crosses_3d(ax, points, values=None, thresholds=None, colors=None,
                  marker='x', size=100, linewidth=2, cmap=None, add_colorbar=True):
    """
    Add color-coded crosses to a 3D plot based on threshold values.
    
    Parameters:
    -----------
    ax : matplotlib 3D axis
        The axis to add crosses to
    points : list or array
        List of (x, y, z) coordinates for crosses
    values : list or array, optional
        Values corresponding to each point (for coloring)
    thresholds : list, optional
        Threshold values for color changes
    colors : list, optional
        Colors corresponding to threshold ranges
    marker : str, optional
        Marker style ('x', '+', etc.)
    size : int, optional
        Size of the markers
    linewidth : int, optional
        Width of the marker lines
    cmap : str or colormap, optional
        Custom colormap (if not using thresholds/colors)
    add_colorbar : bool, optional
        Whether to add a colorbar to the plot
        
    Returns:
    --------
    scatter : matplotlib scatter plot
        The scatter plot object for the crosses
    """
    if points is None or len(points) == 0:
        print("No points provided for crosses")
        return None
    
    # For 3D points, we need x, y, and z coordinates
    if len(points[0]) == 3:
        x_coords, y_coords, z_coords = zip(*points)
    else:
        # For 2D points, set z to zeros
        x_coords, y_coords = zip(*points)
        z_coords = [0] * len(x_coords)
    
    # Different coloring approaches based on inputs
    if values is not None:
        if thresholds is not None and colors is not None:
            # Use threshold-based coloring
            point_colors = []
            for value in values:
                color_assigned = False
                for i, threshold in enumerate(thresholds):
                    if i == 0 and value < threshold:
                        point_colors.append(colors[0])
                        color_assigned = True
                        break
                    elif i < len(thresholds) - 1 and threshold <= value < thresholds[i+1]:
                        point_colors.append(colors[i+1])
                        color_assigned = True
                        break
                    elif i == len(thresholds) - 1 and value >= threshold:
                        point_colors.append(colors[-1])
                        color_assigned = True
                        break
                # If no threshold matched, use the last color
                if not color_assigned:
                    point_colors.append(colors[-1])
            
            scatter = ax.scatter(x_coords, y_coords, z_coords, c=point_colors, marker=marker,
                               s=size, linewidth=linewidth)
            
            if add_colorbar:
                from matplotlib.lines import Line2D
                legend_elements = []
                
                # Add first range
                legend_elements.append(
                    Line2D([0], [0], marker=marker, color=colors[0], markerfacecolor=colors[0],
                          markersize=10, label=f'< {thresholds[0]}')
                )
                
                # Add middle ranges
                for i in range(len(thresholds) - 1):
                    legend_elements.append(
                        Line2D([0], [0], marker=marker, color=colors[i+1], markerfacecolor=colors[i+1],
                              markersize=10, label=f'{thresholds[i]} - {thresholds[i+1]}')
                    )
                
                # Add last range
                legend_elements.append(
                    Line2D([0], [0], marker=marker, color=colors[-1], markerfacecolor=colors[-1],
                          markersize=10, label=f'> {thresholds[-1]}')
                )
                
                ax.legend(handles=legend_elements, title="Value Ranges",
                        loc='best', framealpha=0.7)
        else:
            # Use continuous coloring with a colormap
            if cmap is None:
                cmap = 'viridis'
            
            norm = Normalize(vmin=min(values), vmax=max(values))
            scatter = ax.scatter(x_coords, y_coords, z_coords, c=values, marker=marker,
                               s=size, linewidth=linewidth, cmap=cmap, norm=norm)
            
            if add_colorbar:
                plt.colorbar(scatter, ax=ax, label='Value')
    else:
        # Just add crosses without color coding
        scatter = ax.scatter(x_coords, y_coords, z_coords, marker=marker, s=size,
                           linewidth=linewidth, color='red')
    
    return scatter


def show_multiple_projections(stl_file, points_dict=None, values_dict=None, 
                             thresholds=None, colors=None):
    """
    Show all three standard projections of an STL file side by side,
    with optional color-coded crosses.
    
    Parameters:
    -----------
    stl_file : str
        Path to the STL file
    points_dict : dict, optional
        Dictionary with projection keys ('xy', 'xz', 'yz') and point coordinates
    values_dict : dict, optional
        Dictionary with projection keys and values for coloring points
    thresholds : list, optional
        Threshold values for color coding
    colors : list, optional
        Colors corresponding to thresholds
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    your_mesh = mesh.Mesh.from_file(stl_file)
    
    projections = ['xy', 'xz', 'yz']
    titles = ['Top View (XY)', 'Front View (XZ)', 'Side View (YZ)']
    
    for i, (proj, title) in enumerate(zip(projections, titles)):
        ax = axes[i]
        
        #Model vertices
        for face in your_mesh.vectors:
            if proj == 'xy':
                points = [(vertex[0], vertex[1]) for vertex in face]
            elif proj == 'xz':
                points = [(vertex[0], vertex[2]) for vertex in face]
            else:  # yz
                points = [(vertex[1], vertex[2]) for vertex in face]
            
            poly = Polygon(points, closed=True, alpha=0.7, 
                          facecolor='lightblue', edgecolor='blue')
            ax.add_patch(poly)
        
        # Add crosses if provided for this projection
        if points_dict is not None and proj in points_dict:
            proj_points = points_dict[proj]
            proj_values = None if values_dict is None else values_dict.get(proj, None)
            
            add_crosses(ax, proj_points, values=proj_values, 
                       thresholds=thresholds, colors=colors)
        
        if proj == 'xy':
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        elif proj == 'xz':
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
        else:  # yz
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
            
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_aspect('equal')
        ax.autoscale()
    
    plt.tight_layout()
    return fig, axes



def plot_stl_3d(stl_file, color_map=None, default_color='gray', 
                x_range=None, y_range=None, z_range=None, 
                clip_polygons=True,elev=30,azim=45,toFullScale="False",scaling=None):
    """
    Create a 3D visualization of an STL file with optional region filtering.
    
    Parameters:
    -----------
    stl_file : str
        Path to the STL file
    color_map : dict, optional
        Dictionary mapping face indices to colors
    default_color : str, optional
        Default color for faces not in the color map
    x_range, y_range, z_range : tuple, optional
        (min, max) ranges to filter polygons (None = no filtering)
    clip_polygons : bool, optional
        If True, clip polygons to the specified ranges
        If False, only show polygons whose centroid is in range
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis
        The figure and 3D axis containing the plot
    """
    your_mesh = mesh.Mesh.from_file(stl_file)
   
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    #Transform to FS
    if toFullScale == "True" and scaling is not None:
        your_mesh.vectors = your_mesh.vectors * scaling / 1000  # [mm] to [m]
    
    # If no color map provided, use default color for all faces
    if color_map is None:
        color_map = {}

    
    mesh_min = your_mesh.points.reshape(-1, 3).min(axis=0)
    mesh_max = your_mesh.points.reshape(-1, 3).max(axis=0)
    #If range overgiven use this instead of mesh range
    x_min = x_range[0] if x_range is not None else mesh_min[0]
    x_max = x_range[1] if x_range is not None else mesh_max[0]
    y_min = y_range[0] if y_range is not None else mesh_min[1]
    y_max = y_range[1] if y_range is not None else mesh_max[1]
    z_min = z_range[0] if z_range is not None else mesh_min[2]
    z_max = z_range[1] if z_range is not None else mesh_max[2]
    
   
    polygons = []
    face_colors = []
    for i, face in enumerate(your_mesh.vectors):
        #Center of of the faces
        centroid = np.mean(face, axis=0)
        # Check if this face should be included based on ranges
        #if (x_range is not None and (centroid[0] < x_min or centroid[0] > x_max)):
        #    continue
        #if (y_range is not None and (centroid[1] < y_min or centroid[1] > y_max)):
        #    continue
        #if (z_range is not None and (centroid[2] < z_min or centroid[2] > z_max)):
        #    continue
        if ((centroid[0] < x_min or centroid[0] > x_max)):
            continue
        if ((centroid[1] < y_min or centroid[1] > y_max)):
            continue
        if ((centroid[2] < z_min or centroid[2] > z_max)):
            continue
        
        polygons.append(face)
        
        if i in color_map:
            face_colors.append(color_map[i])
        else:
            face_colors.append(default_color)
    
    poly = Poly3DCollection(polygons, alpha=0.7)
    poly.set_facecolor(face_colors)
    poly.set_edgecolor('black')
    ax.add_collection3d(poly)
    
    # Set view limits and angle
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.view_init(elev=elev, azim=azim)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D STL Viewer')
    
    
    return fig, ax



def add_velocity_field(ax, points, values, components=None, 
                 interpolate=True, grid_density=30, transparencyFactor=0.5, cmap='viridis',
                 add_colorbar=True, show_vectors=False, vector_scale=20, 
                 vector_spacing=5, vector_color='black', vector_width=0.5, show_contour_lines=False,
                 is_latlon=False, zorder=5):
    """
    Add a 2D color-coded velocity field to the plot, with optional vector overlay.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to add the velocity field to
    points : list or array
        List of (x, y) or (lat, lon) coordinates for velocity data points
    values : list or array
        Speed values corresponding to each point (magnitude)
    components : list or array, optional
        List of [u, v] components for each point, required if show_vectors=True
    interpolate : bool, optional
        Whether to interpolate velocities between points (True) or just show at discrete points (False)
    grid_density : int, optional
        Number of grid points in each dimension for interpolation
    transparencyFactor : float, optional
        Transparency of the velocity field, low=high transparency, high=low transparency
    cmap : str or colormap, optional
        Colormap for the velocity field
    add_colorbar : bool, optional
        Whether to add a colorbar to the plot
    show_vectors : bool, optional
        Whether to overlay velocity vectors on the field
    vector_scale : float, optional
        Scaling factor for vector arrows
    vector_spacing : int, optional
        Spacing between vectors when using interpolation (higher = fewer vectors)
    vector_color : str, optional
        Color for the vector arrows
    vector_width : float, optional
        Line width for the vector arrows
    is_latlon : bool, optional
        Whether the points are in (lat, lon) format
    zorder : int, optional
        Z-order for drawing (lower number = drawn earlier/underneath)
        
    Returns:
    --------
    field : matplotlib contourf or scatter
        The field object for the velocities
    vectors : matplotlib quiver or None
        The vector field object if show_vectors=True, otherwise None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    
    if points is None or len(points) == 0 or values is None or len(values) == 0:
        print("No valid points or values provided")
        return None, None

    points_array = np.array(points)
    values_array = np.array(values)
    
    # Handle coordinate format for plotting
    if is_latlon:
        x_coords = points_array[:, 1]  # Longitude as x
        y_coords = points_array[:, 0]  # Latitude as y
    else:
        # Standard x/y coordinates
        x_coords = points_array[:, 0]
        y_coords = points_array[:, 1]
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    
    field = None
    vectors = None
    if interpolate:
        # Create a regular grid for interpolation
        xi = np.linspace(xlim[0], xlim[1], grid_density)
        yi = np.linspace(ylim[0], ylim[1], grid_density)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolate the speed values onto the grid
        Zi = griddata((x_coords, y_coords), values_array, (Xi, Yi), method='cubic', fill_value=np.nan)

        # Create a filled contour plot with the velocity field
        if show_contour_lines:
            field = ax.contourf(Xi, Yi, Zi, levels=50, cmap=cmap, alpha=transparencyFactor, zorder=zorder)
        else:
            # Remove visible isolines by setting linewidths=0 and antialiased=True
            field = ax.contourf(Xi, Yi, Zi, levels=50, cmap=cmap, alpha=transparencyFactor, zorder=zorder,
                               linewidths=0, antialiased=True)
        
        #Filled contour plot with the velocity field
       # field = ax.contourf(Xi, Yi, Zi, levels=50, cmap=cmap, alpha=alpha, zorder=zorder)
        
        if add_colorbar:
            cbar = plt.colorbar(field, ax=ax, label='Speed')
        
        #Add vector overlaying
        if show_vectors and components is not None:
            # Prepare vector components
            if is_latlon:
                u_components = np.array([comp[0] for comp in components])
                v_components = np.array([comp[1] for comp in components])
            else:
                u_components = np.array([comp[0] for comp in components])
                v_components = np.array([comp[1] for comp in components])
            
            # Interpolate the vector components onto sparser grid to avoid overcrowding
            step = vector_spacing
            Xi_sparse = Xi[::step, ::step]
            Yi_sparse = Yi[::step, ::step]
            # Interpolate u and v components separately
            Ui = griddata((x_coords, y_coords), u_components, (Xi_sparse, Yi_sparse), method='cubic', fill_value=0)
            Vi = griddata((x_coords, y_coords), v_components, (Xi_sparse, Yi_sparse), method='cubic', fill_value=0)
            
            # Create vector field
            vectors = ax.quiver(Xi_sparse, Yi_sparse, Ui, Vi, 
                              scale=vector_scale, color=vector_color, 
                              width=vector_width, zorder=zorder+1)
    else:
        # For discrete points, use a scatter plot
        field = ax.scatter(x_coords, y_coords, c=values_array, cmap=cmap, 
                         alpha=transparencyFactor, s=50, zorder=zorder)
        
    
        if add_colorbar:
            cbar = plt.colorbar(field, ax=ax, label='Speed')
        
        # Add vector overlay
        if show_vectors and components is not None:
            # Prepare vector components
            if is_latlon:
                u_components = np.array([comp[0] for comp in components])
                v_components = np.array([comp[1] for comp in components])
            else:
                u_components = np.array([comp[0] for comp in components])
                v_components = np.array([comp[1] for comp in components])
            
            # Create vector field at discrete points
            vectors = ax.quiver(x_coords, y_coords, u_components, v_components, 
                              scale=vector_scale, color=vector_color, 
                              width=vector_width, zorder=zorder+1)
    
    return field, vectors


if __name__ == "__main__":
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert STL to 2D shaded plot with optional markers')
    parser.add_argument('stl_file', help='Path to the STL file')
    parser.add_argument('--projection', choices=['xy', 'xz', 'yz', 'all'], default='xy',
                        help='Projection plane (default: xy)')
    parser.add_argument('--output', help='Output file path (default: display only)')
    parser.add_argument('--points', help='CSV file with marker coordinates (x,y,value)')
    parser.add_argument('--thresholds', help='Comma-separated threshold values (e.g., 10,20,30)')
    parser.add_argument('--colors', help='Comma-separated colors (e.g., red,yellow,green,blue)')
    
    args = parser.parse_args()
    
    # Load points and values if provided
    points_dict = {}
    values_dict = {}
    
    if args.points:
        import csv
        with open(args.points, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header row
            
            xy_points = []
            xz_points = []
            yz_points = []
            xy_values = []
            xz_values = []
            yz_values = []
            
            for row in reader:
                if len(row) >= 3:  # At least x, y, z
                    x, y, z = float(row[0]), float(row[1]), float(row[2])
                    
                    # Add points to respective projections
                    xy_points.append((x, y))
                    xz_points.append((x, z))
                    yz_points.append((y, z))
                    
                    # Add values if provided
                    if len(row) >= 4:
                        value = float(row[3])
                        xy_values.append(value)
                        xz_values.append(value)
                        yz_values.append(value)
            
            points_dict = {'xy': xy_points, 'xz': xz_points, 'yz': yz_points}
            if xy_values:
                values_dict = {'xy': xy_values, 'xz': xz_values, 'yz': yz_values}
    
    # Parse thresholds and colors
    thresholds = None
    colors = None
    
    if args.thresholds:
        thresholds = [float(t) for t in args.thresholds.split(',')]
        
    if args.colors:
        colors = args.colors.split(',')
    
    # Generate the plot(s)
    if args.projection == 'all':
        fig, axes = show_multiple_projections(
            args.stl_file, 
            points_dict=points_dict, 
            values_dict=values_dict if values_dict else None,
            thresholds=thresholds,
            colors=colors
        )
    else:
        fig, ax = stl_to_2d_plot(args.stl_file, projection=args.projection)
        
        # Add crosses if points are provided
        if args.projection in points_dict:
            add_crosses(
                ax, 
                points_dict[args.projection], 
                values=values_dict.get(args.projection) if values_dict else None,
                thresholds=thresholds,
                colors=colors
            )
    
    # Save or display the plot
    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {args.output}")
    else:
        plt.show()
    """

    
    

def load_avg_file(file_path):
    """
    Load concentration measurement data from a specifically formatted text file
    into a dictionary.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        dict: Dictionary containing metadata and concentration data
    """
 
    with open(file_path, 'r') as file:
        lines = file.readlines()
    result = {
        'metadata': {},
        'data': []
    }
    
    # Extract metadata and find where data begins
    data_section = False
    column_names = None
    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        
        # Process metadata lines
        if line.startswith('#'):
            # Skip general header
            if 'General concentration' in line:
                continue
            # Extract column headers
            if line.startswith('# "'):
                column_names = [name.strip('"') for name in line[2:].split('" "')]
                data_section = True
                continue
            # Process other metadata
            line = line[1:].strip()  # Remove the '#' prefix
            
            # Handle Variables line which contains multiple key-value pairs
            if line.startswith('Variables:'):
                variables_str = line[len('Variables:'):].strip()
                var_pattern = re.compile(r'([^:,]+):\s*([^,\[]+)(?:\s*\[([^\]]+)\])?')
                for match in var_pattern.finditer(variables_str):
                    key, value, unit = match.groups()
                    key = key.strip()
                    value = value.strip()
                    
                    # Convert value to float if possible
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    
                    if unit:
                        result['metadata'][key] = {'value': value, 'unit': f'[{unit}]'}
                    else:
                        result['metadata'][key] = value
            
            # Handle other metadata lines with key-value pairs
            elif ':' in line:
                key, value = line.split(':', 1)
                value = value.strip()
                key = key.strip()
                
                # Extract unit if available
                unit_match = re.search(r'\[(.*?)\]', value)
                if unit_match:
                    unit = unit_match.group(0)
                    value_only = value[:value.find('[')].strip()
                    
                    # Convert value to float if possible
                    try:
                        value_only = float(value_only)
                    except ValueError:
                        pass
                        
                    result['metadata'][key] = {'value': value_only, 'unit': unit}
                else:
                    result['metadata'][key] = value
        
        # Process data rows
        elif data_section and column_names:
            values = line.split()
            # Only process if we have the right number of values
            if len(values) == len(column_names):
                row_data = {}
                for i, col_name in enumerate(column_names):
                    try:
                        row_data[col_name] = float(values[i])
                    except ValueError:
                        row_data[col_name] = values[i]
                        
                result['data'].append(row_data)
    
    return result



import csv
import re
import os

def load_stats_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    result = {'metadata': {}, 'data': []}
    data_section = False
    column_names = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('#'):
            if 'General concentration' in line:
                continue
                
            if line.startswith('# "'):
                header_line = line[2:]
                column_names = []
                parts = header_line.split('"')
                for part in parts:
                    if part.strip() and not part.strip().startswith(' '):
                        column_names.append(part.strip())
                
                if not column_names:
                    column_names = [name.strip('"') for name in header_line.split('" "')]
                
                data_section = True
                continue
                
            line = line[1:].strip()
            
            if line.startswith('Variables:'):
                variables_str = line[len('Variables:'):].strip()
                var_pattern = re.compile(r'([^:,]+):\s*([^,\[]+)(?:\s*\[([^\]]+)\])?')
                for match in var_pattern.finditer(variables_str):
                    key, value, unit = match.groups()
                    key = key.strip()
                    value = value.strip()
                    
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    
                    if unit:
                        result['metadata'][key] = {'value': value, 'unit': f'[{unit}]'}
                    else:
                        result['metadata'][key] = value
                        
            elif ':' in line:
                key, value = line.split(':', 1)
                value = value.strip()
                key = key.strip()
                
                unit_match = re.search(r'\[(.*?)\]', value)
                if unit_match:
                    unit = unit_match.group(0)
                    value_only = value[:value.find('[')].strip()
                    
                    try:
                        value_only = float(value_only)
                    except ValueError:
                        pass
                    
                    result['metadata'][key] = {'value': value_only, 'unit': unit}
                else:
                    result['metadata'][key] = value
                    
        elif data_section and column_names:
            values = line.split()
            
            if len(values) >= 3:
                row_data = {}
                
                for i, value in enumerate(values):
                    if i < len(column_names):
                        col_name = column_names[i]
                    else:
                        col_name = f'stat_col_{i}'
                    
                    try:
                        row_data[col_name] = float(value)
                    except ValueError:
                        row_data[col_name] = value
                
                result['data'].append(row_data)
    
    return result

import csv
import re
import os

def load_stats_file(file_path):
    """Load stats file with enhanced parsing for all column types"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    result = {'metadata': {}, 'data': []}
    data_section = False
    column_names = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('#'):
            if 'General concentration' in line:
                continue
                
            if line.startswith('# "'):
                # Enhanced parsing for complex column headers
                header_line = line[2:]  # Remove '# '
                column_names = []
                
                # Split by quote patterns to handle complex headers
                parts = re.findall(r'"([^"]+)"', header_line)
                if parts:
                    column_names = parts
                else:
                    # Fallback parsing
                    parts = header_line.split('"')
                    for part in parts:
                        if part.strip() and not part.strip().startswith(' '):
                            column_names.append(part.strip())
                
                data_section = True
                continue
                
            line = line[1:].strip()
            
            if line.startswith('Variables:'):
                variables_str = line[len('Variables:'):].strip()
                var_pattern = re.compile(r'([^:,]+):\s*([^,\[]+)(?:\s*\[([^\]]+)\])?')
                for match in var_pattern.finditer(variables_str):
                    key, value, unit = match.groups()
                    key = key.strip()
                    value = value.strip()
                    
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    
                    if unit:
                        result['metadata'][key] = {'value': value, 'unit': f'[{unit}]'}
                    else:
                        result['metadata'][key] = value
                        
            elif ':' in line:
                key, value = line.split(':', 1)
                value = value.strip()
                key = key.strip()
                
                unit_match = re.search(r'\[(.*?)\]', value)
                if unit_match:
                    unit = unit_match.group(0)
                    value_only = value[:value.find('[')].strip()
                    
                    try:
                        value_only = float(value_only)
                    except ValueError:
                        pass
                    
                    result['metadata'][key] = {'value': value_only, 'unit': unit}
                else:
                    result['metadata'][key] = value
                    
        elif data_section and column_names:
            values = line.split()
            
            if len(values) >= 3:
                row_data = {}
                
                for i, value in enumerate(values):
                    if i < len(column_names):
                        col_name = column_names[i]
                    else:
                        col_name = f'col_{i}'
                    
                    try:
                        row_data[col_name] = float(value)
                    except ValueError:
                        row_data[col_name] = value
                
                result['data'].append(row_data)
    
    return result



def batch_combine_data(file_path, file_names, output_path, 
                      folder_combined="combined_data", 
                      output_filename="all_measurements_combined.csv",
                      file_type="both"):
    """
    Combine data from multiple files with option to choose file type
    
    Args:
        file_path: Path to directory containing data files
        file_names: List of file names to process
        output_path: Path for output directory
        folder_combined: Name of output subdirectory
        output_filename: Name of output CSV file
        file_type: "avg", "stats", or "both" - determines which files to read
    """
    combined_dir = os.path.join(output_path, folder_combined)
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)
    
    all_avg_columns = set()
    all_stats_columns = set()
    
    print(f"Analyzing files for file_type: {file_type}...")
    
    for file_name in file_names:
        try:
            # Process avg files
            if file_type in ["avg", "both"]:
                avg_file_path = os.path.join(file_path, file_name)
                if os.path.exists(avg_file_path):
                    avg_data = load_avg_file(avg_file_path)
                    if avg_data['data']:
                        all_avg_columns.update(avg_data['data'][0].keys())
            
            # Process stats files
            if file_type in ["stats", "both"]:
                if "_avg_" in file_name:
                    stats_file_name = file_name.replace('_avg_', '_stats_')
                else:
                    # If no _avg_ pattern, try to find corresponding stats file
                    base_name = file_name.replace('.ts#0', '')
                    stats_file_name = f"_stats_{base_name}.ts#0"
                
                stats_file_path = os.path.join(file_path, stats_file_name)
                if os.path.exists(stats_file_path):
                    stats_data = load_stats_file(stats_file_path)
                    if stats_data['data']:
                        all_stats_columns.update(stats_data['data'][0].keys())
                        
        except Exception as e:
            print(f"Warning: Could not analyze {file_name}: {e}")
    
    # Build header
    header = ['Filename', 'X_fs [m]', 'Y_fs [m]', 'Z_fs [m]']
    
    if file_type in ["avg", "both"]:
        for col in sorted(all_avg_columns):
            header.append(f'Avg_{col}')
    
    if file_type in ["stats", "both"]:
        for col in sorted(all_stats_columns):
            header.append(f'Stats_{col}')
    
    # Write combined data
    output_file = os.path.join(combined_dir, output_filename)
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for file_name in file_names:
            try:
                row = [file_name]
                
                # Load avg data
                avg_data = None
                if file_type in ["avg", "both"]:
                    avg_file_path = os.path.join(file_path, file_name)
                    if os.path.exists(avg_file_path):
                        avg_data = load_avg_file(avg_file_path)
                
                # Load stats data
                stats_data = None
                if file_type in ["stats", "both"]:
                    if "_avg_" in file_name:
                        stats_file_name = file_name.replace('_avg_', '_stats_')
                    else:
                        base_name = file_name.replace('.ts#0', '')
                        stats_file_name = f"_stats_{base_name}.ts#0"
                    
                    stats_file_path = os.path.join(file_path, stats_file_name)
                    if os.path.exists(stats_file_path):
                        stats_data = load_stats_file(stats_file_path)
                
                # Extract coordinates (prioritize available data)
                x = y = z = 0.0
                metadata_source = avg_data if avg_data else stats_data
                if metadata_source and 'metadata' in metadata_source:
                    x = metadata_source['metadata'].get('x (measurement relativ to source)', {}).get('value', 0.0)
                    y = metadata_source['metadata'].get('y (measurement relativ to source)', {}).get('value', 0.0)
                    z = metadata_source['metadata'].get('z (measurement relativ to source)', {}).get('value', 0.0)
                
                # Convert from mm to m
                row.extend([x/1000, y/1000, z/1000])
                
                # Add avg data columns
                if file_type in ["avg", "both"]:
                    avg_row_data = {}
                    if avg_data and avg_data['data']:
                        avg_row_data = avg_data['data'][0]
                    
                    for col in sorted(all_avg_columns):
                        row.append(avg_row_data.get(col, 0.0))
                
                # Add stats data columns
                if file_type in ["stats", "both"]:
                    stats_row_data = {}
                    if stats_data and stats_data['data']:
                        stats_row_data = stats_data['data'][0]
                    
                    for col in sorted(all_stats_columns):
                        row.append(stats_row_data.get(col, 0.0))
                
                writer.writerow(row)
                
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                error_row = [file_name] + [0.0] * (len(header) - 1)
                writer.writerow(error_row)
    
    print(f"Combined {len(file_names)} files saved to: {output_file}")
    return output_file

def load_combined_data_from_csv(file_path):
    """Load combined data from CSV file"""
    filenames = []
    points = []
    avg_data = {}
    stats_data = {}
    
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            try:
                filenames.append(row['Filename'])
                
                x = float(row['X_fs [m]'])
                y = float(row['Y_fs [m]'])
                z = float(row['Z_fs [m]'])
                points.append((x, y, z))
                
                for key, value in row.items():
                    if key.startswith('Avg_'):
                        param_name = key[4:]
                        if param_name not in avg_data:
                            avg_data[param_name] = []
                        avg_data[param_name].append(float(value))
                    elif key.startswith('Stats_'):
                        param_name = key[6:]
                        if param_name not in stats_data:
                            stats_data[param_name] = []
                        stats_data[param_name].append(float(value))
                        
            except (ValueError, KeyError) as e:
                print(f"Skipping invalid row: {e}")
                
    return filenames, points, avg_data, stats_data


def load_data_from_csv(file_path):
    """
    Load data from a CSV file returns points and values lists for printing/or further analysis.
    
    Args:
        filename (str): Path to the CSV file
        
    Returns:
        tuple: (points, values) where:
            - points is a list of (x,y,z) tuples
            - values is a list of concentration values
    """
    points = []
    values = []
    
    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Check if we have enough elements in the row
            if len(row) >= 4:
                try:
                    # Parse data, assuming the format matches X_fs [m],Y_fs [m],Z_fs [m],C* [-]
                    x = float(row[0])
                    y = float(row[1])
                    z = float(row[2])
                    c = float(row[3])
                    
                    # Add to our lists
                    points.append((x, y, z))
                    values.append(c)
                except ValueError:
                    # Skip rows that can't be converted to float
                    print(f"Skipping invalid row: {row}")
    
    return points, values

    # Example usage:
    # data_dict = load_concentration_data('concentration_data.txt')
    # print(data_dict)
