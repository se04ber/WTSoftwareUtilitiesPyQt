"""
logic/data_loader.py
Data loading and preprocessing logic for the application.
"""
from typing import List
import pandas as pd
import os
import sys
import numpy as np

# Add the windtunnel directory to the path to import the utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'windtunnel', 'concentration'))
from utils import load_avg_file, load_stats_file

class DataLoader:
    """
    Handles loading and preprocessing of measurement data files.
    """
    def load_files(self, file_paths: List[str]) -> List[pd.DataFrame]:
        """
        Load and preprocess data from the given file paths.
        Args:
            file_paths: List of file paths to load.
        Returns:
            List of loaded pandas DataFrames.
        """
        data = []
        for path in file_paths:
            # Check if this is a raw data file (.txt.ts#) or processed file
            if '.txt.ts#' in path:
                # Raw data file - skip specialized loaders
                file_data = None
            else:
                # Try to load as stats file first, then as avg file
                file_data = load_stats_file(path)
                if file_data is None:
                    file_data = load_avg_file(path)
            
            if file_data:
                # Convert the dict to a DataFrame with a single row
                df = pd.DataFrame([file_data])
                data.append(df)
            else:
                # Fallback: try to read as raw data file (space-separated values)
                try:
                    # Read as space-separated values
                    raw_data = np.loadtxt(path)
                    if len(raw_data.shape) == 1:
                        # Single row of data
                        raw_data = raw_data.reshape(1, -1)
                    
                    # Create DataFrame with generic column names
                    df = pd.DataFrame(raw_data, columns=[f'col_{i}' for i in range(raw_data.shape[1])])
                    df['Filename'] = os.path.basename(path)
                    data.append(df)
                except Exception as e:
                    print(f"Warning: Could not load file {path}: {e}")
                    # Create an empty DataFrame with filename
                    df = pd.DataFrame({'Filename': [os.path.basename(path)]})
                    data.append(df)
        
        return data
