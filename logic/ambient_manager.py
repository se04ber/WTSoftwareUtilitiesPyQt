"""
logic/ambient_manager.py
Ambient conditions handling logic for the application.
"""
from typing import Dict, Any, Optional
import pandas as pd

class AmbientManager:
    """
    Handles loading and applying ambient conditions to data.
    """
    def load_ambient_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load ambient conditions from a CSV file.
        Args:
            file_path: Path to the ambient CSV file.
        Returns:
            Dictionary of ambient conditions, or None if not found.
        """
        if not file_path:
            return None
        
        try:
            df = pd.read_csv(file_path)
            
            # Look for filename column with different possible names
            filename_col = None
            for col in df.columns:
                if col.lower() in ['filename', 'file', 'name', 'file_name', 'file name']:
                    filename_col = col
                    break
            
            if filename_col is None:
                print(f"Warning: No filename column found in {file_path}")
                print(f"Available columns: {list(df.columns)}")
                return None
            
            # Create dictionary with filename as key
            ambient_dict = {}
            for _, row in df.iterrows():
                filename = str(row[filename_col]).strip()
                if filename and filename != 'nan':
                    ambient_dict[filename] = row.to_dict()
            
            print(f"Loaded ambient conditions for {len(ambient_dict)} files")
            return ambient_dict
            
        except Exception as e:
            print(f"Error loading ambient file {file_path}: {e}")
            return None
