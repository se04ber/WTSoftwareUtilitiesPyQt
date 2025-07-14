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
        df = pd.read_csv(file_path)
        if 'Filename' not in df.columns:
            return None
        return {row['Filename']: row for _, row in df.iterrows()}
