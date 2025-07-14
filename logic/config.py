"""
logic/config.py
Configuration and defaults for the application.
"""
from typing import Dict, Any
import json
import os

class AppConfig:
    """
    Handles loading and saving of application configuration and defaults.
    """
    DEFAULTS = {
        'theme': 'dark',
        'last_data_dir': '',
        'last_output_dir': '',
        # Add more defaults as needed
    }

    def __init__(self, config_path: str = 'app_config.json'):
        self.config_path = config_path
        self.config = self.DEFAULTS.copy()
        self.load()

    def load(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config.update(json.load(f))
            except Exception:
                pass

    def save(self):
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception:
            pass

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        self.config[key] = value
        self.save()
