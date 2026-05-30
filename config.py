"""

Global configuration loader for both apps Webapp-0 & 1
"""
import os
import sys
import tomllib
CONFIG_PATH = "/Explainable-Image-Search/app_config.toml"
CONFIG = {}

try:

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "rb") as f:
            CONFIG = tomllib.load(f)
    else:
        raise FileNotFoundError
except ImportError:

    try:
        import toml
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                CONFIG = toml.load(f)
        else:
            raise FileNotFoundError
            
    except ImportError:
        print("Error: run 'pip install toml' to parse configurations on this Python version.")
        sys.exit(1)
        
except FileNotFoundError:
    
    print(f"Error: '{CONFIG_PATH}' file is     missing in your project directory.")
    sys.exit(1)

def get_config(key, default=None):
    """
        fetch values from the configuration dictionary."""
    return CONFIG.get(key, default)
