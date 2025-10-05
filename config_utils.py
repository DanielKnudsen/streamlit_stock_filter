import os
from pathlib import Path
from typing import Any, Dict, Optional
from io_utils import load_yaml

def load_config(config_file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load configuration from a YAML file.
    Args:
        config_file_path (str): Path to the YAML config file.
    Returns:
        Optional[Dict[str, Any]]: Configuration dictionary or None if not found.
    """
    if not Path(config_file_path).exists():
        print(f"Config file {config_file_path} not found.")
        return None
    return load_yaml(config_file_path)


# Bestäm miljön (default till 'local')
ENVIRONMENT = os.getenv('ENVIRONMENT', 'local')
FETCH_DATA = os.getenv('FETCH_DATA', 'Yes')
config = load_config("rank-config.yaml")
# Välj CSV-path
CSV_PATH = Path('data') / ('local' if ENVIRONMENT == 'local' else 'remote')