import pandas as pd
import yaml
import pickle
import os
from pathlib import Path
from typing import Any, Dict

def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        Dict[str, Any]: Contents of the YAML file.
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data: Dict[str, Any], file_path: str) -> None:
    """
    Save a dictionary to a YAML file.

    Args:
        data (Dict[str, Any]): Data to save.
        file_path (str): Path to the YAML file.
    """
    with open(file_path, 'w') as f:
        yaml.safe_dump(data, f)

def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Args:
        file_path (str): Path to the CSV file.
        **kwargs: Additional arguments for pandas.read_csv.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(file_path, **kwargs)

def save_csv(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """
    Save a DataFrame to a CSV file, ensuring the directory exists.

    Args:
        df (pd.DataFrame): DataFrame to save.
        file_path (str): Path to the CSV file.
        **kwargs: Additional arguments for pandas.DataFrame.to_csv.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, **kwargs)

def load_pickle(file_path: str) -> Any:
    """
    Load a Python object from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        Any: Loaded Python object.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj: Any, file_path: str) -> None:
    """
    Save a Python object to a pickle file, ensuring the directory exists.

    Args:
        obj (Any): Python object to save.
        file_path (str): Path to the pickle file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def file_exists(file_path: str) -> bool:
    """
    Check if a file exists.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if file exists, False otherwise.
    """
    return Path(file_path).exists()
