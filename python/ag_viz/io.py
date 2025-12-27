"""
Data I/O utilities for ag-viz package.

Provides functions to load and parse various file formats:
- CSV time series data
- JSON model files
- Forecast output files
- Diagnostics JSON files
- Simulation CSV files
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


def load_csv_data(filepath: Path) -> pd.DataFrame:
    """
    Load time series CSV data.
    
    Parameters
    ----------
    filepath : Path
        Path to the CSV file. First column is used as the time series data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with the loaded time series data.
    
    Raises
    ------
    FileNotFoundError
        If the CSV file doesn't exist.
    ValueError
        If the CSV file is empty or invalid.
    
    Examples
    --------
    >>> data = load_csv_data(Path('returns.csv'))
    >>> print(data.head())
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            raise ValueError(f"CSV file is empty: {filepath}")
        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV file {filepath}: {e}") from e


def load_model_json(filepath: Path) -> Dict[str, Any]:
    """
    Load model JSON file.
    
    Parameters
    ----------
    filepath : Path
        Path to the JSON model file.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing model specification and parameters.
    
    Raises
    ------
    FileNotFoundError
        If the JSON file doesn't exist.
    ValueError
        If the JSON file is invalid.
    
    Examples
    --------
    >>> model = load_model_json(Path('model.json'))
    >>> print(model['spec'])
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            model = json.load(f)
        
        # Validate basic structure
        if 'spec' not in model:
            raise ValueError("Model JSON missing 'spec' field")
        if 'parameters' not in model:
            raise ValueError("Model JSON missing 'parameters' field")
        
        return model
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in model file {filepath}: {e}") from e
    except Exception as e:
        raise ValueError(f"Error loading model file {filepath}: {e}") from e


def load_forecast_csv(filepath: Path) -> pd.DataFrame:
    """
    Load forecast CSV output.
    
    Expected columns: step, mean, variance, std_dev
    
    Parameters
    ----------
    filepath : Path
        Path to the forecast CSV file.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with forecast data.
    
    Raises
    ------
    FileNotFoundError
        If the CSV file doesn't exist.
    ValueError
        If the CSV file is invalid or missing required columns.
    
    Examples
    --------
    >>> forecast = load_forecast_csv(Path('forecast.csv'))
    >>> print(forecast[['step', 'mean', 'std_dev']])
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Forecast file not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        
        # Check for required columns
        required_cols = ['step', 'mean', 'std_dev']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Forecast CSV missing required columns: {missing_cols}")
        
        return df
    except Exception as e:
        raise ValueError(f"Error loading forecast file {filepath}: {e}") from e


def load_diagnostics_json(filepath: Path) -> Dict[str, Any]:
    """
    Load diagnostics JSON output.
    
    Parameters
    ----------
    filepath : Path
        Path to the diagnostics JSON file.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing diagnostic test results.
    
    Raises
    ------
    FileNotFoundError
        If the JSON file doesn't exist.
    ValueError
        If the JSON file is invalid.
    
    Examples
    --------
    >>> diagnostics = load_diagnostics_json(Path('diagnostics.json'))
    >>> print(diagnostics['ljung_box_residuals'])
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Diagnostics file not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            diagnostics = json.load(f)
        return diagnostics
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in diagnostics file {filepath}: {e}") from e
    except Exception as e:
        raise ValueError(f"Error loading diagnostics file {filepath}: {e}") from e


def parse_simulation_csv(filepath: Path) -> Tuple[pd.DataFrame, int, int]:
    """
    Parse multi-path simulation CSV output.
    
    Expected columns: path, observation, return, volatility
    
    Parameters
    ----------
    filepath : Path
        Path to the simulation CSV file.
    
    Returns
    -------
    Tuple[pd.DataFrame, int, int]
        A tuple containing:
        - DataFrame with simulation data
        - Number of paths
        - Number of observations per path
    
    Raises
    ------
    FileNotFoundError
        If the CSV file doesn't exist.
    ValueError
        If the CSV file is invalid or missing required columns.
    
    Examples
    --------
    >>> sim_data, n_paths, n_obs = parse_simulation_csv(Path('simulation.csv'))
    >>> print(f"Loaded {n_paths} paths with {n_obs} observations each")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Simulation file not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        
        # Check for required columns
        required_cols = ['path', 'observation', 'return', 'volatility']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Simulation CSV missing required columns: {missing_cols}")
        
        # Calculate dimensions
        n_paths = df['path'].nunique()
        n_obs_per_path = df.groupby('path').size().iloc[0]
        
        return df, n_paths, n_obs_per_path
    except Exception as e:
        raise ValueError(f"Error loading simulation file {filepath}: {e}") from e
