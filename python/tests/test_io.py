"""Tests for data I/O utilities."""

import json
import pytest
from pathlib import Path
import pandas as pd
import tempfile
import os

from ag_viz.io import (
    load_csv_data,
    load_model_json,
    load_forecast_csv,
    load_diagnostics_json,
    parse_simulation_csv,
)


class TestLoadCsvData:
    """Test CSV data loading functionality."""
    
    def test_load_valid_csv(self):
        """Test loading a valid CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("value\n")
            f.write("0.01\n")
            f.write("-0.02\n")
            f.write("0.03\n")
            temp_path = f.name
        
        try:
            df = load_csv_data(Path(temp_path))
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3
            assert 'value' in df.columns
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_csv(self):
        """Test loading a non-existent CSV file."""
        with pytest.raises(FileNotFoundError):
            load_csv_data(Path('nonexistent.csv'))
    
    def test_load_empty_csv(self):
        """Test loading an empty CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("value\n")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="empty"):
                load_csv_data(Path(temp_path))
        finally:
            os.unlink(temp_path)


class TestLoadModelJson:
    """Test model JSON loading functionality."""
    
    def test_load_valid_model(self):
        """Test loading a valid model JSON."""
        model_data = {
            "spec": {
                "arima": {"p": 1, "d": 0, "q": 1},
                "garch": {"p": 1, "q": 1}
            },
            "parameters": {
                "arima": {"intercept": 0.05, "ar_coef": [0.6], "ma_coef": [0.3]},
                "garch": {"omega": 0.01, "alpha_coef": [0.1], "beta_coef": [0.85]}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(model_data, f)
            temp_path = f.name
        
        try:
            model = load_model_json(Path(temp_path))
            assert isinstance(model, dict)
            assert 'spec' in model
            assert 'parameters' in model
            assert model['spec']['arima']['p'] == 1
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_model(self):
        """Test loading a non-existent model file."""
        with pytest.raises(FileNotFoundError):
            load_model_json(Path('nonexistent.json'))
    
    def test_load_invalid_model_structure(self):
        """Test loading a model with invalid structure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"invalid": "structure"}, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="missing"):
                load_model_json(Path(temp_path))
        finally:
            os.unlink(temp_path)


class TestLoadForecastCsv:
    """Test forecast CSV loading functionality."""
    
    def test_load_valid_forecast(self):
        """Test loading a valid forecast CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("step,mean,variance,std_dev\n")
            f.write("1,0.05,0.01,0.1\n")
            f.write("2,0.04,0.012,0.11\n")
            temp_path = f.name
        
        try:
            df = load_forecast_csv(Path(temp_path))
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert all(col in df.columns for col in ['step', 'mean', 'std_dev'])
        finally:
            os.unlink(temp_path)
    
    def test_load_forecast_missing_columns(self):
        """Test loading a forecast with missing columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("step,mean\n")
            f.write("1,0.05\n")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="missing required columns"):
                load_forecast_csv(Path(temp_path))
        finally:
            os.unlink(temp_path)


class TestLoadDiagnosticsJson:
    """Test diagnostics JSON loading functionality."""
    
    def test_load_valid_diagnostics(self):
        """Test loading valid diagnostics JSON."""
        diag_data = {
            "ljung_box_residuals": {"statistic": 10.5, "p_value": 0.15},
            "ljung_box_squared": {"statistic": 8.2, "p_value": 0.25}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(diag_data, f)
            temp_path = f.name
        
        try:
            diagnostics = load_diagnostics_json(Path(temp_path))
            assert isinstance(diagnostics, dict)
            assert 'ljung_box_residuals' in diagnostics
        finally:
            os.unlink(temp_path)


class TestParseSimulationCsv:
    """Test simulation CSV parsing functionality."""
    
    def test_parse_valid_simulation(self):
        """Test parsing a valid simulation CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("path,observation,return,volatility\n")
            f.write("0,0,0.01,0.05\n")
            f.write("0,1,0.02,0.06\n")
            f.write("1,0,-0.01,0.04\n")
            f.write("1,1,0.03,0.05\n")
            temp_path = f.name
        
        try:
            df, n_paths, n_obs = parse_simulation_csv(Path(temp_path))
            assert isinstance(df, pd.DataFrame)
            assert n_paths == 2
            assert n_obs == 2
            assert all(col in df.columns for col in ['path', 'observation', 'return', 'volatility'])
        finally:
            os.unlink(temp_path)
    
    def test_parse_simulation_missing_columns(self):
        """Test parsing simulation with missing columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("path,observation\n")
            f.write("0,0\n")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="missing required columns"):
                parse_simulation_csv(Path(temp_path))
        finally:
            os.unlink(temp_path)
