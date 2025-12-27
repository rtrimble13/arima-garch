"""Tests for plotting functionality."""

import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import json
import os
from unittest.mock import patch, MagicMock

from ag_viz.plotting import (
    plot_fit_diagnostics,
    plot_forecast,
    plot_residual_diagnostics,
    plot_simulation_paths,
)


class TestPlotFitDiagnostics:
    """Test fit diagnostics plotting."""
    
    def test_plot_fit_diagnostics_creates_file(self):
        """Test that plot_fit_diagnostics creates an output file."""
        # Create sample data
        data = pd.DataFrame({'value': np.random.randn(100)})
        model_json = {
            'spec': {
                'arima': {'p': 1, 'd': 0, 'q': 1},
                'garch': {'p': 1, 'q': 1}
            },
            'parameters': {}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Mock plt.show to prevent display
            with patch('matplotlib.pyplot.show'):
                plot_path = plot_fit_diagnostics(data, model_json, output_dir)
            
            assert plot_path.exists()
            assert plot_path.name == 'fit_diagnostics.png'


class TestPlotForecast:
    """Test forecast plotting."""
    
    def test_plot_forecast_creates_file(self):
        """Test that plot_forecast creates an output file."""
        # Create sample model and forecast data
        model_data = {
            'spec': {
                'arima': {'p': 1, 'd': 0, 'q': 1},
                'garch': {'p': 1, 'q': 1}
            },
            'parameters': {}
        }
        
        forecast_data = {
            'step': list(range(1, 11)),
            'mean': [0.05 + 0.01 * i for i in range(10)],
            'variance': [0.01] * 10,
            'std_dev': [0.1] * 10
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create temporary files
            model_path = Path(tmpdir) / 'model.json'
            forecast_path = Path(tmpdir) / 'forecast.csv'
            output_path = Path(tmpdir) / 'forecast_plot.png'
            
            with open(model_path, 'w') as f:
                json.dump(model_data, f)
            
            pd.DataFrame(forecast_data).to_csv(forecast_path, index=False)
            
            # Mock plt.show to prevent display
            with patch('matplotlib.pyplot.show'):
                plot_path = plot_forecast(
                    model_path,
                    forecast_path,
                    confidence_levels=[0.68, 0.95],
                    show=False,
                    save=output_path
                )
            
            assert plot_path.exists()
            assert plot_path == output_path


class TestPlotResidualDiagnostics:
    """Test residual diagnostics plotting."""
    
    def test_plot_residual_diagnostics_creates_file(self):
        """Test that plot_residual_diagnostics creates an output file."""
        # Create sample data
        model_data = {
            'spec': {
                'arima': {'p': 1, 'd': 0, 'q': 1},
                'garch': {'p': 1, 'q': 1}
            },
            'parameters': {}
        }
        
        data_df = pd.DataFrame({'value': np.random.randn(100)})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'model.json'
            data_path = Path(tmpdir) / 'data.csv'
            output_dir = Path(tmpdir) / 'diagnostics'
            
            with open(model_path, 'w') as f:
                json.dump(model_data, f)
            
            data_df.to_csv(data_path, index=False)
            
            # Mock plt.show to prevent display
            with patch('matplotlib.pyplot.show'):
                plot_path = plot_residual_diagnostics(
                    model_path,
                    data_path,
                    None,
                    output_dir
                )
            
            assert plot_path.exists()
            assert plot_path.name == 'residual_diagnostics.png'


class TestPlotSimulationPaths:
    """Test simulation paths plotting."""
    
    def test_plot_simulation_paths_creates_file(self):
        """Test that plot_simulation_paths creates an output file."""
        # Create sample simulation data
        paths = []
        for path_id in range(5):
            for obs in range(100):
                paths.append({
                    'path': path_id,
                    'observation': obs,
                    'return': np.random.randn() * 0.01,
                    'volatility': 0.05 + np.random.rand() * 0.02
                })
        
        sim_df = pd.DataFrame(paths)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            sim_path = Path(tmpdir) / 'simulation.csv'
            output_path = Path(tmpdir) / 'sim_plot.png'
            
            sim_df.to_csv(sim_path, index=False)
            
            # Mock plt.show to prevent display
            with patch('matplotlib.pyplot.show'):
                plot_path = plot_simulation_paths(
                    sim_path,
                    n_paths_to_plot=3,
                    output_path=output_path,
                    show=False
                )
            
            assert plot_path.exists()
            assert plot_path == output_path


class TestPlottingEdgeCases:
    """Test edge cases in plotting functions."""
    
    def test_plot_with_small_dataset(self):
        """Test plotting with a small dataset."""
        data = pd.DataFrame({'value': [0.01, -0.02, 0.03]})
        model_json = {
            'spec': {
                'arima': {'p': 1, 'd': 0, 'q': 1},
                'garch': {'p': 1, 'q': 1}
            },
            'parameters': {}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            with patch('matplotlib.pyplot.show'):
                plot_path = plot_fit_diagnostics(data, model_json, output_dir)
            
            assert plot_path.exists()
