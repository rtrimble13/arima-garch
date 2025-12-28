"""Tests for markdown report generation functionality."""

import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import json
from unittest.mock import patch

from ag_viz.markdown_reports import (
    generate_fit_report,
    generate_forecast_report,
    generate_diagnostics_report,
    generate_simulation_report,
)


class TestGenerateFitReport:
    """Test fit report generation."""
    
    def test_generate_fit_report_creates_file(self):
        """Test that generate_fit_report creates a markdown file."""
        # Create sample data
        data = pd.DataFrame({'value': np.random.randn(100)})
        model_json = {
            'spec': {
                'arima': {'p': 1, 'd': 0, 'q': 1},
                'garch': {'p': 1, 'q': 1}
            },
            'parameters': {
                'arima': {
                    'intercept': 0.05,
                    'ar_coef': [0.6],
                    'ma_coef': [0.3]
                },
                'garch': {
                    'omega': 0.01,
                    'alpha_coef': [0.1],
                    'beta_coef': [0.85]
                }
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / 'fit_diagnostics.png'
            output_path = Path(tmpdir) / 'fit_report.md'
            
            # Create a dummy plot file
            plot_path.touch()
            
            report_file = generate_fit_report(
                data=data,
                model_json=model_json,
                plot_path=plot_path,
                output_path=output_path,
                use_data_uri=False
            )
            
            assert report_file.exists()
            assert report_file == output_path
            
            # Check that the file contains expected content
            with open(report_file, 'r') as f:
                content = f.read()
            
            assert '# ARIMA-GARCH Model Fit Report' in content
            assert 'ARIMA(1,0,1)-GARCH(1,1)' in content
            assert 'Overview' in content
            assert 'Methodology' in content
            assert 'Data Summary Statistics' in content
            assert 'Model Parameters' in content
            assert 'Next Steps' in content
    
    def test_generate_fit_report_with_minimal_params(self):
        """Test report generation with minimal parameters."""
        data = pd.DataFrame({'value': [0.01, -0.02, 0.03, 0.05]})
        model_json = {
            'spec': {
                'arima': {'p': 0, 'd': 0, 'q': 0},
                'garch': {'p': 1, 'q': 1}
            },
            'parameters': {}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / 'plot.png'
            output_path = Path(tmpdir) / 'report.md'
            plot_path.touch()
            
            report_file = generate_fit_report(
                data=data,
                model_json=model_json,
                plot_path=plot_path,
                output_path=output_path
            )
            
            assert report_file.exists()


class TestGenerateForecastReport:
    """Test forecast report generation."""
    
    def test_generate_forecast_report_creates_file(self):
        """Test that generate_forecast_report creates a markdown file."""
        model_json = {
            'spec': {
                'arima': {'p': 1, 'd': 0, 'q': 1},
                'garch': {'p': 1, 'q': 1}
            },
            'parameters': {}
        }
        
        forecast_df = pd.DataFrame({
            'step': list(range(1, 11)),
            'mean': [0.05 + 0.01 * i for i in range(10)],
            'variance': [0.01] * 10,
            'std_dev': [0.1] * 10
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / 'forecast.png'
            output_path = Path(tmpdir) / 'forecast_report.md'
            plot_path.touch()
            
            report_file = generate_forecast_report(
                model_json=model_json,
                forecast_df=forecast_df,
                plot_path=plot_path,
                output_path=output_path
            )
            
            assert report_file.exists()
            assert report_file == output_path
            
            # Check content
            with open(report_file, 'r') as f:
                content = f.read()
            
            assert '# ARIMA-GARCH Forecast Report' in content
            assert 'Forecast Horizon' in content
            assert 'Confidence Intervals' in content
            assert 'Detailed Forecast Table' in content
    
    def test_generate_forecast_report_with_various_horizons(self):
        """Test forecast report with different horizons."""
        model_json = {
            'spec': {
                'arima': {'p': 2, 'd': 0, 'q': 2},
                'garch': {'p': 1, 'q': 1}
            },
            'parameters': {}
        }
        
        # Test with short horizon
        forecast_df = pd.DataFrame({
            'step': [1, 2, 3],
            'mean': [0.01, 0.02, 0.03],
            'variance': [0.001, 0.002, 0.003],
            'std_dev': [0.03, 0.04, 0.05]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / 'forecast.png'
            output_path = Path(tmpdir) / 'forecast_report.md'
            plot_path.touch()
            
            report_file = generate_forecast_report(
                model_json=model_json,
                forecast_df=forecast_df,
                plot_path=plot_path,
                output_path=output_path
            )
            
            assert report_file.exists()


class TestGenerateDiagnosticsReport:
    """Test diagnostics report generation."""
    
    def test_generate_diagnostics_report_creates_file(self):
        """Test that generate_diagnostics_report creates a markdown file."""
        model_json = {
            'spec': {
                'arima': {'p': 1, 'd': 0, 'q': 1},
                'garch': {'p': 1, 'q': 1}
            },
            'parameters': {}
        }
        
        data = pd.DataFrame({'value': np.random.randn(100)})
        
        diagnostics_json = {
            'ljung_box_test': {
                'lags': [5, 10, 15, 20],
                'statistics': [3.2, 8.5, 12.1, 15.3],
                'pvalues': [0.67, 0.58, 0.67, 0.76]
            },
            'jarque_bera_test': {
                'statistic': 2.5,
                'pvalue': 0.28
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / 'diagnostics.png'
            output_path = Path(tmpdir) / 'diagnostics_report.md'
            plot_path.touch()
            
            report_file = generate_diagnostics_report(
                model_json=model_json,
                data=data,
                diagnostics_json=diagnostics_json,
                plot_path=plot_path,
                output_path=output_path
            )
            
            assert report_file.exists()
            assert report_file == output_path
            
            # Check content
            with open(report_file, 'r') as f:
                content = f.read()
            
            assert '# ARIMA-GARCH Diagnostic Analysis Report' in content
            assert 'Ljung-Box Test' in content
            assert 'Jarque-Bera' in content
            assert 'Residual Analysis Plots' in content
    
    def test_generate_diagnostics_report_without_tests(self):
        """Test diagnostics report without test results."""
        model_json = {
            'spec': {
                'arima': {'p': 1, 'd': 0, 'q': 1},
                'garch': {'p': 1, 'q': 1}
            },
            'parameters': {}
        }
        
        data = pd.DataFrame({'value': np.random.randn(50)})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / 'diagnostics.png'
            output_path = Path(tmpdir) / 'diagnostics_report.md'
            plot_path.touch()
            
            report_file = generate_diagnostics_report(
                model_json=model_json,
                data=data,
                diagnostics_json=None,
                plot_path=plot_path,
                output_path=output_path
            )
            
            assert report_file.exists()


class TestGenerateSimulationReport:
    """Test simulation report generation."""
    
    def test_generate_simulation_report_creates_file(self):
        """Test that generate_simulation_report creates a markdown file."""
        model_json = {
            'spec': {
                'arima': {'p': 1, 'd': 0, 'q': 1},
                'garch': {'p': 1, 'q': 1}
            },
            'parameters': {}
        }
        
        # Create simulation data in expected format
        paths = []
        for path_id in range(10):
            for obs in range(100):
                paths.append({
                    'path': path_id,
                    'observation': obs,
                    'return': np.random.randn() * 0.01,
                    'volatility': 0.05 + np.random.rand() * 0.02
                })
        
        simulation_df = pd.DataFrame(paths)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / 'simulation.png'
            output_path = Path(tmpdir) / 'simulation_report.md'
            plot_path.touch()
            
            report_file = generate_simulation_report(
                model_json=model_json,
                simulation_df=simulation_df,
                plot_path=plot_path,
                output_path=output_path,
                n_paths=10,
                length=100
            )
            
            assert report_file.exists()
            assert report_file == output_path
            
            # Check content
            with open(report_file, 'r') as f:
                content = f.read()
            
            assert '# ARIMA-GARCH Simulation Report' in content
            assert 'Monte Carlo Simulation' in content
            assert 'Aggregate Statistics' in content
            assert 'Terminal Value Statistics' in content
    
    def test_generate_simulation_report_with_few_paths(self):
        """Test simulation report with few paths."""
        model_json = {
            'spec': {
                'arima': {'p': 1, 'd': 0, 'q': 1},
                'garch': {'p': 1, 'q': 1}
            },
            'parameters': {}
        }
        
        # Create minimal simulation data
        paths = []
        for path_id in range(2):
            for obs in range(10):
                paths.append({
                    'path': path_id,
                    'observation': obs,
                    'return': 0.01 * obs,
                    'volatility': 0.05
                })
        
        simulation_df = pd.DataFrame(paths)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / 'simulation.png'
            output_path = Path(tmpdir) / 'simulation_report.md'
            plot_path.touch()
            
            report_file = generate_simulation_report(
                model_json=model_json,
                simulation_df=simulation_df,
                plot_path=plot_path,
                output_path=output_path,
                n_paths=2,
                length=10
            )
            
            assert report_file.exists()


class TestMarkdownReportEdgeCases:
    """Test edge cases in markdown report generation."""
    
    def test_reports_create_output_directories(self):
        """Test that reports create output directories if they don't exist."""
        model_json = {
            'spec': {
                'arima': {'p': 1, 'd': 0, 'q': 1},
                'garch': {'p': 1, 'q': 1}
            },
            'parameters': {}
        }
        
        data = pd.DataFrame({'value': np.random.randn(50)})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / 'plot.png'
            # Use a nested path that doesn't exist
            output_path = Path(tmpdir) / 'reports' / 'nested' / 'report.md'
            plot_path.touch()
            
            report_file = generate_fit_report(
                data=data,
                model_json=model_json,
                plot_path=plot_path,
                output_path=output_path
            )
            
            assert report_file.exists()
            assert report_file.parent.exists()
    
    def test_report_with_data_uri(self):
        """Test report generation with embedded data URIs."""
        model_json = {
            'spec': {
                'arima': {'p': 1, 'd': 0, 'q': 1},
                'garch': {'p': 1, 'q': 1}
            },
            'parameters': {}
        }
        
        data = pd.DataFrame({'value': [0.01, 0.02, 0.03]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / 'plot.png'
            output_path = Path(tmpdir) / 'report.md'
            
            # Create a small dummy PNG file
            with open(plot_path, 'wb') as f:
                # Minimal PNG header
                f.write(b'\x89PNG\r\n\x1a\n')
            
            report_file = generate_fit_report(
                data=data,
                model_json=model_json,
                plot_path=plot_path,
                output_path=output_path,
                use_data_uri=True
            )
            
            assert report_file.exists()
            
            # Check that data URI is embedded
            with open(report_file, 'r') as f:
                content = f.read()
            
            assert 'data:image/png;base64,' in content
