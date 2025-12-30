"""
ag-viz: Visualization tools for ARIMA-GARCH models.

This package provides publication-quality plotting and visualization
capabilities for ARIMA-GARCH model outputs from the ag CLI tool.
"""

__version__ = "1.1.1"
__author__ = "rtrimble13"

from ag_viz.plotting import (
    plot_fit_diagnostics,
    plot_forecast,
    plot_residual_diagnostics,
    plot_simulation_paths,
)
from ag_viz.io import (
    load_csv_data,
    load_model_json,
    load_forecast_csv,
    load_diagnostics_json,
    parse_simulation_csv,
)
from ag_viz.utils import (
    find_ag_executable,
    run_ag_command,
    ensure_output_dir,
    format_model_spec,
)

__all__ = [
    "plot_fit_diagnostics",
    "plot_forecast",
    "plot_residual_diagnostics",
    "plot_simulation_paths",
    "load_csv_data",
    "load_model_json",
    "load_forecast_csv",
    "load_diagnostics_json",
    "parse_simulation_csv",
    "find_ag_executable",
    "run_ag_command",
    "ensure_output_dir",
    "format_model_spec",
]
