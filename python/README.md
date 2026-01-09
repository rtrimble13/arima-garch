# ag-viz: Visualization Tools for ARIMA-GARCH Models

`ag-viz` is a Python package that provides publication-quality visualization capabilities for ARIMA-GARCH model outputs. It wraps the C++ `ag` CLI tool and generates professional plots for model diagnostics, forecasts, residual analysis, and simulation paths.

## Features

- **Model Fit Diagnostics**: Visualize time series data with model specifications and summary statistics
- **Forecast Plots**: Generate forecasts with multiple confidence interval bands
- **Residual Diagnostics**: Comprehensive multi-panel diagnostic plots including:
  - Standardized residuals time series
  - Histogram with normal distribution overlay
  - QQ-plots for normality assessment
  - ACF plots of residuals and squared residuals
- **Simulation Visualizations**: Plot multiple simulation paths with statistical summaries
- **Markdown Reports**: Professional, publication-ready analysis reports for all operations:
  - Thorough methodology explanations
  - Statistical interpretations and insights
  - Embedded visualizations
  - Practical recommendations and next steps
  - Academic references

## Installation

### Prerequisites

1. **Build the C++ `ag` CLI tool first**:
   ```bash
   # From the repository root
   # Option 1: Using Make + Ninja (Recommended)
   make
   
   # Option 2: Using CMake directly
   cmake -S . -B build
   cmake --build build
   ```

2. **Install Python package**:
   ```bash
   # Install in development mode (recommended)
   pip install -e python/
   
   # Or install from source
   cd python/
   pip install .
   ```

### Optional: Set Environment Variable

If the `ag` executable is not in your PATH, set the environment variable:

```bash
# If you used Make + Ninja
export AG_EXECUTABLE=/path/to/arima-garch/build/ninja-release/src/ag

# If you used direct CMake
export AG_EXECUTABLE=/path/to/arima-garch/build/src/ag
```

## Quick Start

### Basic Usage

```bash
# Fit a model and generate diagnostic plots
ag-viz fit -d examples/returns.csv -a 1,0,1 -g 1,1 -o model.json

# Generate forecast with visualization
ag-viz forecast -m model.json -n 30 -o forecast.csv

# Run comprehensive diagnostics
ag-viz diagnostics -m model.json -d examples/returns.csv -o ./diagnostics/

# Simulate and visualize paths
ag-viz simulate -m model.json -p 100 -n 1000 -o simulation.csv
```

### Using Markdown Reports

All subcommands support the `--markdown` flag to generate professional, publication-ready analysis reports:

```bash
# Fit model with comprehensive Markdown report
ag-viz fit -d examples/returns.csv -a 1,0,1 -g 1,1 -o model.json --markdown

# Forecast with detailed Markdown analysis
ag-viz forecast -m model.json -n 30 -o forecast.csv --markdown

# Diagnostics with interpretation report
ag-viz diagnostics -m model.json -d examples/returns.csv -o ./diagnostics/ --markdown

# Simulation with statistical summary report
ag-viz simulate -m model.json -p 100 -n 1000 -o simulation.csv --markdown
```

Markdown reports are saved to `./reports/` directory and include:
- Comprehensive methodology explanations
- Statistical summaries and interpretations
- Embedded visualizations
- Key insights and practical guidance
- References and next steps

### Programmatic Usage

```python
from pathlib import Path
from ag_viz import (
    load_csv_data,
    load_model_json,
    plot_fit_diagnostics,
    plot_forecast,
    plot_residual_diagnostics,
    plot_simulation_paths
)

# Load data and model
data = load_csv_data(Path('data.csv'))
model = load_model_json(Path('model.json'))

# Generate plots
plot_fit_diagnostics(data, model, Path('./output'))
plot_forecast(Path('model.json'), Path('forecast.csv'))
plot_residual_diagnostics(Path('model.json'), Path('data.csv'))
plot_simulation_paths(Path('simulation.csv'), n_paths_to_plot=20)
```

## Command Reference

### `ag-viz fit`

Fit an ARIMA-GARCH model and generate diagnostic plots.

```bash
ag-viz fit -d <data.csv> -a <p,d,q> -g <P,Q> -o <model.json> [--plot-dir <dir>] [--markdown]
```

**Options:**
- `-d, --data`: Input CSV file with time series data (required)
- `-a, --arima`: ARIMA order as `p,d,q` (required)
- `-g, --garch`: GARCH order as `P,Q` (required)
- `-o, --output`: Output model JSON file
- `--plot-dir`: Directory for diagnostic plots (default: `./output`)
- `--markdown`: Generate a professional Markdown report with analysis and visuals

**Example:**
```bash
ag-viz fit -d returns.csv -a 1,0,1 -g 1,1 -o model.json --plot-dir ./plots

# With Markdown report
ag-viz fit -d returns.csv -a 1,0,1 -g 1,1 -o model.json --markdown
```

The `--markdown` flag generates a comprehensive report at `./reports/fit_report.md` that includes:
- Model specification and methodology overview
- Data summary statistics with interpretations
- Detailed parameter estimates
- Key insights and caveats
- Next steps and recommendations

### `ag-viz forecast`

Generate forecasts and visualize with confidence intervals.

```bash
ag-viz forecast -m <model.json> -n <horizon> -o <forecast.csv> [--plot <path>] [--show] [--markdown]
```

**Options:**
- `-m, --model`: Input model JSON file (required)
- `-n, --horizon`: Forecast horizon in steps (default: 10)
- `-o, --output`: Output forecast CSV file
- `--plot`: Path to save forecast plot
- `--show`: Display the plot interactively
- `--markdown`: Generate a professional Markdown report with analysis and visuals

**Example:**
```bash
ag-viz forecast -m model.json -n 30 -o forecast.csv --plot forecast.png

# With Markdown report
ag-viz forecast -m model.json -n 30 -o forecast.csv --markdown
```

The `--markdown` flag generates a detailed report at `./reports/forecast_report.md` that includes:
- Forecast trajectory visualization
- Complete forecast table with confidence intervals
- Trend analysis and uncertainty assessment
- Key insights and practical guidance

### `ag-viz diagnostics`

Run diagnostic tests and generate comprehensive residual plots.

```bash
ag-viz diagnostics -m <model.json> -d <data.csv> -o <output_dir> [--markdown]
```

**Options:**
- `-m, --model`: Input model JSON file (required)
- `-d, --data`: Input CSV data file (required)
- `-o, --output`: Output directory for plots and JSON (default: `./diagnostics`)
- `--markdown`: Generate a professional Markdown report with analysis and visuals

**Example:**
```bash
ag-viz diagnostics -m model.json -d returns.csv -o ./diagnostics

# With Markdown report
ag-viz diagnostics -m model.json -d returns.csv -o ./diagnostics --markdown
```

The `--markdown` flag generates a comprehensive report at `./reports/diagnostics_report.md` that includes:
- Detailed test results (Ljung-Box, Jarque-Bera)
- Interpretation of diagnostic plots
- Model adequacy assessment
- Recommendations for model refinement

### `ag-viz simulate`

Simulate multiple paths and visualize distributions.

```bash
ag-viz simulate -m <model.json> -p <paths> -n <length> -o <output.csv> [options] [--markdown]
```

**Options:**
- `-m, --model`: Input model JSON file (required)
- `-p, --paths`: Number of simulation paths (default: 100)
- `-n, --length`: Observations per path (default: 1000)
- `-s, --seed`: Random seed for reproducibility (default: 42)
- `-o, --output`: Output simulation CSV file
- `--plot`: Path to save simulation plot
- `--n-plot`: Number of paths to display in plot (default: 10)
- `--show`: Display the plot interactively
- `--stats`: Compute and display summary statistics
- `--markdown`: Generate a professional Markdown report with analysis and visuals

**Example:**
```bash
ag-viz simulate -m model.json -p 100 -n 1000 -o sim.csv --n-plot 20 --stats

# With Markdown report
ag-viz simulate -m model.json -p 100 -n 1000 -o sim.csv --markdown
```

The `--markdown` flag generates a detailed report at `./reports/simulation_report.md` that includes:
- Comprehensive simulation statistics
- Distribution analysis and tail behavior
- Terminal value statistics
- Risk management applications and insights

## Plot Examples

### Fit Diagnostics
Shows the time series data with summary statistics and model specification.

### Forecast Visualization
Displays mean forecast with 68% and 95% confidence intervals.

### Residual Diagnostics
Multi-panel plot including:
- Standardized residuals over time
- Histogram with normal overlay
- QQ-plot
- ACF of residuals
- ACF of squared residuals

### Simulation Paths
Shows multiple simulated paths with mean path, percentile bands, and terminal value distribution.

## Dependencies

**Required:**
- click >= 8.0
- matplotlib >= 3.5
- seaborn >= 0.12
- pandas >= 1.5
- numpy >= 1.23
- scipy >= 1.9

**Optional (for development):**
- pytest >= 7.0
- pytest-cov >= 4.0
- black >= 23.0
- jupyter >= 1.0

## Development

Install development dependencies:

```bash
pip install -e python/[dev]
```

Run tests:

```bash
cd python/
pytest tests/
```

Format code:

```bash
black ag_viz/ tests/
```

## Integration with C++ CLI

The `ag-viz` package wraps the C++ `ag` CLI tool. Each visualization command:

1. Calls the corresponding `ag` command via subprocess
2. Loads the output files (JSON/CSV)
3. Generates and saves visualizations
4. Displays success messages

The underlying C++ CLI must be built before using `ag-viz`. See the main repository [README](../README.md) for build instructions.

## Troubleshooting

### ag executable not found

If you see a warning about the `ag` executable not being found:

1. Ensure the C++ project is built: 
   - `make` (if using Make + Ninja)
   - `cmake --build build` (if using CMake directly)
2. Set the environment variable:
   - `export AG_EXECUTABLE=/path/to/build/ninja-release/src/ag` (for Make)
   - `export AG_EXECUTABLE=/path/to/build/src/ag` (for CMake)
3. Or add the build directory to your PATH:
   - `export PATH=$PATH:/path/to/build/ninja-release/src` (for Make)
   - `export PATH=$PATH:/path/to/build/src` (for CMake)

### Import errors

Ensure all dependencies are installed:

```bash
pip install -r python/requirements.txt
```

### Plot display issues

By default, plots are saved to files without displaying. Use `--show` flag to display plots interactively (requires a display/X server).

## Contributing

Contributions are welcome! Please see the main repository [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

When contributing to `ag-viz`:
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings for all functions
- Update tests for new features
- Run `black` for code formatting

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Links

- [Main Project Repository](https://github.com/rtrimble13/arima-garch)
- [C++ CLI Documentation](../docs/cli.md)
- [ARIMA-GARCH Model Documentation](../README.md)

## Citation

If you use this package in your research, please cite:

```
ag-viz: Visualization Tools for ARIMA-GARCH Models
rtrimble13 (2024)
https://github.com/rtrimble13/arima-garch
```
