# CLI Documentation

The `ag` command-line tool provides a comprehensive interface for ARIMA-GARCH time series modeling. This document describes the available subcommands and their usage.

## Installation

After building the project, the `ag` executable is located at `build/src/ag`.

```bash
# Add to PATH (optional)
export PATH="/path/to/arima-garch/build/src:$PATH"
```

## General Usage

```bash
ag [subcommand] [options]
```

To see available subcommands:

```bash
ag --help
```

To see version information:

```bash
ag --version
```

## Subcommands

### 1. `fit` - Fit ARIMA-GARCH Model

Fit an ARIMA-GARCH model to time series data and generate a comprehensive fit summary report.

**Usage:**
```bash
ag fit -d <data_file> -a <arima_order> -g <garch_order> [-o <output_file>]
```

**Options:**
- `-d, --data` (required): Input data file in CSV format (uses first column)
- `-a, --arima` (required): ARIMA order as `p,d,q` (e.g., `1,1,1`)
- `-g, --garch` (required): GARCH order as `p,q` (e.g., `1,1`)
- `-o, --output` (optional): Output model file in JSON format
- `--t-dist FLOAT` (optional): Use Student-t distribution with specified degrees of freedom (e.g., `--t-dist 5.0`). If not specified, Gaussian innovations are used (default)
- `--no-header` (optional): CSV file has no header row (default: expect header)

**Example:**
```bash
# Fit ARIMA(1,1,1)-GARCH(1,1) model to data with Gaussian innovations (default)
ag fit -d timeseries.csv -a 1,1,1 -g 1,1 -o fitted_model.json

# Fit with Student-t innovations (df=5)
ag fit -d returns.csv -a 2,0,1 -g 1,1 --t-dist 5.0 -o model_student_t.json

# Fit without saving the model
ag fit -d returns.csv -a 2,0,1 -g 1,1
```

**Output:**
- Model convergence information
- Innovation distribution used (Normal or Student-t with df)
- AIC, BIC, and log-likelihood
- Parameter estimates
- Distribution comparison (when fitted with Gaussian, shows if Student-t would be better)
- Diagnostic test results (Ljung-Box, Jarque-Bera)
- Full fit summary report

### 2. `select` - Automatic Model Selection

Automatically select the best ARIMA-GARCH model from a grid of candidate specifications using information criteria or cross-validation. The selection process evaluates models using Gaussian innovations and includes distribution comparison to recommend Student-t if it provides a better fit.

**Usage:**
```bash
ag select -d <data_file> [--max-p <p>] [--max-d <d>] [--max-q <q>] \
          [--max-garch-p <P>] [--max-garch-q <Q>] [-c <criterion>] [-o <output_file>]
```

**Options:**
- `-d, --data` (required): Input data file in CSV format
- `--max-p` (default: 2): Maximum ARIMA AR order
- `--max-d` (default: 1): Maximum ARIMA differencing order
- `--max-q` (default: 2): Maximum ARIMA MA order
- `--max-garch-p` (default: 1): Maximum GARCH p order
- `--max-garch-q` (default: 1): Maximum GARCH q order
- `-c, --criterion` (default: BIC): Selection criterion (`BIC`, `AIC`, `AICc`, or `CV`)
- `-o, --output` (optional): Output model file in JSON format
- `--top-k` (optional): Display top K models in ranking table
- `--no-header` (optional): CSV file has no header row (default: expect header)

**Example:**
```bash
# Select best model using BIC (default)
ag select -d timeseries.csv -o best_model.json

# Search larger model space with AIC criterion
ag select -d returns.csv --max-p 3 --max-q 3 --max-garch-p 2 --max-garch-q 2 -c AIC

# Use cross-validation for selection (slower but better forecast accuracy)
ag select -d data.csv -c CV -o cv_selected_model.json
```

**Output:**
- Number of candidates evaluated
- Number of candidates that failed to fit
- Best model specification
- Model fit statistics and diagnostics
- **Distribution comparison**: Recommendation on whether Student-t innovations would provide better fit
- Full fit summary report for the selected model

**Note:** If the distribution comparison suggests Student-t would be better, consider refitting the selected model with `--t-dist` option using the `fit` command.

**For detailed information about selection criteria, candidate generation, and best practices, see [Model Selection Documentation](model_selection.md).**

### 3. `forecast` - Generate Forecasts

Generate h-step ahead forecasts from a fitted model.

**Usage:**
```bash
ag forecast -m <model_file> [-n <horizon>] [-o <output_file>]
```

**Options:**
- `-m, --model` (required): Input model file in JSON format
- `-n, --horizon` (default: 10): Forecast horizon (number of steps ahead)
- `-o, --output` (optional): Output forecast file in CSV format

**Example:**
```bash
# Generate 10-step ahead forecasts
ag forecast -m fitted_model.json

# Generate 30-step forecasts and save to file
ag forecast -m best_model.json -n 30 -o forecasts.csv
```

**Output:**
- Table of mean forecasts and standard deviations for each step
- Optional CSV file with columns: `step`, `mean`, `variance`, `std_dev`

### 4. `sim` - Simulate Synthetic Data

Simulate synthetic time series data from an ARIMA-GARCH model with default parameters.

**Usage:**
```bash
ag sim -a <arima_order> -g <garch_order> -o <output_file> [-n <length>] [-s <seed>] [--t-dist <df>]
```

**Options:**
- `-a, --arima` (required): ARIMA order as `p,d,q` (e.g., `1,1,1`)
- `-g, --garch` (required): GARCH order as `p,q` (e.g., `1,1`)
- `-o, --output` (required): Output data file in CSV format
- `-n, --length` (default: 1000): Number of observations to simulate
- `-s, --seed` (default: 42): Random seed for reproducibility
- `--t-dist FLOAT` (optional): Use Student-t distribution with specified degrees of freedom (e.g., `--t-dist 3.0`). If not specified, Gaussian innovations are used (default)

**Example:**
```bash
# Simulate 1000 observations from ARIMA(1,1,1)-GARCH(1,1) with Gaussian innovations
ag sim -a 1,1,1 -g 1,1 -o synthetic_data.csv

# Simulate with Student-t innovations (df=4)
ag sim -a 2,0,1 -g 1,1 -n 5000 --t-dist 4.0 -o heavy_tail_data.csv

# Simulate 5000 observations with custom seed
ag sim -a 2,0,1 -g 1,1 -n 5000 -s 12345 -o large_sample.csv
```

**Output:**
- CSV file with columns: `observation`, `return`, `volatility`
- Console message confirming successful simulation and distribution used

**Note:** The simulation uses default parameter values. For custom parameters, use the library API directly or modify and re-save a model JSON file.

### 5. `simulate` - Simulate from Saved Model

Generate multiple simulation paths from a saved model file.

**Usage:**
```bash
ag simulate -m <model_file> [-p <num_paths>] [-n <length>] [-s <seed>] -o <output_file> [--stats]
```

**Options:**
- `-m, --model` (required): Input model file in JSON format
- `-p, --paths` (default: 1): Number of simulation paths to generate
- `-n, --length` (default: 1000): Number of observations per path
- `-s, --seed` (default: 42): Random seed for reproducibility
- `-o, --output` (required): Output CSV file (e.g., `sim_returns.csv`)
- `--stats`: Compute and display summary statistics

**Example:**
```bash
# Simulate 3 paths of 1000 observations each from a fitted model
ag simulate -m fitted_model.json -p 3 -n 1000 -s 42 -o sim_returns.csv

# Simulate single path with statistics
ag simulate -m best_model.json -p 1 -n 500 -s 123 -o sim_path.csv --stats

# Generate many paths for Monte Carlo analysis
ag simulate -m model.json -p 100 -n 252 -s 42 -o monte_carlo.csv
```

**Output:**
- CSV file with columns: `path`, `observation`, `return`, `volatility`
- Optional summary statistics (mean, std dev, min, max, skewness, kurtosis)
- First path values displayed for reproducibility verification

**Key Features:**
- **Reproducibility**: Same seed produces identical first path values
- **Multiple paths**: Generate N independent simulation paths with different realizations
- **Loaded parameters**: Uses the exact parameters from your fitted model
- **Statistical summary**: Optional aggregate statistics across all paths

### 6. `diagnostics` - Run Diagnostic Tests

Run diagnostic tests on a fitted model with the original data.

**Usage:**
```bash
ag diagnostics -m <model_file> -d <data_file> [-o <output_file>]
```

**Options:**
- `-m, --model` (required): Input model file in JSON format
- `-d, --data` (required): Input data file in CSV format
- `-o, --output` (optional): Output diagnostics file in JSON format

**Example:**
```bash
# Run diagnostics and display results
ag diagnostics -m fitted_model.json -d timeseries.csv

# Run diagnostics and save results to JSON
ag diagnostics -m best_model.json -d returns.csv -o diagnostics.json
```

**Output:**
- Ljung-Box test results for standardized residuals
- Ljung-Box test results for squared residuals
- Jarque-Bera test results (normality test)
- Optional JSON file with detailed test statistics

## Data Format

### Input Data (CSV)

The CLI expects CSV files with a header row. Only the first column is used:

```csv
returns
0.012
-0.005
0.018
...
```

For multi-column CSV files, the CLI will automatically use the first column:

```csv
date,returns,volume
2024-01-01,0.012,1000
2024-01-02,-0.005,1200
...
```

**For detailed CSV format specifications and requirements, see [File Formats Documentation](file_formats.md).**

### Model Files (JSON)

Models are saved and loaded in JSON format containing:
- Model specification (ARIMA and GARCH orders)
- Parameter estimates
- Model state (for sequential forecasting)
- Metadata (timestamp, version)

Example model JSON structure:

```json
{
  "spec": {
    "arima": {"p": 1, "d": 1, "q": 1},
    "garch": {"p": 1, "q": 1}
  },
  "parameters": {
    "arima": {
      "intercept": 0.05,
      "ar_coef": [0.6],
      "ma_coef": [0.3]
    },
    "garch": {
      "omega": 0.01,
      "alpha_coef": [0.1],
      "beta_coef": [0.85]
    }
  },
  ...
}
```

**For complete model JSON schema and field descriptions, see [File Formats Documentation](file_formats.md#model-files-json).**

## Workflow Examples

### Example 1: Basic Fitting and Forecasting

```bash
# 1. Fit a model to your data
ag fit -d my_returns.csv -a 1,1,1 -g 1,1 -o my_model.json

# 2. Generate forecasts
ag forecast -m my_model.json -n 20 -o my_forecasts.csv

# 3. Run diagnostics to check model adequacy
ag diagnostics -m my_model.json -d my_returns.csv
```

### Example 2: Model Selection Workflow

```bash
# 1. Automatically select the best model
ag select -d my_returns.csv --max-p 2 --max-q 2 -c BIC -o best_model.json

# 2. Generate forecasts from selected model
ag forecast -m best_model.json -n 30 -o forecasts.csv

# 3. Verify model with diagnostics
ag diagnostics -m best_model.json -d my_returns.csv -o diag_results.json
```

### Example 3: Simulation Study

```bash
# 1. Simulate synthetic data with default parameters (Gaussian)
ag sim -a 1,1,1 -g 1,1 -n 1000 -s 42 -o synthetic.csv

# 2. Fit a model to the synthetic data
ag fit -d synthetic.csv -a 1,1,1 -g 1,1 -o recovered_model.json

# 3. Compare recovered parameters with true parameters
# (inspect recovered_model.json to verify parameter recovery)
```

### Example 4: Student-t Innovation Workflow

```bash
# 1. Fit a model with Gaussian innovations (default)
ag fit -d market_returns.csv -a 1,0,1 -g 1,1 -o gaussian_model.json

# 2. Check distribution comparison in the output
# If Student-t is recommended, refit with Student-t innovations:
ag fit -d market_returns.csv -a 1,0,1 -g 1,1 --t-dist 5.0 -o student_t_model.json

# 3. Simulate heavy-tailed data using Student-t for stress testing
ag sim -a 1,0,1 -g 1,1 -n 1000 --t-dist 4.0 -o stress_scenario.csv
```

### Example 5: Monte Carlo Simulation from Fitted Model

```bash
# 1. Fit a model to real data
ag fit -d market_returns.csv -a 1,0,1 -g 1,1 -o market_model.json

# 2. Generate multiple simulation paths for risk analysis
ag simulate -m market_model.json -p 1000 -n 252 -s 42 -o mc_simulations.csv --stats

# 3. Analyze the distribution of simulated paths
# (use external tools or Python/R to analyze mc_simulations.csv)
```

### Example 6: Reproducibility Check

```bash
# 1. Fit and save a model
ag fit -d data.csv -a 1,1,1 -g 1,1 -o model.json

# 2. Generate simulation with specific seed
ag simulate -m model.json -p 1 -n 100 -s 12345 -o sim1.csv --stats

# 3. Verify reproducibility - should produce identical first path
ag simulate -m model.json -p 1 -n 100 -s 12345 -o sim2.csv --stats

# 4. Compare the first path values (should be identical)
```

## Tips and Best Practices

### Model Specification

- **Start simple**: Begin with low-order models like ARIMA(1,1,1)-GARCH(1,1)
- **Use model selection**: Let the `select` subcommand find the optimal orders
- **Check diagnostics**: Always run diagnostics to verify model adequacy

### Data Preparation

- **Sufficient data**: ARIMA-GARCH models require at least 100-200 observations for reliable estimation
- **Stationarity**: Consider differencing (d > 0) for non-stationary series
- **Outliers**: Large outliers can affect parameter estimates

### Performance

- **Model selection is slow**: The `select` subcommand with CV criterion and large search space can take several minutes
- **Use BIC for speed**: BIC criterion is much faster than cross-validation
- **Parallel execution**: Currently not supported, but planned for future versions

### Interpretation

- **AIC vs. BIC**: BIC tends to select more parsimonious models; AIC may fit better
- **CV for forecasting**: Use cross-validation criterion when forecast accuracy is the primary goal
- **Diagnostics**: p-values > 0.05 in Ljung-Box tests indicate adequate fit

## Error Handling

The CLI provides informative error messages for common issues:

- **Insufficient data**: At least 10 observations required, more recommended
- **Convergence failure**: Try different initial values or simpler models
- **Invalid model specification**: Check that p, d, q, P, Q are non-negative integers
- **File not found**: Verify file paths are correct

## Getting Help

For help with a specific subcommand:

```bash
ag fit --help
ag select --help
ag forecast --help
ag sim --help
ag diagnostics --help
```

For library usage and advanced features, see:
- [README.md](../README.md) - Library overview and examples
- [File Formats](file_formats.md) - CSV and JSON format specifications
- [Model Selection](model_selection.md) - Selection criteria and strategies
- [examples/](../examples/) - Complete example programs
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development guidelines

## Future Enhancements

Planned improvements for future versions:

- Custom parameter specification for simulation
- Batch processing of multiple files
- Plotting and visualization output
- Support for exogenous variables
- Parallel model selection
- Progress bars for long-running operations
