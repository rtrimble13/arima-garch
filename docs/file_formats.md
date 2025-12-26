# File Formats

This document describes the file formats used by the `arima-garch` library and CLI tool for input data and model persistence.

## CSV Files

### Input Data Format

The library and CLI tool expect time series data in CSV (Comma-Separated Values) format. The CSV parser is flexible and supports multiple layouts.

#### Simple Format (Single Column)

The simplest format is a single column of values, optionally with a header:

```csv
Value
0.0123
-0.0145
0.0201
-0.0089
0.0156
```

Or without header:
```csv
1.5
2.3
1.8
3.2
2.9
```

**Requirements:**
- Each row contains a single numeric value
- Values can be positive or negative
- Values can use decimal notation (e.g., `0.0123`) or scientific notation (e.g., `1.23e-2`)
- Optional header row (will be automatically detected and skipped)

#### Multi-Column Format

When multiple columns are present, the **first column after any date/time column** is used as the time series data. Other columns are ignored by the CLI.

```csv
Date,Value,Volume
2024-01-01,0.0123,1000
2024-01-02,-0.0145,1200
2024-01-03,0.0201,1100
```

The CLI will automatically use the first numeric column (in this case, "Value").

```csv
returns,volatility,other
0.012,-0.005,100
-0.005,0.003,120
0.018,0.004,110
```

The CLI will use the first column ("returns").

**Requirements:**
- Comma-separated values
- First column is used as the time series data (unless it's a date/time column)
- Optional header row
- Other columns are ignored by CLI tools

#### Date-Indexed Format

Time series with date or timestamp indices are supported:

```csv
Date,Returns
2024-01-01,0.0123
2024-01-02,-0.0145
2024-01-03,0.0201
```

**Requirements:**
- First column can be date/time (will be ignored, only values are used)
- Second column contains the numeric values to be modeled
- Date formats are not parsed; the library only uses the numeric values

### Output Data Format

#### Forecast Output

When generating forecasts with `ag forecast`, the output CSV contains:

```csv
step,mean,variance,std_dev
1,0.0125,0.0001,0.01
2,0.0126,0.00012,0.01095
3,0.0127,0.00013,0.0114
```

**Columns:**
- `step`: Forecast horizon (1, 2, 3, ...)
- `mean`: Mean forecast (conditional expectation)
- `variance`: Forecast variance (conditional variance)
- `std_dev`: Standard deviation (square root of variance)

#### Simulation Output (Simple)

When using `ag sim`, the output contains simulated returns and volatilities:

```csv
observation,return,volatility
1,0.0156,0.0098
2,-0.0089,0.0102
3,0.0234,0.0105
```

**Columns:**
- `observation`: Time index (1, 2, 3, ...)
- `return`: Simulated return value
- `volatility`: Simulated conditional volatility (standard deviation)

#### Simulation Output (Multiple Paths)

When using `ag simulate` with multiple paths:

```csv
path,observation,return,volatility
1,1,0.0156,0.0098
1,2,-0.0089,0.0102
1,3,0.0234,0.0105
2,1,0.0143,0.0097
2,2,-0.0067,0.0099
2,3,0.0198,0.0103
```

**Columns:**
- `path`: Path number (1, 2, 3, ...)
- `observation`: Time index within path
- `return`: Simulated return value
- `volatility`: Simulated conditional volatility

### CSV Reading Options

When using the library API, you can customize CSV reading behavior:

```cpp
ag::io::CsvReaderOptions options;
options.has_header = true;      // Skip first row as header
options.value_column = 1;       // Use second column (0-indexed)
options.delimiter = ',';         // Use comma as separator

auto result = ag::io::CsvReader::read("data.csv", options);
```

### Data Requirements

**Minimum Size:**
- At least 10 observations required for model fitting
- Recommended: 100-200+ observations for reliable parameter estimation
- More data generally improves estimation quality

**Data Characteristics:**
- Values should be returns or differences (not prices/levels for ARIMA models with d=0)
- For non-stationary data, use differencing (d > 0) or transform data beforehand
- Missing values are not supported; data must be complete
- Extreme outliers can affect parameter estimates

## Model Files (JSON)

### Overview

Fitted models are saved in JSON format, which enables:
- **Persistence**: Save fitted models for later use
- **Reproducibility**: Exact parameter values are preserved
- **Portability**: JSON is human-readable and language-agnostic
- **Versioning**: Track model versions and metadata

### Model JSON Schema

The model JSON file has the following structure:

```json
{
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "version": "0.1.0",
    "model_type": "ArimaGarch"
  },
  "spec": {
    "arima": {
      "p": 1,
      "d": 1,
      "q": 1
    },
    "garch": {
      "p": 1,
      "q": 1
    }
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
  "state": {
    "arima": {
      "observation_history": [1.2, 1.5, 1.3],
      "residual_history": [0.1, -0.05, 0.15],
      "differenced_series": [0.3, -0.2],
      "initialized": true
    },
    "garch": {
      "variance_history": [0.01, 0.012, 0.011],
      "squared_residual_history": [0.0001, 0.00015, 0.0001],
      "initial_variance": 0.01,
      "initialized": true
    }
  }
}
```

### Field Descriptions

#### `metadata` (object)

Contains information about when and how the model was created.

- **`timestamp`** (string): ISO 8601 timestamp of when the model was saved
  - Format: `YYYY-MM-DDTHH:MM:SSZ`
  - Example: `"2024-01-15T10:30:00Z"`

- **`version`** (string): Library version used to create the model
  - Format: Semantic versioning (MAJOR.MINOR.PATCH)
  - Example: `"0.1.0"`
  - Used for compatibility checking

- **`model_type`** (string): Type of model
  - Currently always `"ArimaGarch"`
  - Reserved for future model types

#### `spec` (object)

Defines the model structure and orders.

##### `spec.arima` (object)

ARIMA model specification:

- **`p`** (integer, ≥ 0): Autoregressive (AR) order
  - Number of AR terms in the model
  - Example: `p=1` means AR(1) term: φ₁ × yₜ₋₁

- **`d`** (integer, ≥ 0): Differencing order
  - Number of times the series is differenced
  - `d=0`: No differencing (stationary series)
  - `d=1`: First differences (removes linear trend)
  - `d=2`: Second differences (removes quadratic trend)

- **`q`** (integer, ≥ 0): Moving average (MA) order
  - Number of MA terms in the model
  - Example: `q=1` means MA(1) term: θ₁ × εₜ₋₁

##### `spec.garch` (object)

GARCH model specification:

- **`p`** (integer, ≥ 1): GARCH order
  - Number of lagged conditional variance terms
  - Example: `p=1` means GARCH(1) term: β₁ × hₜ₋₁

- **`q`** (integer, ≥ 1): ARCH order
  - Number of lagged squared residual terms
  - Example: `q=1` means ARCH(1) term: α₁ × ε²ₜ₋₁

**Note:** Both GARCH p and q must be at least 1 for a valid GARCH model.

#### `parameters` (object)

Contains all fitted parameter values.

##### `parameters.arima` (object)

ARIMA model parameters:

- **`intercept`** (number): Constant term in the mean equation
  - Also called the drift or mean level
  - Represents the long-run mean (when appropriately scaled)

- **`ar_coef`** (array of numbers): Autoregressive coefficients [φ₁, φ₂, ..., φₚ]
  - Length must equal `spec.arima.p`
  - Each coefficient represents the impact of past values
  - Stationarity requires roots of AR polynomial outside unit circle

- **`ma_coef`** (array of numbers): Moving average coefficients [θ₁, θ₂, ..., θ_q]
  - Length must equal `spec.arima.q`
  - Each coefficient represents the impact of past errors
  - Invertibility requires roots of MA polynomial outside unit circle

##### `parameters.garch` (object)

GARCH model parameters:

- **`omega`** (number, > 0): Constant term in variance equation
  - Also written as ω or α₀
  - Represents the baseline variance level
  - Must be positive for positive variance

- **`alpha_coef`** (array of numbers): ARCH coefficients [α₁, α₂, ..., α_q]
  - Length must equal `spec.garch.q`
  - Captures the impact of past squared residuals
  - Must be non-negative for positive variance

- **`beta_coef`** (array of numbers): GARCH coefficients [β₁, β₂, ..., βₚ]
  - Length must equal `spec.garch.p`
  - Captures the persistence of volatility
  - Must be non-negative for positive variance

**Stationarity Constraint:**
The sum α₁ + α₂ + ... + α_q + β₁ + β₂ + ... + βₚ must be < 1 for variance stationarity.

#### `state` (object)

Contains the model's internal state for sequential forecasting. This allows the model to continue from where it left off when processing additional data.

##### `state.arima` (object)

ARIMA model state:

- **`observation_history`** (array of numbers): Recent observed values
  - Length up to `max(p, q)`
  - Most recent observation is last in array

- **`residual_history`** (array of numbers): Recent residuals (errors)
  - Length up to `q`
  - Most recent residual is last in array

- **`differenced_series`** (array of numbers): Differenced values (if d > 0)
  - Used to maintain differencing state
  - Empty if `d = 0`

- **`initialized`** (boolean): Whether the model has been initialized with data

##### `state.garch` (object)

GARCH model state:

- **`variance_history`** (array of numbers): Recent conditional variances
  - Length up to `p`
  - Most recent variance is last in array

- **`squared_residual_history`** (array of numbers): Recent squared residuals
  - Length up to `q`
  - Most recent squared residual is last in array

- **`initial_variance`** (number, > 0): Initial variance estimate
  - Used as starting value for variance recursion

- **`initialized`** (boolean): Whether the model has been initialized with data

### Versioning Strategy

The library uses semantic versioning (MAJOR.MINOR.PATCH) in the model JSON:

- **MAJOR**: Incremented for breaking changes to JSON schema
  - Models from different major versions may not be compatible
  - Example: Changing field names or structure

- **MINOR**: Incremented for new features that maintain backward compatibility
  - Models from older minor versions can be loaded by newer versions
  - Example: Adding optional fields

- **PATCH**: Incremented for bug fixes and patches
  - No schema changes, only implementation fixes

**Compatibility Guidelines:**
- The library will attempt to load models with the same major version
- A warning may be issued if minor versions differ
- Models from future versions may not load in older library versions
- Always save the library version when creating derived results

### Creating and Modifying Model JSON

While model JSON files are typically created by the CLI or library, you can also create or modify them manually.

**Example: Creating a Model JSON from Scratch**

```json
{
  "metadata": {
    "timestamp": "2024-01-15T12:00:00Z",
    "version": "0.1.0",
    "model_type": "ArimaGarch"
  },
  "spec": {
    "arima": {"p": 1, "d": 0, "q": 1},
    "garch": {"p": 1, "q": 1}
  },
  "parameters": {
    "arima": {
      "intercept": 0.0,
      "ar_coef": [0.5],
      "ma_coef": [0.3]
    },
    "garch": {
      "omega": 0.01,
      "alpha_coef": [0.1],
      "beta_coef": [0.85]
    }
  },
  "state": {
    "arima": {
      "observation_history": [],
      "residual_history": [],
      "differenced_series": [],
      "initialized": false
    },
    "garch": {
      "variance_history": [],
      "squared_residual_history": [],
      "initial_variance": 0.01,
      "initialized": false
    }
  }
}
```

**Validation Rules:**
- Array lengths must match specification orders
- GARCH parameters must be non-negative
- Sum of GARCH α and β coefficients should be < 1
- Initial variance must be positive
- All required fields must be present

### Using Model JSON Files

**Loading a Model:**
```bash
# Generate forecasts from saved model
ag forecast -m model.json -n 10 -o forecasts.csv

# Simulate from saved model
ag simulate -m model.json -p 5 -n 1000 -o simulations.csv

# Run diagnostics on saved model
ag diagnostics -m model.json -d original_data.csv
```

**Saving a Model:**
```bash
# Save model after fitting
ag fit -d data.csv -a 1,1,1 -g 1,1 -o model.json

# Save model after selection
ag select -d data.csv --max-p 2 --max-q 2 -o best_model.json
```

## Diagnostics Files (JSON)

### Overview

Diagnostic test results can be saved to JSON format for further analysis or reporting.

### Diagnostics JSON Schema

```json
{
  "ljung_box_residuals": {
    "statistic": 15.234,
    "p_value": 0.123,
    "lags": 10,
    "test_type": "standardized_residuals"
  },
  "ljung_box_squared_residuals": {
    "statistic": 8.456,
    "p_value": 0.567,
    "lags": 10,
    "test_type": "squared_residuals"
  },
  "jarque_bera": {
    "statistic": 2.345,
    "p_value": 0.309,
    "test_type": "normality"
  },
  "timestamp": "2024-01-15T14:30:00Z"
}
```

### Field Descriptions

#### Ljung-Box Test Results

Tests for autocorrelation in residuals or squared residuals.

- **`statistic`** (number): Test statistic value
- **`p_value`** (number): p-value for the test
  - Values > 0.05 suggest adequate fit (no significant autocorrelation)
  - Values ≤ 0.05 suggest model inadequacy (significant autocorrelation remains)
- **`lags`** (integer): Number of lags tested
- **`test_type`** (string): Type of residuals tested
  - `"standardized_residuals"`: Tests mean equation adequacy
  - `"squared_residuals"`: Tests variance equation adequacy

#### Jarque-Bera Test

Tests for normality of standardized residuals.

- **`statistic`** (number): Test statistic value
- **`p_value`** (number): p-value for the test
  - Values > 0.05 suggest residuals are approximately normal
  - Values ≤ 0.05 suggest non-normal residuals (fat tails, skewness)

#### Metadata

- **`timestamp`** (string): ISO 8601 timestamp when diagnostics were generated

### Using Diagnostics Files

```bash
# Save diagnostics to JSON
ag diagnostics -m model.json -d data.csv -o diagnostics.json

# View diagnostics (prints to console)
ag diagnostics -m model.json -d data.csv
```

## Common File Workflows

### Workflow 1: Data Preparation

```bash
# Prepare your data as CSV
echo "returns" > data.csv
echo "0.012" >> data.csv
echo "-0.005" >> data.csv
echo "0.018" >> data.csv
# ... add more data points

# Verify your data format
head -n 5 data.csv
```

### Workflow 2: Model Fitting and Persistence

```bash
# Fit model and save
ag fit -d data.csv -a 1,1,1 -g 1,1 -o model.json

# Use saved model for forecasting
ag forecast -m model.json -n 20 -o forecasts.csv

# Verify forecasts
head -n 5 forecasts.csv
```

### Workflow 3: Model Selection and Analysis

```bash
# Select best model
ag select -d data.csv --max-p 2 --max-q 2 -c BIC -o best_model.json

# Run diagnostics
ag diagnostics -m best_model.json -d data.csv -o diagnostics.json

# Generate forecasts
ag forecast -m best_model.json -n 10 -o forecasts.csv
```

### Workflow 4: Simulation and Validation

```bash
# Fit model to real data
ag fit -d real_data.csv -a 1,0,1 -g 1,1 -o fitted_model.json

# Generate simulation paths
ag simulate -m fitted_model.json -p 100 -n 252 -o simulations.csv --stats

# Analyze simulations with external tools
# python analyze_simulations.py simulations.csv
```

## Best Practices

### Data Files
- Always include headers in your CSV files for clarity
- Use descriptive column names (e.g., "returns", "log_returns")
- Keep one value per row; avoid manual line breaks
- Check for missing values before analysis (library doesn't support missing data)
- Save data with sufficient precision (at least 6 decimal places for returns)

### Model Files
- Always save fitted models with descriptive filenames
- Include date or version in filename (e.g., `model_2024-01-15.json`)
- Store models under version control for reproducibility
- Document the data used to fit each model
- Keep a log of model selection results for comparison

### File Organization
```
project/
├── data/
│   ├── raw/
│   │   └── market_returns.csv
│   └── processed/
│       └── stationary_returns.csv
├── models/
│   ├── model_v1_2024-01-15.json
│   └── model_v2_2024-01-20.json
├── forecasts/
│   ├── forecast_2024-01-15.csv
│   └── forecast_2024-01-20.csv
└── diagnostics/
    └── diagnostics_v2.json
```

## Troubleshooting

### Common CSV Issues

**Problem:** "Failed to parse CSV"
- **Solution:** Check for non-numeric values, ensure proper delimiters

**Problem:** "Insufficient data"
- **Solution:** Ensure at least 10 observations (preferably 100+)

**Problem:** "Could not read file"
- **Solution:** Verify file path and permissions

### Common JSON Issues

**Problem:** "JSON parse error"
- **Solution:** Validate JSON syntax with a JSON validator

**Problem:** "Version mismatch"
- **Solution:** Check model version vs. library version

**Problem:** "Parameter array size mismatch"
- **Solution:** Ensure array lengths match model specification orders

## References

- [CLI Documentation](cli.md) - Command-line interface usage
- [Model Selection](model_selection.md) - Model selection strategies
- [Parameter Constraints](parameter_constraints.md) - Parameter validation rules
- [README](../README.md) - Library overview and examples
