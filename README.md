# arima-garch

A C++ library and command-line interface (CLI) for ARIMA-GARCH time series modeling.

## Project Goals

This project provides tools for:
- **Fitting** ARIMA-GARCH models to time series data
- **Forecasting** future values using fitted models
- **Simulating** synthetic time series data from ARIMA-GARCH specifications

ARIMA-GARCH models combine autoregressive integrated moving average (ARIMA) for the conditional mean with generalized autoregressive conditional heteroskedasticity (GARCH) for the conditional variance, making them ideal for modeling financial time series with volatility clustering.

## Build Instructions

This project uses CMake for building. Requirements:
- CMake 3.14 or higher
- C++20 compatible compiler (GCC 10+, Clang 10+, MSVC 2019+)
- Git (for automatic dependency fetching)
- Internet connection (first build only)

Dependencies are automatically downloaded and built using CMake FetchContent. See [docs/dependencies.md](docs/dependencies.md) for details on dependency management strategy.

### Building from Source

```bash
# Configure the build
cmake -S . -B build

# Build the project
cmake --build build

# Optionally, install (requires appropriate permissions)
cmake --install build
```

### Build Targets

The build produces the following targets:
- `arimagarch` - Static library for ARIMA-GARCH modeling
- `ag` - Command-line interface executable
- Unit test executables (29 test suites, including cross-validation tests)
- Example programs:
  - `example_simulation` - Demonstrates simulation of ARIMA-GARCH paths
  - `example_json_io` - Demonstrates model serialization to/from JSON
  - `example_model_selector` - Demonstrates automatic model selection with BIC/AIC/AICc/CV
  - `example_basic` - Basic usage example
  - Additional examples in `examples/` directory
- Benchmark programs:
  - `bench_likelihood` - Benchmarks likelihood computation performance
  - `bench_optimizer` - Benchmarks optimizer performance

To run a specific example:
```bash
./build/examples/example_simulation
```

### Running Benchmarks

To track performance and detect regressions:

```bash
# Run likelihood computation benchmark
./build/benchmarks/bench_likelihood

# Run optimizer benchmark
./build/benchmarks/bench_optimizer
```

The benchmarks measure:
- **Likelihood computation**: Throughput (observations/second) on mid-size data (5000 observations)
- **Optimizer performance**: Average optimization time and convergence rate on standard fitting tasks (1000 observations)

Benchmarks are useful for:
- Tracking performance over time
- Detecting performance regressions
- Comparing different model specifications

## Usage

### Command-Line Interface

Run the CLI tool:

```bash
# After building
./build/src/ag

# Fit a model to data with Gaussian innovations (default)
./build/src/ag fit --data examples/returns.csv --arima 1,0,1 --garch 1,1 --out model.json

# Fit a model with Student-t innovations (heavier tails)
./build/src/ag fit --data examples/returns.csv --arima 1,0,1 --garch 1,1 --t-dist 5.0 --out model_tdist.json

# Automatic model selection (includes distribution comparison)
./build/src/ag select --data examples/returns.csv --max-p 2 --max-q 2 --out model.json

# Forecast future values
./build/src/ag forecast --model model.json --horizon 10 --out forecasts.csv

# Simulate synthetic data with Gaussian innovations (default)
./build/src/ag sim --arima 1,0,1 --garch 1,1 --length 1000 --out simulated.csv

# Simulate synthetic data with Student-t innovations (for stress testing)
./build/src/ag sim --arima 1,0,1 --garch 1,1 --length 1000 --t-dist 4.0 --out simulated_tdist.csv

# Simulate multiple paths from a fitted model
./build/src/ag simulate --model model.json --paths 10 --length 1000 --seed 42 --out sim_returns.csv --stats

# Run diagnostics on a fitted model
./build/src/ag diagnostics --model model.json --data examples/returns.csv --out diagnostics.json
```

### Library Usage

#### Simulating Time Series

```cpp
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"

int main() {
    // Define an ARIMA(1,0,1)-GARCH(1,1) model
    ag::models::ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ag::models::composite::ArimaGarchParameters params(spec);
    
    // Set model parameters
    params.arima_params.intercept = 0.05;
    params.arima_params.ar_coef[0] = 0.6;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.01;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.85;
    
    // Create simulator and generate synthetic data with Normal innovations (default)
    ag::simulation::ArimaGarchSimulator simulator(spec, params);
    auto result = simulator.simulate(1000, 42);  // 1000 observations, seed=42
    
    // Or use Student-t innovations for heavier tails (better for modeling extreme events)
    auto result_t = simulator.simulate(1000, 42, 
                                       ag::simulation::InnovationDistribution::StudentT, 
                                       5.0);  // df=5
    
    // Access simulated returns and volatilities
    std::vector<double>& returns = result.returns;
    std::vector<double>& volatilities = result.volatilities;
    
    return 0;
}
```

The simulator supports two innovation distributions:
- **Normal (N(0,1))**: Default, suitable for most applications
- **Student-t**: Optional, produces heavier tails (higher kurtosis), useful for modeling extreme events and fat-tailed distributions

See `examples/example_simulation.cpp` for a complete working example with statistical analysis and comparisons between both distributions.

#### Saving and Loading Models

Models can be serialized to JSON format for persistence, versioning, and reproducibility:

```cpp
#include "ag/io/Json.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"

int main() {
    // Create and configure a model
    ag::models::ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ag::models::composite::ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.05;
    params.arima_params.ar_coef[0] = 0.6;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.01;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.85;
    
    ag::models::composite::ArimaGarchModel model(spec, params);
    
    // Save model to JSON file
    auto save_result = ag::io::JsonWriter::saveModel("model.json", model);
    
    // Load model from JSON file
    auto load_result = ag::io::JsonReader::loadModel("model.json");
    if (load_result.has_value()) {
        auto loaded_model = *load_result;
        // Use loaded_model for forecasting...
    }
    
    return 0;
}
```

The JSON format includes:
- Model specification (ARIMA and GARCH orders)
- All fitted parameters (coefficients, intercepts)
- Model state (histories for sequential forecasting)
- Metadata (timestamp, version)

This enables reproducible forecasts and easy model deployment. See `examples/example_json_io.cpp` for a complete demonstration.

#### Model Selection

The library supports automatic model selection from a grid of candidate specifications using either information criteria or cross-validation:

```cpp
#include "ag/selection/ModelSelector.hpp"
#include "ag/selection/CandidateGrid.hpp"

// Generate candidate specifications
ag::selection::CandidateGridConfig config(2, 0, 2, 1, 1);  // ARIMA(0-2,0,0-2)-GARCH(1,1)
ag::selection::CandidateGrid grid(config);
auto candidates = grid.generate();

// Option 1: Select using BIC (fast, good for large candidate sets)
ag::selection::ModelSelector selector_bic(ag::selection::SelectionCriterion::BIC);
auto result_bic = selector_bic.select(data.data(), data.size(), candidates);

// Option 2: Select using Cross-Validation (slower, better forecast performance assessment)
ag::selection::ModelSelector selector_cv(ag::selection::SelectionCriterion::CV);
auto result_cv = selector_cv.select(data.data(), data.size(), candidates);

// Use the best model
if (result_bic.has_value()) {
    auto best_model = ag::models::composite::ArimaGarchModel(
        result_bic->best_spec, result_bic->best_parameters);
}
```

**Selection Criteria:**
- **BIC/AIC/AICc**: Fast, uses information criteria to balance fit and complexity
- **CV (Cross-Validation)**: Slower but more direct measure of out-of-sample forecast performance using rolling origin validation with 1-step-ahead MSE

See `examples/example_model_selector.cpp` for a complete demonstration.

## Visualization

The `ag-viz` Python package provides publication-quality visualization tools for ARIMA-GARCH model outputs. It wraps the C++ CLI and generates professional plots for diagnostics, forecasts, residuals, and simulations.

### Installation

```bash
# Install the C++ CLI first
cmake -S . -B build
cmake --build build

# Install the Python visualization package
pip install -e python/
```

### Quick Start

```bash
# Fit model and generate diagnostic plots
ag-viz fit -d examples/returns.csv -a 1,0,1 -g 1,1 -o model.json

# Generate forecast with confidence intervals
ag-viz forecast -m model.json -n 30 -o forecast.csv

# Run comprehensive diagnostics with plots
ag-viz diagnostics -m model.json -d examples/returns.csv -o ./diagnostics/

# Simulate and visualize paths
ag-viz simulate -m model.json -p 100 -n 1000 -o simulation.csv

# Generate Markdown reports for any subcommand
ag-viz fit -d examples/returns.csv -a 1,0,1 -g 1,1 -o model.json --markdown
ag-viz forecast -m model.json -n 30 -o forecast.csv --markdown
```

### Features

- **Fit Diagnostics**: Time series plots with model specifications and summary statistics
- **Forecast Plots**: Mean forecasts with 68% and 95% confidence interval bands
- **Residual Diagnostics**: Multi-panel plots including standardized residuals, histograms, QQ-plots, and ACF plots
- **Simulation Paths**: Visualize multiple paths with percentile bands and terminal value distributions
- **Markdown Reports**: Professional analysis reports with embedded visuals, interpretations, and recommendations (use `--markdown` flag)

For detailed documentation, see [python/README.md](python/README.md).

## Project Structure

```
arima-garch/
├── include/ag/       # Public header files
├── src/              # Library and CLI implementation
├── tests/            # Unit tests
├── examples/         # Example programs
├── python/           # Python visualization package (ag-viz)
│   ├── ag_viz/       # Package source code
│   ├── tests/        # Python tests
│   └── examples/     # Jupyter notebook demos
├── docs/             # Documentation
├── cmake/            # CMake modules and scripts
└── .github/          # GitHub Actions workflows
    └── workflows/
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

Quick overview:
- Code follows C++20 best practices (see [docs/style.md](docs/style.md))
- Dependencies are managed via CMake FetchContent (see [docs/dependencies.md](docs/dependencies.md))
- Use standard library and approved dependencies (fmt, nlohmann/json, CLI11, Catch2)
- Update documentation for new features
- Add tests for new functionality

## Documentation

- [CLI Documentation](docs/cli.md) - Command-line interface usage and examples
- [Python Visualization Package](python/README.md) - ag-viz plotting and visualization tools
- [File Formats](docs/file_formats.md) - CSV and JSON format specifications
- [Model Selection](docs/model_selection.md) - Automatic model selection strategies
- [Parameter Constraints](docs/parameter_constraints.md) - Parameter validation rules
- [Parameter Initialization](docs/parameter_initialization.md) - Parameter initialization strategies
- [ARIMA Residuals](docs/arima_residuals.md) - ARIMA residual handling
- [Dependencies](docs/dependencies.md) - Dependency management
- [Style Guide](docs/style.md) - Code style guidelines

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
