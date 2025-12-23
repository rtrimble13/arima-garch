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
- Unit test executables (24 test suites)
- Example programs:
  - `example_simulation` - Demonstrates simulation of ARIMA-GARCH paths
  - `example_basic` - Basic usage example
  - Additional examples in `examples/` directory

To run a specific example:
```bash
./build/examples/example_simulation
```

## Usage

### Command-Line Interface

Run the CLI tool:

```bash
# After building
./build/src/ag

# Example usage (to be implemented):
# Fit a model to data
./build/src/ag fit --data timeseries.csv --arima 1,1,1 --garch 1,1

# Forecast future values
./build/src/ag forecast --model model.bin --horizon 10

# Simulate synthetic data
./build/src/ag simulate --arima 1,1,1 --garch 1,1 --samples 1000
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

## Project Structure

```
arima-garch/
├── include/ag/       # Public header files
├── src/              # Library and CLI implementation
├── tests/            # Unit tests
├── examples/         # Example programs
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
