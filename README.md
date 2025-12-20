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
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)

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
- `ag-cli` - Command-line interface executable
- `test_placeholder` - Unit tests (placeholder)
- `example_basic` - Basic usage example

## Usage

### Command-Line Interface

Run the CLI tool:

```bash
# After building
./build/src/ag-cli

# Example usage (to be implemented):
# Fit a model to data
./build/src/ag-cli fit --data timeseries.csv --arima 1,1,1 --garch 1,1

# Forecast future values
./build/src/ag-cli forecast --model model.bin --horizon 10

# Simulate synthetic data
./build/src/ag-cli simulate --arima 1,1,1 --garch 1,1 --samples 1000
```

### Library Usage

```cpp
#include "ag/arima_garch.hpp"

int main() {
    // Future: Example of using the library API
    // ag::ARIMAGARCHModel model(p, d, q, P, Q);
    // model.fit(data);
    // auto forecast = model.forecast(horizon);
    return 0;
}
```

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

Contributions are welcome! Please ensure:
- Code follows C++17 best practices
- Use standard library and Boost where appropriate
- Update documentation for new features
- Add tests for new functionality

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
