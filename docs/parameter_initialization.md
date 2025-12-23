# Parameter Initialization and Random Restarts

This document describes the parameter initialization and random restart functionality added to improve convergence reliability for ARIMA-GARCH model fitting.

## Overview

The implementation provides two key features:

1. **Intelligent Parameter Initialization**: Uses heuristics based on time series properties (ACF/PACF for ARIMA, method-of-moments for GARCH) to generate good starting points for optimization.

2. **Random Restarts**: Performs multiple optimization runs from perturbed starting points to improve the chance of finding the global optimum.

## Usage

### Basic Parameter Initialization

```cpp
#include "ag/estimation/ParameterInitialization.hpp"
#include "ag/models/ArimaGarchSpec.hpp"

// Your time series data
std::vector<double> data = {...};

// Define model specification: AR(1)-GARCH(1,1)
ag::models::ArimaGarchSpec spec(1, 0, 0, 1, 1);

// Initialize parameters
auto [arima_params, garch_params] = 
    ag::estimation::initializeArimaGarchParameters(data.data(), data.size(), spec);

// arima_params and garch_params now contain reasonable starting values
```

### Optimization with Random Restarts

```cpp
#include "ag/estimation/Optimizer.hpp"

// Create optimizer
ag::estimation::NelderMeadOptimizer optimizer(1e-6, 1e-6, 2000);

// Your objective function
auto objective = [&](const std::vector<double>& params) -> double {
    // ... compute negative log-likelihood ...
};

// Optimize with 5 random restarts
auto result = ag::estimation::optimizeWithRestarts(
    optimizer, objective, initial_params,
    5,      // number of restarts
    0.2,    // perturbation scale (20% of parameter magnitude)
    12345   // random seed for reproducibility
);

if (result.converged) {
    std::cout << "Converged! Successful restarts: " 
              << result.successful_restarts << std::endl;
    // Use result.parameters
}
```

## Implementation Details

### ARIMA Initialization

The ARIMA initialization uses the following heuristics:

- **Intercept**: Set to the sample mean of the (differenced) data
- **AR coefficients**: Initialized from PACF values at corresponding lags, scaled by 0.9 for stability
- **MA coefficients**: Initialized from negative ACF values, scaled by 0.9

These heuristics provide a reasonable starting point based on the autocorrelation structure of the data.

### GARCH Initialization

The GARCH initialization uses method-of-moments:

- **Target persistence**: Sum(alpha) + Sum(beta) ≈ 0.90 (typical for financial data)
- **Alpha coefficients** (ARCH): Allocated 30% of persistence, distributed evenly
- **Beta coefficients** (GARCH): Allocated 70% of persistence, distributed with decay
- **Omega**: Calculated from the unconditional variance formula to match sample variance

The parameters are constructed to satisfy positivity and stationarity constraints.

### Random Restarts

The random restart mechanism:

1. Runs initial optimization from the initialized parameters
2. For each restart:
   - Generates a perturbed starting point by adding Gaussian noise
   - Noise scale: `scale * max(|param|, 0.01)` for each parameter
   - Runs optimization from the perturbed point
3. Keeps track of the best result across all attempts
4. Returns the best result with statistics on successful improvements

## Performance

Based on synthetic data tests:

- **Convergence Rate**: >70% on AR(1)-GARCH(1,1) synthetic data with 500 observations
- **Parameter Accuracy**: Typical estimation errors < 0.05 for well-identified parameters
- **Computational Cost**: ~3x the cost of a single optimization run with 3 restarts

## API Reference

### Parameter Initialization Functions

```cpp
// Initialize ARIMA parameters from time series data
ag::models::arima::ArimaParameters 
initializeArimaParameters(const double* data, std::size_t size, 
                         const ag::models::ArimaSpec& spec);

// Initialize GARCH parameters from residuals
ag::models::garch::GarchParameters
initializeGarchParameters(const double* residuals, std::size_t size,
                         const ag::models::GarchSpec& spec);

// Initialize combined ARIMA-GARCH parameters
std::pair<ag::models::arima::ArimaParameters, ag::models::garch::GarchParameters>
initializeArimaGarchParameters(const double* data, std::size_t size,
                              const ag::models::ArimaGarchSpec& spec);
```

### Random Restart Optimization

```cpp
OptimizationResultWithRestarts
optimizeWithRestarts(IOptimizer& optimizer,
                    const IOptimizer::ObjectiveFunction& objective,
                    const std::vector<double>& initial_params,
                    int num_restarts = 5,
                    double perturbation_scale = 0.2,
                    unsigned int seed = 0);
```

The `OptimizationResultWithRestarts` structure extends `OptimizationResult` with:
- `restarts_performed`: Number of restarts attempted
- `successful_restarts`: Number of restarts that improved the objective

## Examples

See `examples/example_initialization.cpp` for a complete working example that:
1. Generates synthetic AR(1)-GARCH(1,1) data
2. Initializes parameters using heuristics
3. Optimizes with random restarts
4. Compares estimated parameters to true values

Run the example:
```bash
./build/examples/example_initialization
```

## Testing

Unit tests are provided in `tests/unit/test_estimation_initialization.cpp`:

- Tests for ARIMA initialization on AR(1) data
- Tests for GARCH initialization on GARCH(1,1) residuals
- Tests for combined ARIMA-GARCH initialization
- Tests for random restart functionality
- Integration tests with synthetic AR(1)-GARCH(1,1) data
- Convergence rate validation

Run tests:
```bash
cd build && ctest -R test_estimation_initialization
```

## References

- **ACF/PACF**: Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2015). Time Series Analysis: Forecasting and Control.
- **GARCH**: Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3), 307-327.
- **Method-of-Moments**: Francq, C., & Zakoïan, J. M. (2019). GARCH Models: Structure, Statistical Inference and Financial Applications.
