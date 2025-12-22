# ARIMA Residual Computation

This document describes the ARIMA residual computation functionality implemented in the library.

## Overview

The library provides classes for computing residuals (innovations) from ARIMA(p,d,q) models. Residuals represent the difference between observed values and the conditional mean predicted by the model.

## Components

### ArimaState

`ArimaState` manages the state required for ARIMA recursion:
- Historical observations (for AR component)
- Historical residuals (for MA component)
- Differenced series (when d > 0)

**Location**: `include/ag/models/arima/ArimaState.hpp`

### ArimaModel

`ArimaModel` implements the core residual computation logic:
- Computes conditional mean using AR and MA components
- Handles differencing (d > 0)
- Returns residuals via recursive computation

**Location**: `include/ag/models/arima/ArimaModel.hpp`

### ArimaParameters

`ArimaParameters` holds the model coefficients:
- `intercept`: Constant term (c or μ)
- `ar_coef`: AR coefficients (φ₁, φ₂, ..., φₚ)
- `ma_coef`: MA coefficients (θ₁, θ₂, ..., θ_q)

## Model Equation

For an ARIMA(p,d,q) model, after differencing d times, the conditional mean is:

```
E[y_t | history] = c + φ₁*y_{t-1} + ... + φₚ*y_{t-p} + θ₁*ε_{t-1} + ... + θ_q*ε_{t-q}
```

The residual (innovation) at time t is:

```
ε_t = y_t - E[y_t | history]
```

## Usage Example

```cpp
#include "ag/models/ArimaSpec.hpp"
#include "ag/models/arima/ArimaModel.hpp"

using ag::models::ArimaSpec;
using ag::models::arima::ArimaModel;
using ag::models::arima::ArimaParameters;

// Create an AR(1) model: ARIMA(1,0,0)
ArimaSpec spec(1, 0, 0);
ArimaModel model(spec);

// Set parameters
ArimaParameters params(1, 0);
params.intercept = 2.0;
params.ar_coef[0] = 0.7;

// Compute residuals for a time series
std::vector<double> data = {2.5, 3.45, 5.215, 5.45, 6.215};
auto residuals = model.computeResiduals(data.data(), data.size(), params);
```

## Supported Model Types

### AR(p) - Autoregressive

Example: ARIMA(1,0,0)
```
y_t = c + φ₁*y_{t-1} + ε_t
```

### MA(q) - Moving Average

Example: ARIMA(0,0,1)
```
y_t = c + ε_t + θ₁*ε_{t-1}
```

### ARMA(p,q) - Combined

Example: ARIMA(1,0,1)
```
y_t = c + φ₁*y_{t-1} + ε_t + θ₁*ε_{t-1}
```

### Differencing (d > 0)

Example: ARIMA(0,1,0) - Random Walk
```
Δy_t = y_t - y_{t-1} = ε_t
```

Example: ARIMA(1,1,0) - AR(1) on differences
```
Δy_t = c + φ₁*Δy_{t-1} + ε_t
```

## Implementation Notes

### Differencing

When d > 0, the model:
1. Applies differencing d times to the original series
2. Works with the differenced series
3. Returns residuals corresponding to the differenced series
4. The number of residuals is reduced by d (due to differencing loss)

### Initial Conditions

For the first observations where historical values are not available:
- Historical observations are initialized to zero
- Historical residuals are initialized to zero
- This follows standard time series practice

### State Management

The `ArimaState` class maintains a sliding window of:
- The most recent p observations (for AR)
- The most recent q residuals (for MA)

After computing each residual, the state is updated for the next iteration.

## Testing

Comprehensive unit tests are provided in `tests/unit/test_models_arima.cpp`:
- White noise (ARIMA(0,0,0))
- AR(1), AR(2) processes
- MA(1) processes
- ARMA(1,1) processes
- Random walk with differencing (ARIMA(0,1,0))
- AR(1) with differencing (ARIMA(1,1,0))

All tests verify that residuals match the true innovations used to generate synthetic data.

## Example Program

A complete example demonstrating residual computation is available:
- `examples/example_arima_residuals.cpp`

Run with:
```bash
./build/examples/example_arima_residuals
```

## References

- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
