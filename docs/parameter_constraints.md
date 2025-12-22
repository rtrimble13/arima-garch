# Parameter Constraints and Transformations

## Overview

The `ArimaGarchTransform` class provides parameter transformations for ARIMA-GARCH model estimation. It maps between unconstrained optimizer parameters (theta) and constrained GARCH parameters that satisfy all necessary constraints.

## GARCH Parameter Constraints

GARCH models require parameters to satisfy several constraints:

1. **Intercept Constraint**: ω > 0 (omega must be positive)
2. **Non-negativity**: αᵢ ≥ 0, βⱼ ≥ 0 (ARCH and GARCH coefficients must be non-negative)
3. **Stationarity**: Σαᵢ + Σβⱼ < 1 (sum of coefficients must be less than 1 for covariance stationarity)

## Transformation Strategy

### Forward Transform (Unconstrained → Constrained)

The `toConstrained` method transforms unconstrained parameters to constrained GARCH parameters:

```cpp
ParameterVector toConstrained(const ParameterVector& theta, int p, int q);
```

**Transformation Details:**

1. **Omega (ω)**: 
   - Transform: ω = exp(θ₀)
   - Ensures: ω > 0 for any θ₀ ∈ ℝ

2. **ARCH and GARCH Coefficients (α, β)**:
   - First apply exponential: exp(θᵢ) to ensure positivity
   - Then scale using logistic-like transform to ensure stationarity:
   - αᵢ, βⱼ = MAX_PERSISTENCE × exp(θᵢ) / (1 + Σexp(θₖ))
   - This ensures: Σαᵢ + Σβⱼ < MAX_PERSISTENCE = 0.999

**Example:**
```cpp
using ag::estimation::ArimaGarchTransform;
using ag::estimation::ParameterVector;

// Unconstrained parameters from optimizer
ParameterVector theta(3, 0.0);
theta[0] = -4.6;  // omega (unconstrained)
theta[1] = -2.3;  // alpha (unconstrained)
theta[2] = 2.1;   // beta (unconstrained)

// Transform to constrained parameters
ParameterVector params = ArimaGarchTransform::toConstrained(theta, 1, 1);
// params[0] = exp(-4.6) ≈ 0.01 (omega)
// params[1] ≈ 0.01 (alpha)
// params[2] ≈ 0.88 (beta)
// Sum: 0.89 < 1 ✓
```

### Inverse Transform (Constrained → Unconstrained)

The `toUnconstrained` method performs the inverse transformation:

```cpp
ParameterVector toUnconstrained(const ParameterVector& params, int p, int q);
```

This is useful for:
- Initializing optimizers with known valid parameters
- Converting estimated parameters back to optimizer space
- Parameter persistence and serialization

**Example:**
```cpp
// Valid GARCH(1,1) parameters
ParameterVector params(3, 0.0);
params[0] = 0.01;  // omega
params[1] = 0.1;   // alpha
params[2] = 0.85;  // beta

// Convert to unconstrained space
ParameterVector theta = ArimaGarchTransform::toUnconstrained(params, 1, 1);
// theta[0] = log(0.01) ≈ -4.6
// theta values can be used as initial values for optimization
```

### Constraint Validation

The `validateConstraints` method checks if parameters satisfy all GARCH constraints:

```cpp
bool validateConstraints(const ParameterVector& params, int p, int q) noexcept;
```

**Checks performed:**
- params.size() == 1 + p + q
- params[0] > 0 (omega positive)
- params[1:p] ≥ 0 (alpha non-negative)
- params[p+1:p+q] ≥ 0 (beta non-negative)
- Σparams[1:p] + Σparams[p+1:p+q] < 1 (stationarity)

**Example:**
```cpp
ParameterVector params(3, 0.0);
params[0] = 0.01;
params[1] = 0.1;
params[2] = 0.8;

if (ArimaGarchTransform::validateConstraints(params, 1, 1)) {
    std::cout << "Parameters are valid!" << std::endl;
}
```

## Usage in Optimization

The typical workflow for constrained optimization:

```cpp
// 1. Initialize with unconstrained parameters
ParameterVector theta_init(3, 0.0);  // or load from somewhere

// 2. During optimization, convert to constrained for likelihood evaluation
auto objective_function = [&](const ParameterVector& theta) {
    ParameterVector params = ArimaGarchTransform::toConstrained(theta, 1, 1);
    // Compute log-likelihood with params
    return compute_log_likelihood(params, data);
};

// 3. Run unconstrained optimizer on theta space
ParameterVector theta_optimal = optimizer.minimize(objective_function, theta_init);

// 4. Transform final result to constrained space
ParameterVector params_optimal = ArimaGarchTransform::toConstrained(theta_optimal, 1, 1);

// 5. Validate (should always be true if transform is used correctly)
assert(ArimaGarchTransform::validateConstraints(params_optimal, 1, 1));
```

## GARCH Orders

The transform supports any GARCH(p,q) specification where p ≥ 1 and q ≥ 1:

### GARCH(1,1)
```cpp
// theta: [ω_unconstrained, α_unconstrained, β_unconstrained]
// params: [ω, α, β]
ParameterVector theta(3);  // size = 1 + 1 + 1
auto params = ArimaGarchTransform::toConstrained(theta, 1, 1);
```

### GARCH(2,2)
```cpp
// theta: [ω_unconstrained, α₁_unconstrained, α₂_unconstrained, 
//         β₁_unconstrained, β₂_unconstrained]
// params: [ω, α₁, α₂, β₁, β₂]
ParameterVector theta(5);  // size = 1 + 2 + 2
auto params = ArimaGarchTransform::toConstrained(theta, 2, 2);
```

### GARCH(p,q) - General
```cpp
int p = 3;  // ARCH order
int q = 2;  // GARCH order
ParameterVector theta(1 + p + q);  // size = 1 + 3 + 2 = 6
auto params = ArimaGarchTransform::toConstrained(theta, p, q);
// params: [ω, α₁, α₂, α₃, β₁, β₂]
```

## Numerical Stability

The implementation is designed for numerical stability:

- **Extreme theta values**: The transform handles theta values from -100 to +100
- **Underflow protection**: Uses MAX_PERSISTENCE = 0.999 to avoid sum = 1.0 exactly
- **Overflow protection**: Exponential transforms are numerically stable for typical ranges
- **Epsilon protection**: Uses EPSILON = 1e-8 to avoid division by zero

## Testing

The implementation includes comprehensive unit tests:

- **21 test cases** covering all functionality
- **Random testing**: 200+ random theta inputs verified
- **Extreme values**: Testing with theta ∈ [-100, 100]
- **Multiple orders**: GARCH(1,1), (2,2), (3,2) tested
- **Round-trip accuracy**: Forward-inverse transforms verified
- **Error handling**: Invalid inputs properly rejected

Run tests with:
```bash
cmake --build build
./build/tests/test_estimation_constraints
```

## Example Program

A complete working example is provided in `examples/example_constraints.cpp`:

```bash
./build/examples/example_constraints
```

This demonstrates:
- Basic GARCH(1,1) transformation
- Inverse transformation and round-trip accuracy
- Random parameter testing
- GARCH(2,2) transformation
- Constraint verification

## References

- Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity"
- Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns"
- Engle, R.F. (2001). "GARCH 101: The Use of ARCH/GARCH Models in Applied Econometrics"

## See Also

- `include/ag/estimation/Constraints.hpp` - Header file with full API documentation
- `src/estimation/Constraints.cpp` - Implementation details
- `tests/unit/test_estimation_constraints.cpp` - Comprehensive test suite
- `examples/example_constraints.cpp` - Working example code
