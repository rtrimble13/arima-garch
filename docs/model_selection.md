# Model Selection

This document explains the automatic model selection capabilities of the arima-garch library, including selection criteria, candidate generation, and best practices.

## Overview

Automatic model selection helps you find the optimal ARIMA-GARCH specification for your data without manually testing every combination. The library provides:

- **Multiple selection criteria**: BIC, AIC, AICc, and Cross-Validation
- **Flexible search spaces**: Define ranges for ARIMA(p,d,q) and GARCH(p,q) orders
- **Robust fitting**: Automatically handles models that fail to converge
- **Comprehensive reporting**: Detailed summaries of selection results

## Selection Criteria

### 1. BIC (Bayesian Information Criterion)

**Formula:** BIC = -2 × log-likelihood + k × log(n)

**Characteristics:**
- **Penalty:** Stronger penalty for model complexity
- **Behavior:** Tends to select more parsimonious (simpler) models
- **Speed:** Fast (requires only likelihood evaluation)
- **Best for:** When simplicity and interpretability are priorities

**When to use:**
- Large datasets (where overfitting is a concern)
- Need for model interpretability
- Fast selection required
- Prefer simpler models over marginal fit improvements

**Example:**
```bash
ag select -d data.csv --max-p 2 --max-q 2 -c BIC -o model_bic.json
```

### 2. AIC (Akaike Information Criterion)

**Formula:** AIC = -2 × log-likelihood + 2k

**Characteristics:**
- **Penalty:** Moderate penalty for model complexity
- **Behavior:** Balances fit and complexity
- **Speed:** Fast (requires only likelihood evaluation)
- **Best for:** General-purpose model selection

**When to use:**
- Medium-sized datasets
- Balance between fit quality and complexity
- Standard model selection scenarios
- When BIC seems too restrictive

**Example:**
```bash
ag select -d data.csv --max-p 2 --max-q 2 -c AIC -o model_aic.json
```

### 3. AICc (Corrected AIC)

**Formula:** AICc = AIC + (2k² + 2k) / (n - k - 1)

**Characteristics:**
- **Penalty:** AIC with finite-sample correction
- **Behavior:** More conservative than AIC for small samples
- **Speed:** Fast (requires only likelihood evaluation)
- **Best for:** Small to medium-sized datasets

**When to use:**
- Small sample sizes (n < 100)
- Want AIC-like behavior with better small-sample properties
- Concerned about overfitting in limited data
- As a more conservative alternative to AIC

**Example:**
```bash
ag select -d data.csv --max-p 2 --max-q 2 -c AICc -o model_aicc.json
```

### 4. CV (Cross-Validation)

**Method:** Rolling origin cross-validation with 1-step-ahead forecasts

**Characteristics:**
- **Evaluation:** Direct measure of out-of-sample forecast performance
- **Metric:** Mean Squared Error (MSE) of 1-step-ahead forecasts
- **Speed:** Slower (requires multiple model fits per candidate)
- **Best for:** When forecast accuracy is the primary goal

**How it works:**
1. Split data into training and validation windows
2. Fit model on training window
3. Generate 1-step-ahead forecast
4. Compute squared error
5. Roll window forward and repeat
6. Average squared errors across all folds

**When to use:**
- Forecast accuracy is the primary objective
- Sufficient computational time available
- Want direct evidence of forecast performance
- Don't want to rely on information criteria assumptions

**Example:**
```bash
ag select -d data.csv --max-p 2 --max-q 2 -c CV -o model_cv.json
```

**Note:** Cross-validation can be significantly slower than information criteria, especially with large candidate sets or long time series.

## Selection Criteria Comparison

| Criterion | Speed | Sample Size | Model Complexity | Forecast Focus |
|-----------|-------|-------------|------------------|----------------|
| **BIC**   | Fast  | Large       | Simpler          | No             |
| **AIC**   | Fast  | Medium-Large| Balanced         | No             |
| **AICc**  | Fast  | Small       | Conservative     | No             |
| **CV**    | Slow  | Any         | Performance-based| Yes            |

**General Recommendation:**
- Start with **BIC** for fast, robust selection
- Use **AICc** if you have < 100 observations
- Use **CV** when forecast accuracy is critical and you have time

## Candidate Grid Generation

### Basic Configuration

The candidate grid defines the search space of model specifications to evaluate.

```cpp
// C++ API
ag::selection::CandidateGridConfig config(
    2,  // max_p: Maximum ARIMA AR order
    1,  // max_d: Maximum differencing order
    2,  // max_q: Maximum ARIMA MA order
    1,  // max_p_garch: Maximum GARCH order
    1   // max_q_garch: Maximum ARCH order
);

ag::selection::CandidateGrid grid(config);
auto candidates = grid.generate();
```

```bash
# CLI
ag select -d data.csv \
    --max-p 2 \
    --max-d 1 \
    --max-q 2 \
    --max-garch-p 1 \
    --max-garch-q 1 \
    -c BIC -o model.json
```

This generates all combinations:
- ARIMA: (0-2, 0-1, 0-2)
- GARCH: (1, 1)

**Number of candidates = (max_p + 1) × (max_d + 1) × (max_q + 1) × (max_p_garch) × (max_q_garch)**

For example: 3 × 2 × 3 × 1 × 1 = 18 candidates

### Grid Configuration Options

#### `restrict_d_to_01` (boolean, default: false)

When `true`, only considers d ∈ {0, 1} regardless of `max_d` setting.

**Rationale:**
- Most economic/financial time series need at most first differencing (d=1)
- d > 1 is rarely needed and can lead to overdifferencing
- Reduces candidate set size

```cpp
config.restrict_d_to_01 = true;
```

#### Avoiding ARIMA(0,0,0)

The grid automatically excludes ARIMA(0,0,0) as it represents no mean model, which is not meaningful with GARCH.

### Search Space Examples

#### Example 1: Conservative Search

Small, focused search for quick results:

```bash
ag select -d data.csv \
    --max-p 1 \
    --max-d 1 \
    --max-q 1 \
    --max-garch-p 1 \
    --max-garch-q 1 \
    -c BIC -o model.json
```

**Candidates:** 2 × 2 × 2 × 1 × 1 = 8 models
- Fast selection
- Covers most common specifications
- Good starting point

#### Example 2: Standard Search

Balanced search covering common model types:

```bash
ag select -d data.csv \
    --max-p 2 \
    --max-d 1 \
    --max-q 2 \
    --max-garch-p 1 \
    --max-garch-q 1 \
    -c BIC -o model.json
```

**Candidates:** 3 × 2 × 3 × 1 × 1 = 18 models
- Reasonable speed
- Good coverage of ARIMA space
- Standard GARCH(1,1)

#### Example 3: Extensive Search

Thorough search for best possible model:

```bash
ag select -d data.csv \
    --max-p 3 \
    --max-d 1 \
    --max-q 3 \
    --max-garch-p 2 \
    --max-garch-q 2 \
    -c AIC -o model.json
```

**Candidates:** 4 × 2 × 4 × 2 × 2 = 128 models
- Slower selection
- Comprehensive coverage
- May find better-fitting models

**Warning:** Large search spaces with CV can be very slow!

#### Example 4: GARCH Variants

Testing different GARCH specifications with fixed ARIMA:

```bash
# If you know your mean model (e.g., from prior analysis),
# you can search over GARCH specifications by fixing max-p=1, max-d=0, max-q=1
ag select -d data.csv \
    --max-p 1 \
    --max-d 0 \
    --max-q 1 \
    --max-garch-p 2 \
    --max-garch-q 2 \
    -c BIC -o model.json
```

## Selection Process

### Step-by-Step Algorithm

1. **Generate Candidates**
   - Create all combinations within specified ranges
   - Exclude invalid specifications (e.g., ARIMA(0,0,0))

2. **Fit Each Candidate**
   - Estimate parameters using maximum likelihood
   - Handle convergence failures gracefully
   - Store fit results for successful models

3. **Evaluate Criterion**
   - For BIC/AIC/AICc: Compute criterion from likelihood
   - For CV: Perform rolling cross-validation

4. **Select Best Model**
   - Choose model with lowest criterion value
   - Return best specification and parameters

5. **Report Results**
   - Number of candidates evaluated
   - Number of convergence failures
   - Best model specification
   - Fit statistics and diagnostics

### Handling Failed Models

Not all candidate models will converge successfully. Common causes:
- Model too complex for data
- Numerical instability
- Boundary constraints violated
- Insufficient data

**The selector handles failures automatically:**
- Failed models are excluded from consideration
- Selection continues with remaining candidates
- Summary reports the number of failures

**Example output:**
```
Candidates evaluated: 18
Candidates failed: 3
Best model: ARIMA(1,1,1)-GARCH(1,1)
```

This indicates 3 models failed to fit, and the best of the 15 successful models was selected.

## Using Model Selection

### CLI Usage

```bash
# Basic usage with defaults (BIC criterion)
ag select -d data.csv -o best_model.json

# Specify criterion explicitly
ag select -d data.csv -c AIC -o model_aic.json

# Custom search space
ag select -d data.csv \
    --max-p 3 \
    --max-q 3 \
    --max-garch-p 2 \
    --max-garch-q 2 \
    -c BIC -o large_search.json

# Cross-validation selection
ag select -d data.csv \
    --max-p 2 \
    --max-q 2 \
    -c CV -o model_cv.json
```

### Library API Usage

```cpp
#include "ag/selection/ModelSelector.hpp"
#include "ag/selection/CandidateGrid.hpp"

// Load data
std::vector<double> data = {/* your time series */};

// Create candidate grid
ag::selection::CandidateGridConfig config(2, 1, 2, 1, 1);
ag::selection::CandidateGrid grid(config);
auto candidates = grid.generate();

// Select using BIC
ag::selection::ModelSelector selector(ag::selection::SelectionCriterion::BIC);
auto result = selector.select(data.data(), data.size(), candidates, true);

if (result.has_value()) {
    std::cout << "Best model: ARIMA(" 
              << result->best_spec.arimaSpec.p << ","
              << result->best_spec.arimaSpec.d << ","
              << result->best_spec.arimaSpec.q << ")-GARCH("
              << result->best_spec.garchSpec.p << ","
              << result->best_spec.garchSpec.q << ")\n";
    
    std::cout << "BIC: " << result->criterion_value << "\n";
    std::cout << "Log-likelihood: " << result->log_likelihood << "\n";
}
```

### Interpreting Selection Results

The selection output includes:

1. **Best Model Specification**
   - ARIMA(p,d,q) orders
   - GARCH(p,q) orders

2. **Criterion Value**
   - Lower is better
   - Only comparable within same criterion type

3. **Fit Statistics**
   - Log-likelihood
   - Number of parameters
   - AIC, BIC values

4. **Parameter Estimates**
   - ARIMA coefficients (intercept, AR, MA)
   - GARCH parameters (omega, alpha, beta)

5. **Diagnostic Tests**
   - Ljung-Box tests (residuals and squared residuals)
   - Jarque-Bera normality test

## Best Practices

### 1. Start Simple

Begin with a small search space:
```bash
ag select -d data.csv --max-p 1 --max-q 1 -c BIC -o model1.json
```

If the selected model is at the boundary (e.g., p=1, q=1), expand the search:
```bash
ag select -d data.csv --max-p 2 --max-q 2 -c BIC -o model2.json
```

### 2. Check Diagnostics

Always verify the selected model with diagnostics:
```bash
ag select -d data.csv -c BIC -o model.json
ag diagnostics -m model.json -d data.csv
```

**Look for:**
- Ljung-Box p-values > 0.05 (no autocorrelation)
- Jarque-Bera p-value > 0.05 (normality)
- Reasonable parameter estimates

If diagnostics are poor, consider:
- Expanding search space
- Trying a different criterion
- Checking data quality

### 3. Consider Multiple Criteria

When uncertain, compare selections from different criteria:
```bash
ag select -d data.csv -c BIC -o model_bic.json
ag select -d data.csv -c AIC -o model_aic.json
ag select -d data.csv -c CV -o model_cv.json
```

If all criteria select similar models, you have strong evidence. If they differ significantly, investigate further.

### 4. Use CV for Forecasting

If your primary goal is forecasting accuracy:
```bash
ag select -d data.csv -c CV -o forecast_model.json
ag forecast -m forecast_model.json -n 20 -o forecasts.csv
```

### 5. Document Your Selection

Keep a record of:
- Data used for selection
- Search space parameters
- Criterion used
- Date of selection
- Diagnostic results

**Example log:**
```
Date: 2024-01-15
Data: market_returns_2023.csv (n=250)
Search: ARIMA(0-2, 0-1, 0-2), GARCH(1,1)
Criterion: BIC
Candidates: 18
Failed: 2
Selected: ARIMA(1,1,1)-GARCH(1,1)
BIC: 1234.56
Diagnostics: All tests passed (p > 0.05)
```

## Common Pitfalls

### 1. Overfitting with Small Data

**Problem:** Complex models selected despite small sample size

**Solution:**
- Use BIC or AICc (stronger penalties)
- Limit search space
- Require at least 100-200 observations

### 2. Ignoring Diagnostics

**Problem:** Selected model has poor diagnostics

**Solution:**
- Always run diagnostics after selection
- Don't blindly trust criterion values
- Consider alternative specifications if diagnostics fail

### 3. Excessive Search Space

**Problem:** Too many candidates, slow selection

**Solution:**
- Start with standard ranges (p,q ≤ 2)
- Expand only if needed
- Use BIC instead of CV for large searches

### 4. Not Checking Convergence Failures

**Problem:** Many models failing to converge

**Solution:**
- Check data quality (outliers, sufficient length)
- Reduce search space complexity
- Verify data is appropriate for ARIMA-GARCH

### 5. Comparing Criteria Values Directly

**Problem:** "My AIC model has a value of 500, BIC model has 520, so AIC is better"

**Solution:**
- AIC and BIC values are NOT directly comparable
- Each criterion has different penalties
- Compare models within the same criterion only

## Advanced Topics

### Custom Candidate Lists

For complete control, generate your own candidate list:

```cpp
std::vector<ag::models::ArimaGarchSpec> candidates;
candidates.push_back(ag::models::ArimaGarchSpec(1, 0, 1, 1, 1));
candidates.push_back(ag::models::ArimaGarchSpec(1, 1, 1, 1, 1));
candidates.push_back(ag::models::ArimaGarchSpec(2, 0, 2, 1, 1));

ag::selection::ModelSelector selector(ag::selection::SelectionCriterion::BIC);
auto result = selector.select(data.data(), data.size(), candidates, true);
```

### Parallel Selection (Future)

Currently, model selection is sequential. Future versions may support parallel evaluation of candidates for faster selection.

### Cross-Validation Configuration (Future)

Future versions may allow configuration of CV parameters:
- Number of folds
- Minimum training window size
- Multi-step-ahead forecasts
- Alternative error metrics (MAE, MAPE)

## Performance Considerations

### Speed Estimates (Approximate)

For a dataset with 1000 observations:

| Criterion | Candidates | Time     |
|-----------|-----------|----------|
| BIC       | 18        | ~5 sec   |
| BIC       | 128       | ~30 sec  |
| CV        | 18        | ~30 sec  |
| CV        | 128       | ~5 min   |

**Factors affecting speed:**
- Data size (larger → slower)
- Model complexity (higher orders → slower per model)
- Number of candidates (linear scaling)
- CV vs. information criteria (CV is ~5-10x slower)

### Optimization Tips

1. **Use BIC for initial exploration**
   - Fast feedback on search space
   - Identify promising regions

2. **Refine with CV if needed**
   - Use smaller, focused search space
   - Apply to top candidates from BIC selection

3. **Leverage caching** (future feature)
   - Reuse fits from previous selections
   - Incrementally expand search space

## Examples

### Example 1: Quick Selection

```bash
# Fast selection on small dataset
ag select -d small_data.csv --max-p 1 --max-q 1 -c BIC -o quick_model.json
ag diagnostics -m quick_model.json -d small_data.csv
ag forecast -m quick_model.json -n 10
```

### Example 2: Comprehensive Selection

```bash
# Thorough search with diagnostics
ag select -d large_data.csv \
    --max-p 3 --max-d 1 --max-q 3 \
    --max-garch-p 2 --max-garch-q 2 \
    -c AIC -o comprehensive.json

ag diagnostics -m comprehensive.json -d large_data.csv -o diagnostics.json
ag forecast -m comprehensive.json -n 20 -o forecasts.csv
```

### Example 3: Forecast-Optimized Selection

```bash
# CV selection for best forecast accuracy
ag select -d forecast_data.csv --max-p 2 --max-q 2 -c CV -o forecast_model.json
ag forecast -m forecast_model.json -n 30 -o forecasts.csv
```

### Example 4: Comparison Study

```bash
# Compare all criteria
for criterion in BIC AIC AICc; do
    ag select -d data.csv --max-p 2 --max-q 2 -c $criterion -o model_$criterion.json
    ag diagnostics -m model_$criterion.json -d data.csv -o diag_$criterion.json
done

# Manually compare results and choose best based on diagnostics and use case
```

## References

- [File Formats](file_formats.md) - Model JSON schema and CSV formats
- [CLI Documentation](cli.md) - Command-line interface details
- [Parameter Constraints](parameter_constraints.md) - Parameter validation rules
- [README](../README.md) - Library overview and quick start

## Further Reading

- Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and Multimodel Inference* (2nd ed.). Springer.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
