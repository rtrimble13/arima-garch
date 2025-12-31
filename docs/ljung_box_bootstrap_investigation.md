# Ljung-Box Bootstrap Investigation - Final Report

## Problem Statement
A user reported that when simulating an ARIMA-GARCH process with Student-t innovations and then fitting a model with the same configuration, the Ljung-Box bootstrap method incorrectly indicated significant autocorrelation in residuals.

## Investigation Summary

### Methodology
1. Reviewed all relevant implementations (ACF, Ljung-Box statistic, bootstrap algorithm, residual computation)
2. Created comprehensive test cases covering various model specifications
3. Analyzed parameter estimation accuracy in problematic cases
4. Compared bootstrap vs asymptotic test results

### Key Findings

#### 1. Bootstrap Implementation is Mathematically Correct ✓
- **ACF computation**: Uses standard formula with variance from all n observations
- **Ljung-Box Q statistic**: Correct formula Q = n(n+2) * Σ(ρ²ₖ/(n-k)) for k=1 to h
- **Bootstrap algorithm**: Properly centers and resamples residuals
- **All existing tests pass**: 17/17 bootstrap tests, 4/4 Student-t tests, 14/14 diagnostics tests

#### 2. Root Cause: ARMA Model Identification Issues
The reported issue stems from **parameter estimation problems**, not a bootstrap bug:

**Example with ARIMA(2,0,2)-GARCH(1,1), n=150:**
- True AR[0] = 0.6 → Fitted = 0.942 (57% error)
- True MA[0] = 0.3 → Fitted = -0.328 (sign flip!)
- Result: Q-statistic = 65.83, p-value ≈ 0 (both asymptotic and bootstrap)

**Why This Happens:**
- ARMA models with both p,q ≥ 2 suffer from identification problems
- Heavy-tailed distributions (Student-t with low df) exacerbate estimation difficulties
- Small-to-medium sample sizes provide insufficient information
- Poor parameter recovery leads to genuinely autocorrelated residuals

#### 3. The Tests Are Working Correctly
When parameter estimation fails, residuals **will** show autocorrelation. The Ljung-Box test (both bootstrap and asymptotic) correctly detects this. This is **expected behavior**, not a test failure.

### What Was Changed

#### Documentation Improvements (`include/ag/stats/Bootstrap.hpp`)
Clarified the DOF parameter in `ljung_box_test_bootstrap`:
- **Before**: Documentation suggested DOF adjusts the test for estimated parameters
- **After**: Clear explanation that DOF is for reporting only; bootstrap doesn't use chi-squared distribution so doesn't need DOF adjustment
- **Added**: Warning that model misspecification causes legitimate test rejections

#### New Test Suite (`tests/unit/test_ljung_box_model_misspecification.cpp`)
Added 2 tests documenting expected behavior:
1. **`ljung_box_detects_poor_arma_fit`**: Shows that when ARMA(2,2) parameters are poorly estimated, both tests correctly detect autocorrelation
2. **`ljung_box_works_with_well_estimated_arma`**: Verifies that with well-estimated ARMA(1,1), residuals pass the test

### Recommendations for Users

#### When Ljung-Box Test Rejects (Low P-Value):
1. **Check parameter convergence**: Did the optimization converge?
2. **Examine parameter estimates**: Are they reasonable? Compare to simpler models
3. **Try simpler specifications**: ARMA(1,1) or ARMA(2,1) often estimate better than ARMA(2,2)
4. **Increase sample size**: Complex models need more data
5. **Consider alternative specifications**: Maybe a different model structure fits better

#### Model Selection Guidelines:
- **Prefer parsimony**: Use simplest model that fits adequately
- **ARMA(1,1)**: Usually estimates well, good default
- **ARMA(2,2)**: Prone to identification issues, use cautiously
- **Student-t with df < 5**: Can make estimation harder
- **Minimum sample size**: At least 100-200 observations for complex specifications

### Conclusion

The Ljung-Box bootstrap implementation is **working correctly**. The user's concern about test rejections is actually indicative of a real problem: **poor parameter estimation with complex ARMA specifications**. This is a well-known challenge in time series econometrics, not a software bug.

The bootstrap method provides valid inference for any innovation distribution (including Student-t), but it cannot overcome fundamental identification problems in the model specification itself.

### References

For more information on ARMA identification issues:
- Box, G.E.P., Jenkins, G.M., & Reinsel, G.C. (2015). Time Series Analysis: Forecasting and Control.
- Hamilton, J.D. (1994). Time Series Analysis. Chapter 5 on identification and estimation.

---

**Status**: Investigation complete. No bugs found. Documentation improved. Additional tests added.
