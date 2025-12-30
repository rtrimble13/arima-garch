#pragma once

#include "ag/stats/ADF.hpp"
#include "ag/stats/LjungBox.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace ag::stats {

/**
 * @brief Perform bootstrap Ljung-Box test for residual autocorrelation.
 *
 * This function implements a residual bootstrap method to compute empirical
 * p-values for the Ljung-Box test. Unlike the asymptotic test which assumes
 * chi-squared distribution (derived under normality), the bootstrap method
 * works correctly for any innovation distribution, including Student-t.
 *
 * Algorithm:
 * 1. Center the residuals: e_t = residuals - mean(residuals)
 * 2. For b = 1 to n_bootstrap:
 *    a. Resample centered residuals with replacement: e*_t
 *    b. Compute Q* statistic on resampled residuals
 *    c. Store Q*
 * 3. P-value = proportion of Q* >= Q_observed
 *
 * The bootstrap p-value is valid under the null hypothesis of no autocorrelation,
 * regardless of the innovation distribution. This is particularly important when
 * innovations follow a Student-t distribution with low degrees of freedom.
 *
 * @param residuals Span of residual values from a fitted model
 * @param lags Number of lags to test (must be less than residuals.size())
 * @param dof Degrees of freedom for the test (default: 0, meaning dof = lags)
 * @param n_bootstrap Number of bootstrap replications (default: 1000)
 * @param seed Random seed for reproducibility (default: 42)
 * @return LjungBoxResult containing the statistic, bootstrap p-value, lags, and dof
 * @throws std::invalid_argument if residuals is too short or parameters are invalid
 *
 * @note The bootstrap method is computationally intensive. For Normal innovations,
 *       the asymptotic test may be preferred for speed. However, for Student-t or
 *       other heavy-tailed distributions, the bootstrap provides more accurate p-values.
 *
 * @note The degrees of freedom parameter allows specifying the number of estimated
 *       parameters in the model, typically dof = lags - num_estimated_params.
 *       If dof = 0 (default), it is set equal to lags.
 */
[[nodiscard]] LjungBoxResult ljung_box_test_bootstrap(std::span<const double> residuals,
                                                      std::size_t lags, std::size_t dof = 0,
                                                      std::size_t n_bootstrap = 1000,
                                                      unsigned int seed = 42);

/**
 * @brief Perform bootstrap ADF test for unit root (sieve bootstrap under null).
 *
 * This function implements a sieve bootstrap method for the ADF test that correctly
 * imposes the unit root null hypothesis. This is essential for valid inference when
 * the innovation distribution is non-normal, such as Student-t with low degrees of
 * freedom.
 *
 * Algorithm (Sieve Bootstrap under Unit Root Null):
 * 1. Take first differences: Δy_t = y_t - y_{t-1}
 * 2. Fit AR(p) model to differences: Δy_t = φ̂₁Δy_{t-1} + ... + φ̂ₚΔy_{t-p} + ê_t
 * 3. Center residuals: ẽ_t = ê_t - mean(ê_t)
 * 4. For b = 1 to n_bootstrap:
 *    a. Resample centered residuals with replacement: ẽ*_t
 *    b. Generate differences: Δy*_t = φ̂₁Δy*_{t-1} + ... + φ̂ₚΔy*_{t-p} + ẽ*_t
 *    c. Integrate to get levels (imposing unit root): y*_t = y*_{t-1} + Δy*_t
 *    d. Compute ADF statistic τ* on y*_t
 *    e. Store τ*
 * 5. P-value = proportion of τ* <= τ_observed (more negative = more evidence against H₀)
 *
 * The key difference from naive bootstrap is that bootstrap samples are generated
 * under the null hypothesis of a unit root by integrating the AR-generated differences.
 * This ensures that the bootstrap distribution correctly represents the null hypothesis,
 * providing valid critical values and p-values regardless of the innovation distribution.
 *
 * References:
 * - Chang, Y., & Park, J. Y. (2003). "A sieve bootstrap for the test of a unit root."
 *   Journal of Time Series Analysis, 24(4), 379-400.
 * - Palm, F. C., Smeekes, S., & Urbain, J. P. (2008). "Bootstrap unit-root tests:
 *   comparison and extensions." Journal of Time Series Analysis, 29(1), 371-401.
 *
 * @param data Span of time series data
 * @param lags Number of lagged differences to include in ADF regression
 * @param regression_form Type of deterministic components to include
 * @param n_bootstrap Number of bootstrap replications (default: 1000)
 * @param seed Random seed for reproducibility (default: 42)
 * @return ADFResult containing the statistic, bootstrap p-value, and critical values
 * @throws std::invalid_argument if data is too short or parameters are invalid
 *
 * @note The critical values in the result are computed from the bootstrap distribution
 *       (empirical quantiles) rather than using MacKinnon's tables.
 *
 * @note For accurate results, n_bootstrap should be at least 1000. Larger values
 *       (e.g., 5000 or 10000) provide more stable p-values but take longer to compute.
 *
 * @note This implementation correctly imposes the unit root null by fitting the AR model
 *       to differences and then integrating. Previous implementations that fit AR to levels
 *       do not properly test the unit root hypothesis.
 */
[[nodiscard]] ADFResult adf_test_bootstrap(std::span<const double> data, std::size_t lags,
                                           ADFRegressionForm regression_form,
                                           std::size_t n_bootstrap = 1000, unsigned int seed = 42);

// Convenience overloads for std::vector
[[nodiscard]] inline LjungBoxResult ljung_box_test_bootstrap(const std::vector<double>& residuals,
                                                             std::size_t lags, std::size_t dof = 0,
                                                             std::size_t n_bootstrap = 1000,
                                                             unsigned int seed = 42) {
    return ljung_box_test_bootstrap(std::span<const double>(residuals), lags, dof, n_bootstrap,
                                    seed);
}

[[nodiscard]] inline ADFResult adf_test_bootstrap(const std::vector<double>& data, std::size_t lags,
                                                  ADFRegressionForm regression_form,
                                                  std::size_t n_bootstrap = 1000,
                                                  unsigned int seed = 42) {
    return adf_test_bootstrap(std::span<const double>(data), lags, regression_form, n_bootstrap,
                              seed);
}

}  // namespace ag::stats
