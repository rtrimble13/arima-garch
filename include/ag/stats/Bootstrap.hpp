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
 * @param n_bootstrap Number of bootstrap replications (default: 1000)
 * @param seed Random seed for reproducibility (default: 42)
 * @return LjungBoxResult containing the statistic, bootstrap p-value, lags, and dof
 * @throws std::invalid_argument if residuals is too short or parameters are invalid
 *
 * @note The bootstrap method is computationally intensive. For Normal innovations,
 *       the asymptotic test may be preferred for speed. However, for Student-t or
 *       other heavy-tailed distributions, the bootstrap provides more accurate p-values.
 *
 * @note The degrees of freedom (dof) in the result is set equal to lags for consistency
 *       with the asymptotic test, but the p-value is computed from the bootstrap
 *       distribution rather than the chi-squared distribution.
 */
[[nodiscard]] LjungBoxResult ljung_box_test_bootstrap(std::span<const double> residuals,
                                                      std::size_t lags,
                                                      std::size_t n_bootstrap = 1000,
                                                      unsigned int seed = 42);

/**
 * @brief Perform bootstrap ADF test for unit root (sieve bootstrap).
 *
 * This function implements a sieve bootstrap method for the ADF test to compute
 * empirical critical values and p-values. The sieve bootstrap is appropriate for
 * testing unit roots when the innovation distribution is non-normal, such as
 * Student-t with low degrees of freedom.
 *
 * Algorithm (Sieve Bootstrap):
 * 1. Fit AR(p) model to data, estimate φ̂ and residuals ê_t
 * 2. Center residuals: ẽ_t = ê_t - mean(ê_t)
 * 3. For b = 1 to n_bootstrap:
 *    a. Resample centered residuals with replacement: ẽ*_t
 *    b. Generate bootstrap series: y*_t = φ̂_1*y*_{t-1} + ... + ẽ*_t
 *    c. Compute ADF statistic τ* on y*_t
 *    d. Store τ*
 * 4. P-value = proportion of τ* <= τ_observed (for testing stationarity)
 *
 * The bootstrap approach provides valid inference when innovations are non-normal,
 * which is common in financial time series. For Student-t innovations, the tabulated
 * critical values from MacKinnon may be inaccurate, making bootstrap essential.
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
 * @note The sieve bootstrap is specifically designed for unit root testing and is
 *       different from the naive residual bootstrap. It preserves the dependence
 *       structure under the null hypothesis of a unit root.
 */
[[nodiscard]] ADFResult adf_test_bootstrap(std::span<const double> data, std::size_t lags,
                                           ADFRegressionForm regression_form,
                                           std::size_t n_bootstrap = 1000, unsigned int seed = 42);

// Convenience overloads for std::vector
[[nodiscard]] inline LjungBoxResult ljung_box_test_bootstrap(const std::vector<double>& residuals,
                                                             std::size_t lags,
                                                             std::size_t n_bootstrap = 1000,
                                                             unsigned int seed = 42) {
    return ljung_box_test_bootstrap(std::span<const double>(residuals), lags, n_bootstrap, seed);
}

[[nodiscard]] inline ADFResult adf_test_bootstrap(const std::vector<double>& data, std::size_t lags,
                                                  ADFRegressionForm regression_form,
                                                  std::size_t n_bootstrap = 1000,
                                                  unsigned int seed = 42) {
    return adf_test_bootstrap(std::span<const double>(data), lags, regression_form, n_bootstrap,
                              seed);
}

}  // namespace ag::stats
