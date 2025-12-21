#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace ag::stats {

/**
 * @brief Result of the Ljung-Box test for residual autocorrelation.
 *
 * The Ljung-Box test is used to test whether residuals from a time series model
 * are independently distributed (white noise) or exhibit autocorrelation.
 */
struct LjungBoxResult {
    double statistic;  ///< The Q test statistic
    double p_value;    ///< P-value from chi-square distribution
    std::size_t lags;  ///< Number of lags tested
    std::size_t dof;   ///< Degrees of freedom (typically equal to lags)
};

/**
 * @brief Calculate the Ljung-Box Q statistic for testing residual autocorrelation.
 *
 * The Ljung-Box test statistic is computed as:
 * Q = n(n+2) * Σ(ρ²ₖ/(n-k)) for k=1 to h
 *
 * where:
 * - n is the sample size
 * - h is the number of lags tested
 * - ρₖ is the sample autocorrelation at lag k
 *
 * Under the null hypothesis (residuals are independently distributed),
 * Q follows a chi-square distribution with h degrees of freedom.
 *
 * @param residuals Span of residual values from a fitted model
 * @param lags Number of lags to test (must be less than residuals.size())
 * @return The Ljung-Box Q statistic
 * @throws std::invalid_argument if residuals is too short or lags is invalid
 */
[[nodiscard]] double ljung_box_statistic(std::span<const double> residuals, std::size_t lags);

/**
 * @brief Perform the Ljung-Box test for residual autocorrelation.
 *
 * Tests the null hypothesis that residuals are independently distributed (white noise)
 * versus the alternative that they exhibit autocorrelation up to lag h.
 *
 * The test computes:
 * 1. The Ljung-Box Q statistic
 * 2. The p-value from the chi-square distribution with h degrees of freedom
 *
 * Interpretation:
 * - High p-value (e.g., > 0.05): Fail to reject null hypothesis - residuals appear
 *   to be white noise (good model fit)
 * - Low p-value (e.g., < 0.05): Reject null hypothesis - residuals show significant
 *   autocorrelation (model may be inadequate)
 *
 * @param residuals Span of residual values from a fitted model
 * @param lags Number of lags to test (must be less than residuals.size())
 * @param dof Degrees of freedom (default = lags). Can be adjusted if parameters
 *            were estimated (typically dof = lags - number of estimated parameters)
 * @return LjungBoxResult containing the statistic, p-value, lags, and degrees of freedom
 * @throws std::invalid_argument if residuals is too short or parameters are invalid
 */
[[nodiscard]] LjungBoxResult ljung_box_test(std::span<const double> residuals, std::size_t lags,
                                            std::size_t dof = 0);

// Convenience overloads for std::vector
[[nodiscard]] inline double ljung_box_statistic(const std::vector<double>& residuals,
                                                std::size_t lags) {
    return ljung_box_statistic(std::span<const double>(residuals), lags);
}

[[nodiscard]] inline LjungBoxResult ljung_box_test(const std::vector<double>& residuals,
                                                   std::size_t lags, std::size_t dof = 0) {
    return ljung_box_test(std::span<const double>(residuals), lags, dof);
}

}  // namespace ag::stats
