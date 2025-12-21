#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace ag::stats {

/**
 * @brief Calculate the autocorrelation function (ACF) for a data sequence.
 *
 * The ACF measures the correlation between a time series and lagged versions of itself.
 * ACF(k) = Cov(X_t, X_{t-k}) / Var(X_t)
 *
 * This is useful for:
 * - Identifying patterns and seasonality in time series
 * - ARIMA model order selection (identifying MA order q)
 * - Diagnostic checking of residuals
 *
 * @param data Span of time series data
 * @param max_lag Maximum lag to compute ACF for (must be less than data.size())
 * @return Vector of ACF values from lag 0 to max_lag (ACF[0] is always 1.0)
 * @throws std::invalid_argument if data is too short or max_lag is invalid
 */
[[nodiscard]] std::vector<double> acf(std::span<const double> data, std::size_t max_lag);

/**
 * @brief Calculate a single ACF value at a specific lag.
 *
 * @param data Span of time series data
 * @param lag The lag at which to compute ACF
 * @return The ACF value at the specified lag
 * @throws std::invalid_argument if data is too short or lag is invalid
 */
[[nodiscard]] double acf_at_lag(std::span<const double> data, std::size_t lag);

// Convenience overloads for std::vector
[[nodiscard]] inline std::vector<double> acf(const std::vector<double>& data, std::size_t max_lag) {
    return acf(std::span<const double>(data), max_lag);
}

[[nodiscard]] inline double acf_at_lag(const std::vector<double>& data, std::size_t lag) {
    return acf_at_lag(std::span<const double>(data), lag);
}

}  // namespace ag::stats
