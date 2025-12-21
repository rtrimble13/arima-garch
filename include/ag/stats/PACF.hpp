#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace ag::stats {

/**
 * @brief Calculate the partial autocorrelation function (PACF) for a data sequence.
 *
 * The PACF measures the correlation between observations at lag k, with the linear
 * dependence on intermediate lags removed. It's computed using the Durbin-Levinson
 * recursion algorithm.
 *
 * This is useful for:
 * - ARIMA model order selection (identifying AR order p)
 * - Understanding direct relationships between observations at different lags
 * - Diagnostic checking of time series models
 *
 * @param data Span of time series data
 * @param max_lag Maximum lag to compute PACF for (must be less than data.size())
 * @return Vector of PACF values from lag 1 to max_lag (PACF is only defined for lag >= 1)
 * @throws std::invalid_argument if data is too short or max_lag is invalid
 */
[[nodiscard]] std::vector<double> pacf(std::span<const double> data, std::size_t max_lag);

/**
 * @brief Calculate a single PACF value at a specific lag.
 *
 * @param data Span of time series data
 * @param lag The lag at which to compute PACF (must be >= 1)
 * @return The PACF value at the specified lag
 * @throws std::invalid_argument if data is too short or lag is invalid
 */
[[nodiscard]] double pacf_at_lag(std::span<const double> data, std::size_t lag);

// Convenience overloads for std::vector
[[nodiscard]] inline std::vector<double> pacf(const std::vector<double>& data,
                                               std::size_t max_lag) {
    return pacf(std::span<const double>(data), max_lag);
}

[[nodiscard]] inline double pacf_at_lag(const std::vector<double>& data, std::size_t lag) {
    return pacf_at_lag(std::span<const double>(data), lag);
}

}  // namespace ag::stats
