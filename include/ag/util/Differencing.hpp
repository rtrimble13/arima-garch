#pragma once

#include <cstddef>
#include <vector>

namespace ag::util {

/**
 * @brief Apply d-th order discrete differencing to a contiguous data buffer.
 *
 * First-differencing computes y'_t = y_t - y_{t-1}; the operation is
 * applied d times in sequence, shortening the series by d observations.
 * Used in two places: ARIMA state initialization and parameter
 * initialization heuristics — both need to operate on the same
 * differenced view of the data.
 *
 * @param data Pointer to the input series (may be null only if size == 0).
 * @param size Number of observations in the input series.
 * @param d   Order of differencing (must be >= 0).
 * @return Differenced series of length max(0, size - d). For d == 0,
 *         returns a copy of the input series. If d exceeds size - 1,
 *         returns an empty vector rather than throwing.
 */
[[nodiscard]] std::vector<double> differenceSeries(const double* data, std::size_t size, int d);

}  // namespace ag::util
