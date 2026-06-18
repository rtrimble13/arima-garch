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

/**
 * @brief Streaming d-th order differencing with the inverse (integration).
 *
 * Maintains the last value of each lower-order difference so that a series of
 * raw levels can be differenced one observation at a time, and a series of
 * d-th order differences can be integrated back to levels one step at a time.
 * This is the single source of truth for the differencing/integration the
 * composite ARIMA-GARCH model and forecaster need for d > 0.
 *
 * Forward (difference): the first @c d observations only prime the pipeline —
 * a d-th order difference is not defined until @c d+1 raw values have been
 * seen, so difference() returns false for those. Afterwards it returns the
 * d-th difference w_t = Δ^d y_t.
 *
 * Inverse (integrate): reconstructs a level from a d-th order difference using
 * the current anchors, advancing them. A freshly constructed differencer has
 * zero anchors, which integrates a difference series under zero initial
 * conditions (a plain d-fold cumulative sum). The forecaster instead copies
 * the model's primed differencer so integration continues from the real
 * terminal levels.
 *
 * For d == 0 both operations are the identity.
 */
class StreamingDifferencer {
public:
    /**
     * @brief Construct a differencer of the given order.
     * @param d Differencing order (must be >= 0)
     * @throws std::invalid_argument if d < 0
     */
    explicit StreamingDifferencer(int d);

    /**
     * @brief Feed one raw level and obtain its d-th order difference.
     * @param level The new raw observation
     * @param[out] differenced Set to Δ^d of the level when available
     * @return true if a d-th difference was produced, false while priming
     */
    bool difference(double level, double& differenced);

    /**
     * @brief Integrate a d-th order difference back to a level, advancing anchors.
     * @param differenced A d-th order difference (e.g. a forecast on the differenced scale)
     * @return The reconstructed level
     */
    double integrate(double differenced);

    /**
     * @brief Whether enough observations have been seen to produce differences.
     */
    [[nodiscard]] bool primed() const noexcept;

    /**
     * @brief The differencing order.
     */
    [[nodiscard]] int order() const noexcept { return d_; }

private:
    int d_;                     // Differencing order
    std::vector<double> last_;  // last_[k] = most recent k-th order difference (k = 0..d-1)
    std::size_t count_ = 0;     // Number of raw observations seen by difference()
};

}  // namespace ag::util
