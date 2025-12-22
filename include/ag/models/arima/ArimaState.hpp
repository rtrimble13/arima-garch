#pragma once

#include <cstddef>
#include <vector>

namespace ag::models::arima {

/**
 * @brief Maintains the state for ARIMA recursion computation.
 *
 * ArimaState stores the historical values and residuals needed to compute
 * the conditional mean and residuals for ARIMA(p,d,q) models using recursion.
 *
 * The state includes:
 * - Historical observations (for AR component)
 * - Historical residuals (for MA component)
 * - Differenced series (when d > 0)
 */
class ArimaState {
public:
    /**
     * @brief Construct an ARIMA state with specified orders.
     * @param p Order of the autoregressive component
     * @param d Degree of differencing
     * @param q Order of the moving average component
     */
    ArimaState(int p, int d, int q);

    /**
     * @brief Initialize the state with a time series.
     *
     * This prepares the state for residual computation by:
     * - Applying differencing if d > 0
     * - Initializing historical observation buffer
     * - Initializing historical residual buffer (to zero)
     *
     * @param data Pointer to time series data
     * @param size Number of observations in the time series
     */
    void initialize(const double* data, std::size_t size);

    /**
     * @brief Update state with a new observation and residual.
     *
     * This is called after computing each residual to maintain the
     * sliding window of historical values needed for the next iteration.
     *
     * @param observation The new observation value
     * @param residual The computed residual for this observation
     */
    void update(double observation, double residual);

    /**
     * @brief Get the historical observations for AR component.
     * @return Vector of the most recent p observations (oldest first)
     */
    [[nodiscard]] const std::vector<double>& getObservationHistory() const noexcept {
        return obs_history_;
    }

    /**
     * @brief Get the historical residuals for MA component.
     * @return Vector of the most recent q residuals (oldest first)
     */
    [[nodiscard]] const std::vector<double>& getResidualHistory() const noexcept {
        return residual_history_;
    }

    /**
     * @brief Get the differenced series (if d > 0).
     * @return Vector containing the differenced time series
     */
    [[nodiscard]] const std::vector<double>& getDifferencedSeries() const noexcept {
        return differenced_series_;
    }

    /**
     * @brief Check if the state has been initialized.
     * @return true if initialize() has been called, false otherwise
     */
    [[nodiscard]] bool isInitialized() const noexcept { return initialized_; }

    /**
     * @brief Get the number of observations that were lost due to differencing.
     * @return Number of observations lost (equals d)
     */
    [[nodiscard]] int getDifferencingLoss() const noexcept { return d_; }

private:
    int p_;  // AR order
    int d_;  // Differencing degree
    int q_;  // MA order

    bool initialized_;

    std::vector<double> obs_history_;         // Sliding window of p most recent observations
    std::vector<double> residual_history_;    // Sliding window of q most recent residuals
    std::vector<double> differenced_series_;  // Differenced time series (if d > 0)

    /**
     * @brief Apply differencing to a time series.
     * @param data Pointer to original time series data
     * @param size Number of observations
     * @return Vector containing the differenced series
     */
    std::vector<double> applyDifferencing(const double* data, std::size_t size) const;
};

}  // namespace ag::models::arima
