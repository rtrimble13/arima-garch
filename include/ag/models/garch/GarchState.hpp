#pragma once

#include <cstddef>
#include <vector>

namespace ag::models::garch {

/**
 * @brief Maintains the state for GARCH recursion computation.
 *
 * GarchState stores the historical conditional variances and squared residuals
 * needed to compute the conditional variance h_t for GARCH(p,q) models using recursion.
 *
 * The state includes:
 * - Historical conditional variances h_{t-1}, ..., h_{t-p} (for GARCH component)
 * - Historical squared residuals ε²_{t-1}, ..., ε²_{t-q} (for ARCH component)
 */
class GarchState {
public:
    /**
     * @brief Construct a GARCH state with specified orders.
     * @param p Order of the GARCH component (lagged conditional variances)
     * @param q Order of the ARCH component (lagged squared residuals)
     */
    GarchState(int p, int q);

    /**
     * @brief Initialize the state with residuals from ARIMA model.
     *
     * This prepares the state for variance computation by:
     * - Computing initial conditional variance (h_0)
     * - Initializing historical variance buffer
     * - Initializing historical squared residual buffer
     *
     * @param residuals Pointer to residual series from ARIMA model
     * @param size Number of residuals
     * @param unconditional_variance Optional unconditional variance for initialization.
     *                                If not provided or <= 0, uses sample variance of residuals.
     */
    void initialize(const double* residuals, std::size_t size, double unconditional_variance = 0.0);

    /**
     * @brief Update state with new conditional variance and squared residual.
     *
     * This is called after computing each conditional variance to maintain the
     * sliding window of historical values needed for the next iteration.
     *
     * @param conditional_variance The new conditional variance h_t
     * @param squared_residual The squared residual ε²_t
     */
    void update(double conditional_variance, double squared_residual);

    /**
     * @brief Get the historical conditional variances for GARCH component.
     * @return Vector of the most recent p conditional variances (oldest first)
     */
    [[nodiscard]] const std::vector<double>& getVarianceHistory() const noexcept {
        return variance_history_;
    }

    /**
     * @brief Get the historical squared residuals for ARCH component.
     * @return Vector of the most recent q squared residuals (oldest first)
     */
    [[nodiscard]] const std::vector<double>& getSquaredResidualHistory() const noexcept {
        return squared_residual_history_;
    }

    /**
     * @brief Check if the state has been initialized.
     * @return true if initialize() has been called, false otherwise
     */
    [[nodiscard]] bool isInitialized() const noexcept { return initialized_; }

    /**
     * @brief Get the initial conditional variance (h_0).
     * @return The initial conditional variance used
     */
    [[nodiscard]] double getInitialVariance() const noexcept { return initial_variance_; }

private:
    int p_;  // GARCH order
    int q_;  // ARCH order

    bool initialized_;
    double initial_variance_;  // h_0

    std::vector<double> variance_history_;          // Sliding window of p conditional variances
    std::vector<double> squared_residual_history_;  // Sliding window of q squared residuals

    /**
     * @brief Compute sample variance of residuals.
     * @param residuals Pointer to residual series
     * @param size Number of residuals
     * @return Sample variance
     */
    [[nodiscard]] double computeSampleVariance(const double* residuals, std::size_t size) const;
};

}  // namespace ag::models::garch
