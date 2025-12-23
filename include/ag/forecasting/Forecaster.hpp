#pragma once

#include "ag/models/composite/ArimaGarchModel.hpp"

#include <vector>

namespace ag::forecasting {

/**
 * @brief Result of a forecast operation over a horizon.
 *
 * Contains the iterated mean forecasts and variance forecasts for each step
 * in the forecast horizon.
 */
struct ForecastResult {
    std::vector<double> mean_forecasts;      // μ̂_{t+1}, μ̂_{t+2}, ..., μ̂_{t+h}
    std::vector<double> variance_forecasts;  // ĥ_{t+1}, ĥ_{t+2}, ..., ĥ_{t+h}

    /**
     * @brief Construct a ForecastResult with specified horizon.
     * @param horizon Number of steps to forecast
     */
    explicit ForecastResult(int horizon)
        : mean_forecasts(horizon, 0.0), variance_forecasts(horizon, 0.0) {}
};

/**
 * @brief Forecaster for ARIMA-GARCH models.
 *
 * Forecaster implements iterated multi-step ahead forecasting for ARIMA-GARCH models.
 * It produces forecasts for both the conditional mean (via ARIMA) and conditional
 * variance (via GARCH) over a specified horizon.
 *
 * For the mean forecast (ARIMA component):
 * - Uses iterated approach: each forecast becomes input for next step
 * - μ̂_{t+h} = c + Σφᵢ*ŷ_{t+h-i} + Σθⱼ*0  (future errors are zero in expectation)
 *
 * For the variance forecast (GARCH component):
 * - Also uses iterated approach with expected values
 * - ĥ_{t+h} = ω + Σαᵢ*E[ε²_{t+h-i}] + Σβⱼ*ĥ_{t+h-j}
 * - E[ε²_{t+h-i}] = ĥ_{t+h-i} for future steps (variance of forecast error)
 * - For stationary GARCH, variance converges to unconditional variance as h → ∞
 */
class Forecaster {
public:
    /**
     * @brief Construct a Forecaster with a fitted ARIMA-GARCH model.
     *
     * The model should be fully fitted and have its state initialized with
     * historical data. The forecaster will use the current state of the model
     * (most recent observations and residuals) as the starting point for forecasts.
     *
     * @param model Reference to a fitted ARIMA-GARCH model
     */
    explicit Forecaster(const ag::models::composite::ArimaGarchModel& model);

    /**
     * @brief Generate forecasts for a specified horizon.
     *
     * Produces h-step ahead forecasts for both the conditional mean and
     * conditional variance using an iterated approach.
     *
     * The method:
     * 1. Iterates forward h steps from the current model state
     * 2. At each step, computes the expected mean using ARIMA recursion
     * 3. At each step, computes the expected variance using GARCH recursion
     * 4. Uses forecasted values as inputs for subsequent steps
     *
     * @param horizon Number of steps ahead to forecast (must be > 0)
     * @return ForecastResult containing mean and variance forecasts
     * @throws std::invalid_argument if horizon <= 0
     */
    [[nodiscard]] ForecastResult forecast(int horizon) const;

private:
    const ag::models::composite::ArimaGarchModel& model_;

    /**
     * @brief Compute one-step ahead mean forecast.
     *
     * Uses ARIMA model parameters and historical observations/residuals to
     * compute the expected value for the next time step.
     *
     * @param obs_history Historical observations (most recent last)
     * @param res_history Historical residuals (most recent last)
     * @return One-step ahead mean forecast
     */
    [[nodiscard]] double forecastMeanOneStep(const std::vector<double>& obs_history,
                                             const std::vector<double>& res_history) const;

    /**
     * @brief Compute one-step ahead variance forecast.
     *
     * Uses GARCH model parameters and historical variances/squared residuals to
     * compute the expected conditional variance for the next time step.
     *
     * @param var_history Historical conditional variances (most recent last)
     * @param sq_res_history Historical squared residuals (most recent last)
     * @return One-step ahead variance forecast
     */
    [[nodiscard]] double forecastVarianceOneStep(const std::vector<double>& var_history,
                                                 const std::vector<double>& sq_res_history) const;
};

}  // namespace ag::forecasting
