#include "ag/forecasting/Forecaster.hpp"

#include <algorithm>
#include <stdexcept>

namespace ag::forecasting {

namespace {
// Minimum variance threshold to guard against numerical issues
constexpr double MIN_VARIANCE = 1e-10;
}  // namespace

Forecaster::Forecaster(const ag::models::composite::ArimaGarchModel& model) : model_(model) {}

ForecastResult Forecaster::forecast(int horizon) const {
    if (horizon <= 0) {
        throw std::invalid_argument("Forecast horizon must be positive");
    }

    ForecastResult result(horizon);

    const auto& spec = model_.getSpec();
    const auto& arima_params = model_.getArimaParams();
    const auto& garch_params = model_.getGarchParams();
    const auto& arima_state = model_.getArimaState();
    const auto& garch_state = model_.getGarchState();

    // Initialize working histories from current model state
    // For ARIMA: need p observations and q residuals
    std::vector<double> obs_history = arima_state.getObservationHistory();
    std::vector<double> res_history = arima_state.getResidualHistory();

    // For GARCH: need p variances and q squared residuals
    std::vector<double> var_history = garch_state.getVarianceHistory();
    std::vector<double> sq_res_history = garch_state.getSquaredResidualHistory();

    // Iterate over the forecast horizon
    for (int h = 0; h < horizon; ++h) {
        // Step 1: Forecast mean for step h+1
        double mean_forecast = forecastMeanOneStep(obs_history, res_history);
        result.mean_forecasts[h] = mean_forecast;

        // Step 2: Forecast variance for step h+1
        double var_forecast = forecastVarianceOneStep(var_history, sq_res_history);
        result.variance_forecasts[h] = var_forecast;

        // Step 3: Update observation history for next iteration
        // Shift left and add new forecast (oldest removed)
        if (spec.arimaSpec.p > 0) {
            for (int i = 0; i < spec.arimaSpec.p - 1; ++i) {
                obs_history[i] = obs_history[i + 1];
            }
            obs_history[spec.arimaSpec.p - 1] = mean_forecast;
        }

        // Step 4: Update residual history for next iteration
        // Future residuals have expectation zero
        if (spec.arimaSpec.q > 0) {
            for (int i = 0; i < spec.arimaSpec.q - 1; ++i) {
                res_history[i] = res_history[i + 1];
            }
            res_history[spec.arimaSpec.q - 1] = 0.0;
        }

        // Step 5: Update variance history for next iteration
        // Shift left and add new variance forecast
        if (spec.garchSpec.p > 0) {
            for (int i = 0; i < spec.garchSpec.p - 1; ++i) {
                var_history[i] = var_history[i + 1];
            }
            var_history[spec.garchSpec.p - 1] = var_forecast;
        }

        // Step 6: Update squared residual history for next iteration
        // E[ε²_{t+h}] = Var[ε_{t+h}] + E[ε_{t+h}]² = h_{t+h} + 0 = h_{t+h}
        if (spec.garchSpec.q > 0) {
            for (int i = 0; i < spec.garchSpec.q - 1; ++i) {
                sq_res_history[i] = sq_res_history[i + 1];
            }
            sq_res_history[spec.garchSpec.q - 1] = var_forecast;
        }
    }

    return result;
}

double Forecaster::forecastMeanOneStep(const std::vector<double>& obs_history,
                                       const std::vector<double>& res_history) const {
    const auto& arima_params = model_.getArimaParams();
    const auto& spec = model_.getSpec();

    double mean_forecast = arima_params.intercept;

    // Add AR component: φ₁*y_{t-1} + φ₂*y_{t-2} + ... + φₚ*y_{t-p}
    // History is stored with oldest first, so obs_history[p-1] is most recent
    for (int i = 0; i < spec.arimaSpec.p; ++i) {
        mean_forecast += arima_params.ar_coef[i] * obs_history[spec.arimaSpec.p - 1 - i];
    }

    // Add MA component: θ₁*ε_{t-1} + θ₂*ε_{t-2} + ... + θ_q*ε_{t-q}
    // History is stored with oldest first, so res_history[q-1] is most recent
    for (int i = 0; i < spec.arimaSpec.q; ++i) {
        mean_forecast += arima_params.ma_coef[i] * res_history[spec.arimaSpec.q - 1 - i];
    }

    return mean_forecast;
}

double Forecaster::forecastVarianceOneStep(const std::vector<double>& var_history,
                                           const std::vector<double>& sq_res_history) const {
    const auto& garch_params = model_.getGarchParams();
    const auto& spec = model_.getSpec();

    double var_forecast = garch_params.omega;

    // Add ARCH component: α₁*E[ε²_{t-1}] + α₂*E[ε²_{t-2}] + ... + α_q*E[ε²_{t-q}]
    // History is stored with oldest first, so sq_res_history[q-1] is most recent
    for (int i = 0; i < spec.garchSpec.q; ++i) {
        var_forecast += garch_params.alpha_coef[i] * sq_res_history[spec.garchSpec.q - 1 - i];
    }

    // Add GARCH component: β₁*h_{t-1} + β₂*h_{t-2} + ... + βₚ*h_{t-p}
    // History is stored with oldest first, so var_history[p-1] is most recent
    for (int i = 0; i < spec.garchSpec.p; ++i) {
        var_forecast += garch_params.beta_coef[i] * var_history[spec.garchSpec.p - 1 - i];
    }

    // Ensure variance is positive (guard against numerical issues)
    var_forecast = std::max(var_forecast, MIN_VARIANCE);

    return var_forecast;
}

}  // namespace ag::forecasting
