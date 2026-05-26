#include "ag/selection/CrossValidation.hpp"

#include "ag/estimation/FitDriver.hpp"
#include "ag/forecasting/Forecaster.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>

namespace ag::selection {

std::optional<CrossValidationResult>
computeCrossValidationScore(const double* data, std::size_t n_obs,
                            const ag::models::ArimaGarchSpec& spec,
                            const CrossValidationConfig& config) {
    // Validate inputs
    if (data == nullptr) {
        throw std::invalid_argument("computeCrossValidationScore: data cannot be nullptr");
    }
    if (n_obs == 0) {
        throw std::invalid_argument("computeCrossValidationScore: n_obs must be > 0");
    }
    if (config.min_train_size == 0) {
        throw std::invalid_argument("computeCrossValidationScore: min_train_size must be > 0");
    }
    if (config.min_train_size >= n_obs) {
        throw std::invalid_argument("computeCrossValidationScore: min_train_size must be < n_obs");
    }

    // Number of forecast windows
    std::size_t n_windows = n_obs - config.min_train_size;
    if (n_windows == 0) {
        return std::nullopt;
    }

    // Accumulate squared forecast errors
    double sum_squared_errors = 0.0;
    std::size_t successful_forecasts = 0;

    // Rolling origin cross-validation
    for (std::size_t window_end = config.min_train_size; window_end < n_obs; ++window_end) {
        try {
            const double* train_data = data;
            const std::size_t train_size = window_end;
            const double actual_value = data[window_end];

            auto outcome = ag::estimation::runFit(train_data, train_size, spec);
            if (!outcome || !outcome->converged) {
                continue;
            }

            ag::models::composite::ArimaGarchModel model(spec, outcome->parameters);
            for (std::size_t i = 0; i < train_size; ++i) {
                model.update(train_data[i]);
            }

            ag::forecasting::Forecaster forecaster(model);
            auto forecast_result = forecaster.forecast(1);
            const double forecast_value = forecast_result.mean_forecasts[0];

            const double error = actual_value - forecast_value;
            sum_squared_errors += error * error;
            ++successful_forecasts;
        } catch (...) {
            continue;
        }
    }

    // Check if we got at least some successful forecasts
    if (successful_forecasts == 0) {
        return std::nullopt;
    }

    // Compute MSE
    double mse = sum_squared_errors / static_cast<double>(successful_forecasts);

    return CrossValidationResult(mse, successful_forecasts);
}

}  // namespace ag::selection
