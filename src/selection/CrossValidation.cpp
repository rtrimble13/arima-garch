#include "ag/selection/CrossValidation.hpp"

#include "ag/estimation/Likelihood.hpp"
#include "ag/estimation/Optimizer.hpp"
#include "ag/estimation/ParameterInitialization.hpp"
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
            // Training data: [0, window_end)
            const double* train_data = data;
            std::size_t train_size = window_end;

            // Test data: data[window_end] (1-step-ahead target)
            double actual_value = data[window_end];

            // Initialize parameters from training data
            auto [arima_init, garch_init] =
                ag::estimation::initializeArimaGarchParameters(train_data, train_size, spec);

            // Create likelihood function
            ag::estimation::ArimaGarchLikelihood likelihood(spec);

            // Pack parameters into vector for optimization
            std::vector<double> initial_params;

            // Add ARIMA parameters if not zero-order
            if (!spec.arimaSpec.isZeroOrder()) {
                initial_params.push_back(arima_init.intercept);
                for (int i = 0; i < spec.arimaSpec.p; ++i) {
                    initial_params.push_back(arima_init.ar_coef[i]);
                }
                for (int i = 0; i < spec.arimaSpec.q; ++i) {
                    initial_params.push_back(arima_init.ma_coef[i]);
                }
            }

            // Add GARCH parameters
            initial_params.push_back(garch_init.omega);
            for (int i = 0; i < spec.garchSpec.p; ++i) {
                initial_params.push_back(garch_init.alpha_coef[i]);
            }
            for (int i = 0; i < spec.garchSpec.q; ++i) {
                initial_params.push_back(garch_init.beta_coef[i]);
            }

            // Define objective function with constraints
            auto objective = [&](const std::vector<double>& params) -> double {
                ag::models::arima::ArimaParameters arima_p(spec.arimaSpec.p, spec.arimaSpec.q);
                ag::models::garch::GarchParameters garch_p(spec.garchSpec.p, spec.garchSpec.q);

                std::size_t idx = 0;

                // Unpack ARIMA parameters if not zero-order
                if (!spec.arimaSpec.isZeroOrder()) {
                    arima_p.intercept = params[idx++];
                    for (int i = 0; i < spec.arimaSpec.p; ++i) {
                        arima_p.ar_coef[i] = params[idx++];
                    }
                    for (int i = 0; i < spec.arimaSpec.q; ++i) {
                        arima_p.ma_coef[i] = params[idx++];
                    }
                }

                // Unpack GARCH parameters
                garch_p.omega = params[idx++];
                for (int i = 0; i < spec.garchSpec.p; ++i) {
                    garch_p.alpha_coef[i] = params[idx++];
                }
                for (int i = 0; i < spec.garchSpec.q; ++i) {
                    garch_p.beta_coef[i] = params[idx++];
                }

                // Check GARCH constraints
                if (!garch_p.isPositive() || !garch_p.isStationary()) {
                    return 1e10;  // Penalty for constraint violation
                }

                // Compute negative log-likelihood
                try {
                    return likelihood.computeNegativeLogLikelihood(train_data, train_size, arima_p,
                                                                   garch_p);
                } catch (...) {
                    return 1e10;  // Penalty for computation errors
                }
            };

            // Set up optimizer
            ag::estimation::NelderMeadOptimizer optimizer(1e-6, 1e-6, 2000);

            // Optimize with random restarts for robustness
            auto result = ag::estimation::optimizeWithRestarts(optimizer, objective, initial_params,
                                                               3, 0.15, 42);

            // Check if optimization succeeded
            if (!result.converged) {
                // Skip this window if fitting failed
                continue;
            }

            // Unpack optimized parameters
            ag::models::arima::ArimaParameters fitted_arima(spec.arimaSpec.p, spec.arimaSpec.q);
            ag::models::garch::GarchParameters fitted_garch(spec.garchSpec.p, spec.garchSpec.q);

            std::size_t idx = 0;
            if (!spec.arimaSpec.isZeroOrder()) {
                fitted_arima.intercept = result.parameters[idx++];
                for (int i = 0; i < spec.arimaSpec.p; ++i) {
                    fitted_arima.ar_coef[i] = result.parameters[idx++];
                }
                for (int i = 0; i < spec.arimaSpec.q; ++i) {
                    fitted_arima.ma_coef[i] = result.parameters[idx++];
                }
            }
            fitted_garch.omega = result.parameters[idx++];
            for (int i = 0; i < spec.garchSpec.p; ++i) {
                fitted_garch.alpha_coef[i] = result.parameters[idx++];
            }
            for (int i = 0; i < spec.garchSpec.q; ++i) {
                fitted_garch.beta_coef[i] = result.parameters[idx++];
            }

            // Create model with fitted parameters and initialize state
            ag::models::composite::ArimaGarchParameters params(spec);
            params.arima_params = fitted_arima;
            params.garch_params = fitted_garch;

            ag::models::composite::ArimaGarchModel model(spec, params);

            // Initialize model state by processing training data
            for (std::size_t i = 0; i < train_size; ++i) {
                model.update(train_data[i]);
            }

            // Make 1-step-ahead forecast
            ag::forecasting::Forecaster forecaster(model);
            auto forecast_result = forecaster.forecast(1);

            double forecast_value = forecast_result.mean_forecasts[0];

            // Compute squared error
            double error = actual_value - forecast_value;
            double squared_error = error * error;

            sum_squared_errors += squared_error;
            successful_forecasts++;

        } catch (...) {
            // Skip this window if any error occurs
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
