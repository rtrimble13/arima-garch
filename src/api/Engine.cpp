#include "ag/api/Engine.hpp"

#include "ag/diagnostics/DiagnosticReport.hpp"
#include "ag/estimation/Likelihood.hpp"
#include "ag/estimation/Optimizer.hpp"
#include "ag/estimation/ParameterInitialization.hpp"
#include "ag/selection/InformationCriteria.hpp"

#include <cmath>
#include <stdexcept>

namespace ag::api {

Engine::Engine() {}

expected<FitResult, EngineError> Engine::fit(const std::vector<double>& data,
                                             const models::ArimaGarchSpec& spec,
                                             bool compute_diagnostics) {
    // Validate input
    if (data.size() < 10) {
        return unexpected<EngineError>({"Insufficient data: need at least 10 observations, got " +
                                        std::to_string(data.size())});
    }

    if (data.size() <= spec.totalParamCount()) {
        return unexpected<EngineError>(
            {"Insufficient data: need more observations than parameters. Got " +
             std::to_string(data.size()) + " observations but need " +
             std::to_string(spec.totalParamCount()) + " parameters"});
    }

    try {
        // Step 1: Initialize parameters
        auto [arima_init, garch_init] =
            estimation::initializeArimaGarchParameters(data.data(), data.size(), spec);

        // Step 2: Build likelihood function
        estimation::ArimaGarchLikelihood likelihood(spec);

        // Step 3: Pack initial parameters
        std::vector<double> initial_params = packParameters(arima_init, garch_init);

        // Step 4: Define objective function with constraint checking
        auto objective = [&](const std::vector<double>& params) -> double {
            models::arima::ArimaParameters arima_p(spec.arimaSpec.p, spec.arimaSpec.q);
            models::garch::GarchParameters garch_p(spec.garchSpec.p, spec.garchSpec.q);

            unpackParameters(params, spec, arima_p, garch_p);

            // Check GARCH constraints
            if (!garch_p.isPositive() || !garch_p.isStationary()) {
                return 1e10;  // Penalty for constraint violation
            }

            try {
                return likelihood.computeNegativeLogLikelihood(data.data(), data.size(), arima_p,
                                                               garch_p);
            } catch (...) {
                return 1e10;  // Penalty for evaluation failure
            }
        };

        // Step 5: Run optimization with random restarts
        estimation::NelderMeadOptimizer optimizer(OPTIMIZER_FTOL, OPTIMIZER_XTOL,
                                                  OPTIMIZER_MAX_ITER);
        auto opt_result = estimation::optimizeWithRestarts(optimizer, objective, initial_params,
                                                           NUM_RESTARTS, PERTURBATION_SCALE, 0);

        if (!opt_result.converged) {
            return unexpected<EngineError>(
                {"Optimization failed to converge: " + opt_result.message});
        }

        // Step 6: Unpack optimized parameters
        models::arima::ArimaParameters arima_params(spec.arimaSpec.p, spec.arimaSpec.q);
        models::garch::GarchParameters garch_params(spec.garchSpec.p, spec.garchSpec.q);
        unpackParameters(opt_result.parameters, spec, arima_params, garch_params);

        // Step 7: Build the fitted model
        models::composite::ArimaGarchParameters model_params(spec);
        model_params.arima_params = arima_params;
        model_params.garch_params = garch_params;

        auto fitted_model =
            std::make_shared<models::composite::ArimaGarchModel>(spec, model_params);

        // Initialize model state with data
        for (const auto& value : data) {
            fitted_model->update(value);
        }

        // Step 8: Create FitSummary
        report::FitSummary summary(spec);
        summary.parameters = model_params;
        summary.converged = opt_result.converged;
        summary.iterations = opt_result.iterations;
        summary.message = opt_result.message;
        summary.sample_size = data.size();
        summary.neg_log_likelihood = opt_result.objective_value;

        // Compute information criteria
        std::size_t k = spec.totalParamCount();
        std::size_t n = data.size();
        summary.aic = selection::computeAIC(opt_result.objective_value, k);
        summary.bic = selection::computeBIC(opt_result.objective_value, k, n);

        // Step 9: Compute diagnostics if requested
        if (compute_diagnostics) {
            std::size_t ljung_box_lags = std::min(static_cast<std::size_t>(10), n / 5);
            ljung_box_lags = std::max(ljung_box_lags, k + 1);

            try {
                auto diagnostics =
                    diagnostics::computeDiagnostics(spec, model_params, data, ljung_box_lags, true);
                summary.diagnostics = diagnostics;
            } catch (const std::exception& e) {
                // If diagnostics fail, we still return the fit but without diagnostics
                // This is not a critical failure
            }
        }

        return FitResult(fitted_model, summary);

    } catch (const std::exception& e) {
        return unexpected<EngineError>({"Fit failed: " + std::string(e.what())});
    }
}

expected<SelectionResult, EngineError>
Engine::auto_select(const std::vector<double>& data,
                    const std::vector<models::ArimaGarchSpec>& candidates,
                    selection::SelectionCriterion criterion) {
    // Validate input
    if (data.size() < 10) {
        return unexpected<EngineError>({"Insufficient data: need at least 10 observations, got " +
                                        std::to_string(data.size())});
    }

    if (candidates.empty()) {
        return unexpected<EngineError>({"No candidate models provided"});
    }

    try {
        // Step 1: Run model selection
        selection::ModelSelector selector(criterion);
        auto selection_result = selector.select(data.data(), data.size(), candidates, false);

        if (!selection_result) {
            return unexpected<EngineError>(
                {"Model selection failed: all candidates failed to fit"});
        }

        // Step 2: Fit the best model with diagnostics
        auto fit_result = fit(data, selection_result->best_spec, true);

        if (!fit_result) {
            return unexpected<EngineError>(
                {"Failed to fit selected model: " + fit_result.error().message});
        }

        // Step 3: Create SelectionResult
        return SelectionResult(selection_result->best_spec, fit_result.value().model,
                               fit_result.value().summary, selection_result->candidates_evaluated,
                               selection_result->candidates_failed);

    } catch (const std::exception& e) {
        return unexpected<EngineError>({"Auto-selection failed: " + std::string(e.what())});
    }
}

expected<forecasting::ForecastResult, EngineError>
Engine::forecast(const models::composite::ArimaGarchModel& model, int horizon) {
    // Validate input
    if (horizon <= 0) {
        return unexpected<EngineError>(
            {"Invalid horizon: must be positive, got " + std::to_string(horizon)});
    }

    try {
        forecasting::Forecaster forecaster(model);
        auto forecast_result = forecaster.forecast(horizon);
        return forecast_result;
    } catch (const std::exception& e) {
        return unexpected<EngineError>({"Forecast failed: " + std::string(e.what())});
    }
}

expected<simulation::SimulationResult, EngineError>
Engine::simulate(const models::ArimaGarchSpec& spec,
                 const models::composite::ArimaGarchParameters& params, int length,
                 unsigned int seed) {
    // Validate input
    if (length <= 0) {
        return unexpected<EngineError>(
            {"Invalid length: must be positive, got " + std::to_string(length)});
    }

    try {
        simulation::ArimaGarchSimulator simulator(spec, params);
        auto sim_result = simulator.simulate(length, seed);
        return sim_result;
    } catch (const std::exception& e) {
        return unexpected<EngineError>({"Simulation failed: " + std::string(e.what())});
    }
}

std::vector<double>
Engine::packParameters(const models::arima::ArimaParameters& arima_params,
                       const models::garch::GarchParameters& garch_params) const {
    std::vector<double> params;

    // Pack ARIMA parameters
    params.push_back(arima_params.intercept);
    for (const auto& ar : arima_params.ar_coef) {
        params.push_back(ar);
    }
    for (const auto& ma : arima_params.ma_coef) {
        params.push_back(ma);
    }

    // Pack GARCH parameters
    params.push_back(garch_params.omega);
    for (const auto& alpha : garch_params.alpha_coef) {
        params.push_back(alpha);
    }
    for (const auto& beta : garch_params.beta_coef) {
        params.push_back(beta);
    }

    return params;
}

void Engine::unpackParameters(const std::vector<double>& params, const models::ArimaGarchSpec& spec,
                              models::arima::ArimaParameters& out_arima,
                              models::garch::GarchParameters& out_garch) const {
    std::size_t idx = 0;

    // Unpack ARIMA parameters
    out_arima.intercept = params[idx++];
    for (std::size_t i = 0; i < spec.arimaSpec.p; ++i) {
        out_arima.ar_coef[i] = params[idx++];
    }
    for (std::size_t i = 0; i < spec.arimaSpec.q; ++i) {
        out_arima.ma_coef[i] = params[idx++];
    }

    // Unpack GARCH parameters
    out_garch.omega = params[idx++];
    for (std::size_t i = 0; i < spec.garchSpec.p; ++i) {
        out_garch.alpha_coef[i] = params[idx++];
    }
    for (std::size_t i = 0; i < spec.garchSpec.q; ++i) {
        out_garch.beta_coef[i] = params[idx++];
    }
}

}  // namespace ag::api
