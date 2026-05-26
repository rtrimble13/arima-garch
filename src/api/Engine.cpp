#include "ag/api/Engine.hpp"

#include "ag/diagnostics/DiagnosticReport.hpp"
#include "ag/estimation/FitDriver.hpp"
#include "ag/estimation/InnovationSpec.hpp"
#include "ag/selection/DistributionSelector.hpp"
#include "ag/selection/InformationCriteria.hpp"

#include <cmath>
#include <stdexcept>

namespace ag::api {

Engine::Engine() {}

expected<FitResult, EngineError> Engine::fit(const std::vector<double>& data,
                                             const models::ArimaGarchSpec& spec,
                                             bool compute_diagnostics, bool use_student_t,
                                             double student_t_df) {
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

    if (use_student_t) {
        auto error = validateStudentTDF(student_t_df);
        if (!error.empty()) {
            return unexpected<EngineError>({error});
        }
    }

    try {
        estimation::FitOptions options;
        options.innovation = use_student_t ? estimation::InnovationSpec::studentT(student_t_df)
                                           : estimation::InnovationSpec::normal();
        options.optimizer.seed = 0;  // Engine historically used a nondeterministic seed.

        auto fit_outcome = estimation::runFit(data.data(), data.size(), spec, options);
        if (!fit_outcome) {
            return unexpected<EngineError>(
                {"Optimization failed: null or empty data passed to fit pipeline"});
        }
        if (!fit_outcome->converged) {
            return unexpected<EngineError>(
                {"Optimization failed to converge: " + fit_outcome->message});
        }

        // Build the fitted model from the converged parameters.
        models::composite::ArimaGarchParameters model_params = fit_outcome->parameters;
        auto fitted_model =
            std::make_shared<models::composite::ArimaGarchModel>(spec, model_params);

        for (const auto& value : data) {
            fitted_model->update(value);
        }

        report::FitSummary summary(spec);
        summary.parameters = model_params;
        summary.converged = fit_outcome->converged;
        summary.iterations = fit_outcome->iterations;
        summary.message = fit_outcome->message;
        summary.sample_size = data.size();
        summary.neg_log_likelihood = fit_outcome->neg_log_likelihood;
        summary.innovation_distribution = use_student_t ? "Student-t" : "Normal";
        summary.student_t_df = use_student_t ? student_t_df : 0.0;

        // Compute information criteria
        // IMPORTANT: computeAIC/computeBIC expect log-likelihood (positive),
        // but optimization returns negative log-likelihood (NLL), so we negate it
        // Note: df parameter is user-provided, not estimated, so we don't count it in k
        std::size_t k = spec.totalParamCount();
        std::size_t n = data.size();
        double log_lik = -fit_outcome->neg_log_likelihood;  // Convert NLL to log-likelihood
        summary.aic = selection::computeAIC(log_lik, k);
        summary.bic = selection::computeBIC(log_lik, k, n);

        // Step 9: Compute diagnostics if requested
        if (compute_diagnostics) {
            std::size_t ljung_box_lags = std::min(static_cast<std::size_t>(10), n / 5);
            ljung_box_lags = std::max(ljung_box_lags, k + 1);
            ljung_box_lags = std::min(ljung_box_lags, n - 1);

            try {
                std::string innovation_dist = use_student_t ? "Student-t" : "Normal";
                auto diagnostics =
                    diagnostics::computeDiagnostics(spec, model_params, data, ljung_box_lags,
                                                    true,             // include_adf
                                                    innovation_dist,  // innovation distribution
                                                    student_t_df);    // df (ignored if Normal)
                summary.diagnostics = diagnostics;
            } catch (const std::exception& e) {
                // If diagnostics fail, we still return the fit but without diagnostics
                // This is not a critical failure
            }

            // Add distribution comparison
            try {
                auto dist_comparison =
                    selection::compareDistributions(spec, model_params, data.data(), data.size());

                // Compute AIC and BIC for both Normal and Student-t innovation distributions
                // Note: These are based on standardized residual log-likelihoods
                summary.distribution_comparison = report::DistributionComparison{
                    .normal_log_likelihood = dist_comparison.normal_ll,
                    .student_t_log_likelihood = dist_comparison.student_t_ll,
                    .student_t_df = dist_comparison.df,
                    .lr_statistic = dist_comparison.lr_statistic,
                    .lr_p_value = dist_comparison.lr_p_value,
                    .prefer_student_t = dist_comparison.prefer_student_t,
                    .normal_aic = -2.0 * dist_comparison.normal_ll + 2.0 * k,
                    .student_t_aic =
                        -2.0 * dist_comparison.student_t_ll + 2.0 * (k + 1),  // +1 for df param
                    .normal_bic = -2.0 * dist_comparison.normal_ll + k * std::log(n),
                    .student_t_bic = -2.0 * dist_comparison.student_t_ll +
                                     (k + 1) * std::log(n)  // +1 for df param
                };
            } catch (const std::exception& e) {
                // If distribution comparison fails, we still return the fit but without it
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
                    selection::SelectionCriterion criterion, bool build_ranking) {
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
        auto selection_result =
            selector.select(data.data(), data.size(), candidates, false, build_ranking);

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

        // Step 3: Create SelectionResult and transfer ranking
        SelectionResult result(selection_result->best_spec, fit_result.value().model,
                               fit_result.value().summary, selection_result->candidates_evaluated,
                               selection_result->candidates_failed);
        result.ranking = std::move(selection_result->ranking);

        return result;

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
                 unsigned int seed, bool use_student_t, double student_t_df) {
    // Validate input
    if (length <= 0) {
        return unexpected<EngineError>(
            {"Invalid length: must be positive, got " + std::to_string(length)});
    }

    if (use_student_t) {
        auto error = validateStudentTDF(student_t_df);
        if (!error.empty()) {
            return unexpected<EngineError>({error});
        }
    }

    try {
        simulation::ArimaGarchSimulator simulator(spec, params);
        simulation::InnovationDistribution dist = use_student_t
                                                      ? simulation::InnovationDistribution::StudentT
                                                      : simulation::InnovationDistribution::Normal;
        auto sim_result = simulator.simulate(length, seed, dist, student_t_df);
        return sim_result;
    } catch (const std::exception& e) {
        return unexpected<EngineError>({"Simulation failed: " + std::string(e.what())});
    }
}

}  // namespace ag::api
