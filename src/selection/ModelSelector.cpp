#include "ag/selection/ModelSelector.hpp"

#include "ag/diagnostics/DiagnosticReport.hpp"
#include "ag/estimation/Likelihood.hpp"
#include "ag/estimation/Optimizer.hpp"
#include "ag/estimation/ParameterInitialization.hpp"
#include "ag/selection/InformationCriteria.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace ag::selection {

ModelSelector::ModelSelector(SelectionCriterion criterion) : criterion_(criterion) {}

std::optional<SelectionResult>
ModelSelector::select(const double* data, std::size_t n_obs,
                      const std::vector<ag::models::ArimaGarchSpec>& candidates,
                      bool compute_diagnostics) {
    // Validate inputs
    if (data == nullptr) {
        throw std::invalid_argument("ModelSelector::select: data cannot be nullptr");
    }
    if (n_obs == 0) {
        throw std::invalid_argument("ModelSelector::select: n_obs must be > 0");
    }
    if (candidates.empty()) {
        throw std::invalid_argument("ModelSelector::select: candidates cannot be empty");
    }

    // Track best model
    std::optional<SelectionResult> best_result;
    double best_score = std::numeric_limits<double>::infinity();

    std::size_t evaluated = 0;
    std::size_t failed = 0;

    // Iterate through all candidates
    for (const auto& candidate : candidates) {
        ag::report::FitSummary fit_summary(candidate);

        // Try to fit this candidate
        auto score_opt = fitAndScore(data, n_obs, candidate, fit_summary);

        if (score_opt.has_value()) {
            // Successful fit
            evaluated++;
            double score = score_opt.value();

            // Check if this is the best model so far
            if (score < best_score) {
                best_score = score;

                // Create new result with best model
                best_result.emplace(candidate, score, fit_summary.parameters);
                best_result->best_fit_summary.emplace(fit_summary);
            }
        } else {
            // Failed to fit this candidate
            failed++;
        }
    }

    // Update statistics if we found at least one valid model
    if (best_result.has_value()) {
        best_result->candidates_evaluated = evaluated;
        best_result->candidates_failed = failed;

        // Compute diagnostics for best model if requested
        if (compute_diagnostics && best_result->best_fit_summary.has_value()) {
            auto& summary = best_result->best_fit_summary.value();
            std::vector<double> data_vec(data, data + n_obs);
            auto diagnostics = ag::diagnostics::computeDiagnostics(summary.spec, summary.parameters,
                                                                   data_vec, 10, true);
            summary.diagnostics = diagnostics;
        }
    }

    return best_result;
}

std::optional<double> ModelSelector::fitAndScore(const double* data, std::size_t n_obs,
                                                 const ag::models::ArimaGarchSpec& spec,
                                                 ag::report::FitSummary& out_summary) {
    try {
        // Initialize parameters from data
        auto [arima_init, garch_init] =
            ag::estimation::initializeArimaGarchParameters(data, n_obs, spec);

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
                return likelihood.computeNegativeLogLikelihood(data, n_obs, arima_p, garch_p);
            } catch (...) {
                return 1e10;  // Penalty for computation errors
            }
        };

        // Set up optimizer with standard settings
        ag::estimation::NelderMeadOptimizer optimizer(1e-6, 1e-6, 2000);

        // Optimize with random restarts for robustness
        auto result =
            ag::estimation::optimizeWithRestarts(optimizer, objective, initial_params, 3, 0.15, 42);

        // Check if optimization succeeded
        if (!result.converged) {
            return std::nullopt;  // Failed to converge
        }

        // Unpack optimized parameters into FitSummary
        std::size_t idx = 0;
        if (!spec.arimaSpec.isZeroOrder()) {
            out_summary.parameters.arima_params.intercept = result.parameters[idx++];
            for (int i = 0; i < spec.arimaSpec.p; ++i) {
                out_summary.parameters.arima_params.ar_coef[i] = result.parameters[idx++];
            }
            for (int i = 0; i < spec.arimaSpec.q; ++i) {
                out_summary.parameters.arima_params.ma_coef[i] = result.parameters[idx++];
            }
        }
        out_summary.parameters.garch_params.omega = result.parameters[idx++];
        for (int i = 0; i < spec.garchSpec.p; ++i) {
            out_summary.parameters.garch_params.alpha_coef[i] = result.parameters[idx++];
        }
        for (int i = 0; i < spec.garchSpec.q; ++i) {
            out_summary.parameters.garch_params.beta_coef[i] = result.parameters[idx++];
        }

        // Populate FitSummary
        out_summary.neg_log_likelihood = result.objective_value;
        out_summary.converged = result.converged;
        out_summary.iterations = result.iterations;
        out_summary.message = result.message;
        out_summary.sample_size = n_obs;

        // Compute information criteria
        int k = spec.totalParamCount();
        double log_lik = -result.objective_value;  // Convert NLL to log-likelihood
        out_summary.aic = computeAIC(log_lik, k);
        out_summary.bic = computeBIC(log_lik, k, n_obs);

        // Extract and return the score based on selection criterion
        return extractScore(out_summary);

    } catch (...) {
        // Any exception during fitting means this candidate failed
        return std::nullopt;
    }
}

double ModelSelector::extractScore(const ag::report::FitSummary& summary) const noexcept {
    switch (criterion_) {
    case SelectionCriterion::AIC:
        return summary.aic;
    case SelectionCriterion::BIC:
        return summary.bic;
    case SelectionCriterion::AICc: {
        // Compute AICc on the fly
        int k = summary.spec.totalParamCount();
        double log_lik = -summary.neg_log_likelihood;
        try {
            return computeAICc(log_lik, k, summary.sample_size);
        } catch (...) {
            // If AICc computation fails (e.g., sample size too small),
            // fall back to AIC
            return summary.aic;
        }
    }
    }
    // Default to BIC if somehow we get here
    return summary.bic;
}

}  // namespace ag::selection
