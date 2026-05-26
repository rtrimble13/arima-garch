#include "ag/estimation/FitDriver.hpp"

#include "ag/estimation/Likelihood.hpp"
#include "ag/estimation/ParameterInitialization.hpp"
#include "ag/estimation/ParameterVector.hpp"

#include <stdexcept>

namespace ag::estimation {

std::optional<FitOutcome> runFit(const double* data, std::size_t n,
                                 const ag::models::ArimaGarchSpec& spec,
                                 const FitOptions& options) {
    if (data == nullptr || n == 0) {
        return std::nullopt;
    }

    // Initialize parameters from data. If this throws we report failure
    // by returning nullopt -- the caller can't recover without more data.
    ag::models::arima::ArimaParameters arima_init(spec.arimaSpec.p, spec.arimaSpec.q);
    ag::models::garch::GarchParameters garch_init(spec.garchSpec.p, spec.garchSpec.q);
    try {
        auto initialized = initializeArimaGarchParameters(data, n, spec);
        arima_init = std::move(initialized.first);
        garch_init = std::move(initialized.second);
    } catch (...) {
        return std::nullopt;
    }

    const InnovationDistribution dist = options.innovation.type;
    const double df = options.innovation.df;
    ArimaGarchLikelihood likelihood(spec, dist);

    std::vector<double> initial_params = param_vector::pack(spec, arima_init, garch_init);
    if (initial_params.empty()) {
        // Degenerate spec with no free parameters. Nothing to optimize.
        FitOutcome outcome(spec);
        outcome.parameters.arima_params = arima_init;
        outcome.parameters.garch_params = garch_init;
        try {
            outcome.neg_log_likelihood =
                likelihood.computeNegativeLogLikelihood(data, n, arima_init, garch_init, df);
        } catch (...) {
            return std::nullopt;
        }
        outcome.converged = true;
        outcome.iterations = 0;
        outcome.message = "No free parameters; returning initial values";
        return outcome;
    }

    // Single canonical objective. Returns CONSTRAINT_PENALTY for
    // infeasible parameter vectors and for exceptions raised by the
    // likelihood (e.g. non-positive variance at a degenerate point).
    auto objective = [&](const std::vector<double>& params) -> double {
        ag::models::arima::ArimaParameters arima_p(spec.arimaSpec.p, spec.arimaSpec.q);
        ag::models::garch::GarchParameters garch_p(spec.garchSpec.p, spec.garchSpec.q);
        try {
            param_vector::unpack(params, spec, arima_p, garch_p);
        } catch (...) {
            return CONSTRAINT_PENALTY;
        }

        if (!spec.garchSpec.isNull()) {
            if (!garch_p.isPositive() || !garch_p.isStationary()) {
                return CONSTRAINT_PENALTY;
            }
        }

        try {
            return likelihood.computeNegativeLogLikelihood(data, n, arima_p, garch_p, df);
        } catch (...) {
            return CONSTRAINT_PENALTY;
        }
    };

    const OptimizerConfig& cfg = options.optimizer;
    NelderMeadOptimizer optimizer(cfg.ftol, cfg.xtol, cfg.max_iterations);

    OptimizationResultWithRestarts result;
    try {
        result = optimizeWithRestarts(optimizer, objective, initial_params, cfg.restarts,
                                      cfg.perturbation_scale, cfg.seed);
    } catch (...) {
        return std::nullopt;
    }

    FitOutcome outcome(spec);
    try {
        param_vector::unpack(result.parameters, spec, outcome.parameters.arima_params,
                             outcome.parameters.garch_params);
    } catch (...) {
        return std::nullopt;
    }

    // The unpacker leaves arima params at their constructed defaults when
    // the spec is zero-order; carry over the initialized intercept (zero)
    // and any state we'd expect for the report. For non-zero specs the
    // unpacker has already filled them.
    outcome.neg_log_likelihood = result.objective_value;
    outcome.converged = result.converged;
    outcome.iterations = result.iterations;
    outcome.message = result.message;

    return outcome;
}

}  // namespace ag::estimation
