#include "ag/estimation/FitDriver.hpp"

#include "ag/estimation/Likelihood.hpp"
#include "ag/estimation/ParameterInitialization.hpp"
#include "ag/estimation/ParameterVector.hpp"

#include <cmath>
#include <stdexcept>

namespace ag::estimation {

std::optional<FitOutcome> runFit(const double* data, std::size_t n,
                                 const ag::models::ArimaGarchSpec& spec,
                                 const FitOptions& options) {
    if (data == nullptr || n == 0) {
        return std::nullopt;
    }

    // Validate input finiteness once, up front. This must happen outside the
    // optimizer objective: the objective maps thrown exceptions to
    // CONSTRAINT_PENALTY, so non-finite data evaluated there would produce a
    // flat penalty surface that Nelder-Mead can report as "converged" on
    // garbage parameters. Failing here yields an actionable non-converged
    // outcome instead.
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(data[i])) {
            FitOutcome failure(spec);
            failure.converged = false;
            failure.message =
                "Input data contains non-finite values (NaN or Inf) at index " + std::to_string(i);
            return failure;
        }
    }

    // Initialize parameters from data. If this throws we return a non-
    // converged outcome with the error message so the caller has actionable
    // diagnostics (e.g. "Insufficient data after differencing").
    ag::models::arima::ArimaParameters arima_init(spec.arimaSpec.p, spec.arimaSpec.q);
    ag::models::garch::GarchParameters garch_init(spec.garchSpec.p, spec.garchSpec.q);
    try {
        auto initialized = initializeArimaGarchParameters(data, n, spec);
        arima_init = std::move(initialized.first);
        garch_init = std::move(initialized.second);
    } catch (const std::exception& e) {
        FitOutcome failure(spec);
        failure.converged = false;
        failure.message = std::string("Parameter initialization failed: ") + e.what();
        return failure;
    } catch (...) {
        FitOutcome failure(spec);
        failure.converged = false;
        failure.message = "Parameter initialization failed: unknown error";
        return failure;
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
        } catch (const std::exception& e) {
            outcome.converged = false;
            outcome.message = std::string("Likelihood evaluation failed: ") + e.what();
            return outcome;
        } catch (...) {
            outcome.converged = false;
            outcome.message = "Likelihood evaluation failed: unknown error";
            return outcome;
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

        // Enforce AR stationarity (mirrors the GARCH constraint below): steer
        // the optimizer away from the explosive AR region, where the mean
        // recursion diverges. A zero-order AR part is trivially stationary, so
        // this is a no-op for those specs. MA invertibility is intentionally
        // not enforced (a non-invertible MA has an observationally-equivalent
        // invertible representation); it remains report-only.
        if (!arima_p.isStationary()) {
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
    } catch (const std::exception& e) {
        FitOutcome failure(spec);
        failure.converged = false;
        failure.message = std::string("Optimizer failed: ") + e.what();
        return failure;
    } catch (...) {
        FitOutcome failure(spec);
        failure.converged = false;
        failure.message = "Optimizer failed: unknown error";
        return failure;
    }

    FitOutcome outcome(spec);
    try {
        param_vector::unpack(result.parameters, spec, outcome.parameters.arima_params,
                             outcome.parameters.garch_params);
    } catch (const std::exception& e) {
        FitOutcome failure(spec);
        failure.converged = false;
        failure.message = std::string("Parameter unpack failed: ") + e.what();
        return failure;
    } catch (...) {
        FitOutcome failure(spec);
        failure.converged = false;
        failure.message = "Parameter unpack failed: unknown error";
        return failure;
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
