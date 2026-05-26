#pragma once

#include "ag/estimation/InnovationSpec.hpp"
#include "ag/estimation/Optimizer.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"

#include <cstddef>
#include <optional>
#include <string>

namespace ag::estimation {

/**
 * @brief Options bundle for runFit.
 */
struct FitOptions {
    InnovationSpec innovation = InnovationSpec::normal();
    OptimizerConfig optimizer{};
};

/**
 * @brief Output of runFit. Mirrors the relevant fields of
 *        OptimizationResult plus the unpacked parameter structs.
 */
struct FitOutcome {
    ag::models::composite::ArimaGarchParameters parameters;
    double neg_log_likelihood = 0.0;
    bool converged = false;
    int iterations = 0;
    std::string message;

    explicit FitOutcome(const ag::models::ArimaGarchSpec& spec) : parameters(spec) {}
};

/**
 * @brief Single canonical implementation of the ARIMA-GARCH fit pipeline:
 *        initialize parameters -> pack -> Nelder-Mead with restarts ->
 *        unpack -> return.
 *
 * Replaces four previous copies of this exact loop in Engine,
 * ModelSelector (two branches), and CrossValidation. The objective
 * function applies CONSTRAINT_PENALTY for GARCH positivity / stationarity
 * violations and for likelihood-computation exceptions, exactly matching
 * the prior behavior.
 *
 * Returns std::nullopt only when data is null or empty (a programming
 * error that should have been caught by the caller). All other failure
 * modes — parameter initialization exceptions (e.g. insufficient data
 * after differencing), likelihood evaluation failures in the no-free-
 * parameters path, optimizer exceptions, and parameter-unpack failures —
 * are reported as FitOutcome{converged=false, message=<reason>} so callers
 * receive actionable diagnostics. Successful convergence is indicated by
 * FitOutcome{converged=true}.
 */
[[nodiscard]] std::optional<FitOutcome> runFit(const double* data, std::size_t n,
                                               const ag::models::ArimaGarchSpec& spec,
                                               const FitOptions& options = {});

}  // namespace ag::estimation
