#include "ag/selection/ModelSelector.hpp"

#include "ag/diagnostics/DiagnosticReport.hpp"
#include "ag/estimation/FitDriver.hpp"
#include "ag/selection/CrossValidation.hpp"
#include "ag/selection/InformationCriteria.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ag::selection {

ModelSelector::ModelSelector(SelectionCriterion criterion) : criterion_(criterion) {}

std::optional<SelectionResult>
ModelSelector::select(const double* data, std::size_t n_obs,
                      const std::vector<ag::models::ArimaGarchSpec>& candidates,
                      bool compute_diagnostics, bool build_ranking) {
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

    // Track all successful fits for ranking
    std::vector<CandidateRanking> all_rankings;

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

            // Add to ranking if requested
            if (build_ranking) {
                all_rankings.emplace_back(candidate, score, fit_summary.converged);
            }

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

        // Sort and store ranking if requested
        if (build_ranking) {
            std::sort(all_rankings.begin(), all_rankings.end(),
                      [](const CandidateRanking& a, const CandidateRanking& b) {
                          return a.score < b.score;
                      });
            best_result->ranking = std::move(all_rankings);
        }

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

namespace {

// Fit @p spec on the full @p data via the shared FitDriver and populate
// the relevant subset of @p out_summary. Returns true on success.
bool fitFullDataIntoSummary(const double* data, std::size_t n_obs,
                            const ag::models::ArimaGarchSpec& spec,
                            ag::report::FitSummary& out_summary) {
    auto outcome = ag::estimation::runFit(data, n_obs, spec);
    if (!outcome || !outcome->converged) {
        return false;
    }

    out_summary.parameters = outcome->parameters;
    out_summary.neg_log_likelihood = outcome->neg_log_likelihood;
    out_summary.converged = outcome->converged;
    out_summary.iterations = outcome->iterations;
    out_summary.message = outcome->message;
    out_summary.sample_size = n_obs;

    int k = spec.totalParamCount();
    double log_lik = -outcome->neg_log_likelihood;
    out_summary.aic = computeAIC(log_lik, k);
    out_summary.bic = computeBIC(log_lik, k, n_obs);
    return true;
}

}  // namespace

std::optional<double> ModelSelector::fitAndScore(const double* data, std::size_t n_obs,
                                                 const ag::models::ArimaGarchSpec& spec,
                                                 ag::report::FitSummary& out_summary) {
    if (criterion_ == SelectionCriterion::CV) {
        std::size_t min_train_size = static_cast<std::size_t>(n_obs * 0.7);
        if (min_train_size < 50) {
            min_train_size = std::min(static_cast<std::size_t>(50), n_obs - 10);
        }
        if (min_train_size >= n_obs) {
            return std::nullopt;  // Not enough data for CV
        }

        CrossValidationConfig cv_config(min_train_size);
        auto cv_result = computeCrossValidationScore(data, n_obs, spec, cv_config);
        if (!cv_result.has_value()) {
            return std::nullopt;
        }

        // We still fit on the full data to populate out_summary so the
        // selector can return parameters for the chosen model.
        if (!fitFullDataIntoSummary(data, n_obs, spec, out_summary)) {
            return std::nullopt;
        }
        return cv_result->mse;
    }

    // IC-based selection: full-data fit IS the scoring step.
    if (!fitFullDataIntoSummary(data, n_obs, spec, out_summary)) {
        return std::nullopt;
    }
    return extractScore(out_summary);
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
