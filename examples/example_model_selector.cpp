/**
 * @file example_model_selector.cpp
 * @brief Demonstrates automatic model selection using information criteria.
 *
 * This example shows how to:
 * 1. Generate synthetic data from a known ARIMA-GARCH model
 * 2. Create a grid of candidate model specifications
 * 3. Use ModelSelector to automatically select the best model
 * 4. Compare different information criteria (BIC, AIC, AICc)
 * 5. Examine the selection results and fitted model
 */

#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/report/FitSummary.hpp"
#include "ag/selection/CandidateGrid.hpp"
#include "ag/selection/ModelSelector.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"

#include <iostream>

#include <fmt/core.h>

using ag::models::ArimaGarchSpec;
using ag::models::composite::ArimaGarchParameters;
using ag::report::generateTextReport;
using ag::selection::CandidateGrid;
using ag::selection::CandidateGridConfig;
using ag::selection::ModelSelector;
using ag::selection::SelectionCriterion;
using ag::simulation::ArimaGarchSimulator;

int main() {
    fmt::print("=== ARIMA-GARCH Model Selection Example ===\n\n");

    // ========================================================================
    // Step 1: Generate synthetic data from a known model
    // ========================================================================
    fmt::print("Step 1: Generating synthetic data...\n");

    // True model: ARIMA(1,0,1)-GARCH(1,1)
    ArimaGarchSpec true_spec(1, 0, 1, 1, 1);
    ArimaGarchParameters true_params(true_spec);

    // Set true parameters
    true_params.arima_params.intercept = 0.05;
    true_params.arima_params.ar_coef[0] = 0.6;
    true_params.arima_params.ma_coef[0] = 0.3;
    true_params.garch_params.omega = 0.01;
    true_params.garch_params.alpha_coef[0] = 0.1;
    true_params.garch_params.beta_coef[0] = 0.85;

    fmt::print("  True model: ARIMA({},{},{})-GARCH({},{})\n", true_spec.arimaSpec.p,
               true_spec.arimaSpec.d, true_spec.arimaSpec.q, true_spec.garchSpec.p,
               true_spec.garchSpec.q);

    // Simulate data
    ArimaGarchSimulator simulator(true_spec, true_params);
    auto sim_result = simulator.simulate(1000, 42);
    const auto& data = sim_result.returns;

    fmt::print("  Generated {} observations\n\n", data.size());

    // ========================================================================
    // Step 2: Create a grid of candidate models
    // ========================================================================
    fmt::print("Step 2: Creating candidate grid...\n");

    // Define search space: ARIMA up to (2,1,2), GARCH up to (1,1)
    CandidateGridConfig config(2, 1, 2, 1, 1);
    config.restrict_d_to_01 = true;  // Only allow d in {0, 1}

    CandidateGrid grid(config);
    auto candidates = grid.generate();

    fmt::print("  Generated {} candidate specifications\n", candidates.size());
    fmt::print("  Search space: ARIMA(0-{},0-{},0-{})-GARCH(1-{},1-{})\n", config.max_p,
               config.max_d, config.max_q, config.max_p_garch, config.max_q_garch);
    fmt::print("\n");

    // ========================================================================
    // Step 3: Select best model using BIC (Bayesian Information Criterion)
    // ========================================================================
    fmt::print("Step 3: Selecting best model using BIC...\n");
    fmt::print("  (This may take a minute as {} models are being fitted)\n\n", candidates.size());

    ModelSelector selector_bic(SelectionCriterion::BIC);
    auto result_bic = selector_bic.select(data.data(), data.size(), candidates, true);

    if (!result_bic.has_value()) {
        fmt::print("  ERROR: Model selection failed (all candidates failed to fit)\n");
        return 1;
    }

    fmt::print("  Selection complete!\n");
    fmt::print("  Candidates evaluated: {}\n", result_bic->candidates_evaluated);
    fmt::print("  Candidates failed: {}\n", result_bic->candidates_failed);
    fmt::print("\n");

    // ========================================================================
    // Step 4: Display BIC results
    // ========================================================================
    fmt::print("=== BIC Selection Results ===\n\n");

    const auto& best_spec_bic = result_bic->best_spec;
    fmt::print("Best model: ARIMA({},{},{})-GARCH({},{})\n", best_spec_bic.arimaSpec.p,
               best_spec_bic.arimaSpec.d, best_spec_bic.arimaSpec.q, best_spec_bic.garchSpec.p,
               best_spec_bic.garchSpec.q);
    fmt::print("BIC score: {:.4f}\n\n", result_bic->best_score);

    // Check if we recovered the true specification
    bool correct_spec = (best_spec_bic.arimaSpec.p == true_spec.arimaSpec.p &&
                         best_spec_bic.arimaSpec.d == true_spec.arimaSpec.d &&
                         best_spec_bic.arimaSpec.q == true_spec.arimaSpec.q &&
                         best_spec_bic.garchSpec.p == true_spec.garchSpec.p &&
                         best_spec_bic.garchSpec.q == true_spec.garchSpec.q);

    if (correct_spec) {
        fmt::print("✓ BIC correctly identified the true model specification!\n\n");
    } else {
        fmt::print("✓ BIC selected a different specification (this can happen due to\n");
        fmt::print("  finite sample variability or if BIC prefers a simpler model)\n\n");
    }

    // Display full fit summary if available
    if (result_bic->best_fit_summary.has_value()) {
        std::string report = generateTextReport(result_bic->best_fit_summary.value());
        fmt::print("{}\n", report);
    }

    // ========================================================================
    // Step 5: Compare with AIC and AICc
    // ========================================================================
    fmt::print("\n=== Comparing with AIC and AICc ===\n\n");

    // Select using AIC
    ModelSelector selector_aic(SelectionCriterion::AIC);
    auto result_aic = selector_aic.select(data.data(), data.size(), candidates, false);

    if (result_aic.has_value()) {
        const auto& best_spec_aic = result_aic->best_spec;
        fmt::print("AIC best model: ARIMA({},{},{})-GARCH({},{})\n", best_spec_aic.arimaSpec.p,
                   best_spec_aic.arimaSpec.d, best_spec_aic.arimaSpec.q, best_spec_aic.garchSpec.p,
                   best_spec_aic.garchSpec.q);
        fmt::print("AIC score: {:.4f}\n\n", result_aic->best_score);
    }

    // Select using AICc
    ModelSelector selector_aicc(SelectionCriterion::AICc);
    auto result_aicc = selector_aicc.select(data.data(), data.size(), candidates, false);

    if (result_aicc.has_value()) {
        const auto& best_spec_aicc = result_aicc->best_spec;
        fmt::print("AICc best model: ARIMA({},{},{})-GARCH({},{})\n", best_spec_aicc.arimaSpec.p,
                   best_spec_aicc.arimaSpec.d, best_spec_aicc.arimaSpec.q,
                   best_spec_aicc.garchSpec.p, best_spec_aicc.garchSpec.q);
        fmt::print("AICc score: {:.4f}\n\n", result_aicc->best_score);
    }

    // ========================================================================
    // Step 6: Summary
    // ========================================================================
    fmt::print("=== Summary ===\n\n");
    fmt::print("Model selection is a powerful tool for automatically choosing\n");
    fmt::print("the best ARIMA-GARCH specification from a set of candidates.\n\n");

    fmt::print("Key observations:\n");
    fmt::print("- BIC tends to favor simpler models (stronger penalty for complexity)\n");
    fmt::print("- AIC may select more complex models (weaker penalty)\n");
    fmt::print("- AICc is a corrected version of AIC for small samples\n\n");

    fmt::print("In this example with {} observations:\n", data.size());
    fmt::print("- True model was ARIMA({},{},{})-GARCH({},{})\n", true_spec.arimaSpec.p,
               true_spec.arimaSpec.d, true_spec.arimaSpec.q, true_spec.garchSpec.p,
               true_spec.garchSpec.q);
    fmt::print("- BIC selected ARIMA({},{},{})-GARCH({},{})\n", best_spec_bic.arimaSpec.p,
               best_spec_bic.arimaSpec.d, best_spec_bic.arimaSpec.q, best_spec_bic.garchSpec.p,
               best_spec_bic.garchSpec.q);

    if (result_aic.has_value()) {
        const auto& best_spec_aic = result_aic->best_spec;
        fmt::print("- AIC selected ARIMA({},{},{})-GARCH({},{})\n", best_spec_aic.arimaSpec.p,
                   best_spec_aic.arimaSpec.d, best_spec_aic.arimaSpec.q, best_spec_aic.garchSpec.p,
                   best_spec_aic.garchSpec.q);
    }

    if (result_aicc.has_value()) {
        const auto& best_spec_aicc = result_aicc->best_spec;
        fmt::print("- AICc selected ARIMA({},{},{})-GARCH({},{})\n", best_spec_aicc.arimaSpec.p,
                   best_spec_aicc.arimaSpec.d, best_spec_aicc.arimaSpec.q,
                   best_spec_aicc.garchSpec.p, best_spec_aicc.garchSpec.q);
    }

    fmt::print("\n✓ Example complete!\n");

    return 0;
}
