/**
 * @file example_cross_validation.cpp
 * @brief Demonstrates model selection using cross-validation scoring.
 *
 * This example shows how to:
 * 1. Generate synthetic data from a known ARIMA-GARCH model
 * 2. Create a grid of candidate model specifications
 * 3. Use ModelSelector with CV criterion for out-of-sample selection
 * 4. Compare CV-based selection with IC-based selection
 */

#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/selection/CandidateGrid.hpp"
#include "ag/selection/ModelSelector.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"

#include <iostream>

#include <fmt/core.h>

using ag::models::ArimaGarchSpec;
using ag::models::composite::ArimaGarchParameters;
using ag::selection::CandidateGrid;
using ag::selection::CandidateGridConfig;
using ag::selection::ModelSelector;
using ag::selection::SelectionCriterion;
using ag::simulation::ArimaGarchSimulator;

int main() {
    fmt::print("=== ARIMA-GARCH Cross-Validation Selection Example ===\n\n");

    // ========================================================================
    // Step 1: Generate synthetic data
    // ========================================================================
    fmt::print("Step 1: Generating synthetic data...\n");

    // True model: ARIMA(1,0,1)-GARCH(1,1)
    ArimaGarchSpec true_spec(1, 0, 1, 1, 1);
    ArimaGarchParameters true_params(true_spec);

    true_params.arima_params.intercept = 0.05;
    true_params.arima_params.ar_coef[0] = 0.6;
    true_params.arima_params.ma_coef[0] = 0.3;
    true_params.garch_params.omega = 0.01;
    true_params.garch_params.alpha_coef[0] = 0.1;
    true_params.garch_params.beta_coef[0] = 0.85;

    fmt::print("  True model: ARIMA({},{},{})-GARCH({},{})\n", true_spec.arimaSpec.p,
               true_spec.arimaSpec.d, true_spec.arimaSpec.q, true_spec.garchSpec.p,
               true_spec.garchSpec.q);

    ArimaGarchSimulator simulator(true_spec, true_params);
    auto sim_result = simulator.simulate(500, 42);
    const auto& data = sim_result.returns;

    fmt::print("  Generated {} observations\n\n", data.size());

    // ========================================================================
    // Step 2: Create candidate grid
    // ========================================================================
    fmt::print("Step 2: Creating candidate grid...\n");

    // Small search space for demonstration
    CandidateGridConfig config(2, 0, 2, 1, 1);  // ARIMA(0-2,0,0-2), GARCH(1,1)

    CandidateGrid grid(config);
    auto candidates = grid.generate();

    fmt::print("  Generated {} candidate specifications\n", candidates.size());
    fmt::print("  Search space: ARIMA(0-{},0,0-{})-GARCH({},{})\n\n", config.max_p, config.max_q,
               config.max_p_garch, config.max_q_garch);

    // ========================================================================
    // Step 3: Select best model using Cross-Validation
    // ========================================================================
    fmt::print("Step 3: Selecting best model using Cross-Validation...\n");
    fmt::print("  CV uses rolling origin with 1-step-ahead MSE scoring\n");
    fmt::print("  This is more computationally expensive but provides\n");
    fmt::print("  better assessment of out-of-sample forecast performance\n\n");

    ModelSelector selector_cv(SelectionCriterion::CV);
    auto result_cv = selector_cv.select(data.data(), data.size(), candidates);

    if (!result_cv.has_value()) {
        fmt::print("  ERROR: CV selection failed (all candidates failed)\n");
        return 1;
    }

    fmt::print("  Selection complete!\n");
    fmt::print("  Candidates evaluated: {}\n", result_cv->candidates_evaluated);
    fmt::print("  Candidates failed: {}\n\n", result_cv->candidates_failed);

    // ========================================================================
    // Step 4: Display CV results
    // ========================================================================
    fmt::print("=== Cross-Validation Results ===\n\n");

    const auto& best_spec_cv = result_cv->best_spec;
    fmt::print("Best model: ARIMA({},{},{})-GARCH({},{})\n", best_spec_cv.arimaSpec.p,
               best_spec_cv.arimaSpec.d, best_spec_cv.arimaSpec.q, best_spec_cv.garchSpec.p,
               best_spec_cv.garchSpec.q);
    fmt::print("CV MSE score: {:.6f}\n\n", result_cv->best_score);

    bool correct_spec = (best_spec_cv.arimaSpec.p == true_spec.arimaSpec.p &&
                         best_spec_cv.arimaSpec.d == true_spec.arimaSpec.d &&
                         best_spec_cv.arimaSpec.q == true_spec.arimaSpec.q &&
                         best_spec_cv.garchSpec.p == true_spec.garchSpec.p &&
                         best_spec_cv.garchSpec.q == true_spec.garchSpec.q);

    if (correct_spec) {
        fmt::print("✓ CV correctly identified the true model specification!\n\n");
    } else {
        fmt::print("✓ CV selected a different specification\n");
        fmt::print("  (This can happen due to sample variation)\n\n");
    }

    // ========================================================================
    // Step 5: Compare with BIC selection
    // ========================================================================
    fmt::print("=== Comparison: CV vs BIC ===\n\n");

    ModelSelector selector_bic(SelectionCriterion::BIC);
    auto result_bic = selector_bic.select(data.data(), data.size(), candidates);

    if (result_bic.has_value()) {
        const auto& best_spec_bic = result_bic->best_spec;
        fmt::print("BIC best model:  ARIMA({},{},{})-GARCH({},{})\n", best_spec_bic.arimaSpec.p,
                   best_spec_bic.arimaSpec.d, best_spec_bic.arimaSpec.q, best_spec_bic.garchSpec.p,
                   best_spec_bic.garchSpec.q);
        fmt::print("BIC score:       {:.4f}\n\n", result_bic->best_score);

        fmt::print("CV best model:   ARIMA({},{},{})-GARCH({},{})\n", best_spec_cv.arimaSpec.p,
                   best_spec_cv.arimaSpec.d, best_spec_cv.arimaSpec.q, best_spec_cv.garchSpec.p,
                   best_spec_cv.garchSpec.q);
        fmt::print("CV MSE score:    {:.6f}\n\n", result_cv->best_score);

        bool same_model = (best_spec_cv.arimaSpec.p == best_spec_bic.arimaSpec.p &&
                           best_spec_cv.arimaSpec.q == best_spec_bic.arimaSpec.q);

        if (same_model) {
            fmt::print("✓ Both criteria selected the same model\n\n");
        } else {
            fmt::print("! Different models selected by CV and BIC\n");
            fmt::print("  BIC optimizes in-sample fit with complexity penalty\n");
            fmt::print("  CV optimizes out-of-sample forecast performance\n\n");
        }
    }

    // ========================================================================
    // Step 6: Summary
    // ========================================================================
    fmt::print("=== Summary ===\n\n");

    fmt::print("Cross-validation (CV) provides an alternative to information\n");
    fmt::print("criteria (AIC/BIC/AICc) for model selection. Key differences:\n\n");

    fmt::print("Information Criteria (BIC/AIC/AICc):\n");
    fmt::print("  + Fast: Single model fit per candidate\n");
    fmt::print("  + Theoretical foundation in model comparison\n");
    fmt::print("  - Approximates out-of-sample performance\n\n");

    fmt::print("Cross-Validation:\n");
    fmt::print("  + Direct measure of out-of-sample forecast accuracy\n");
    fmt::print("  + No assumptions about model complexity penalty\n");
    fmt::print("  - Computationally expensive: Multiple fits per candidate\n");
    fmt::print("  - Can be unstable with small samples\n\n");

    fmt::print("Recommendation: Use BIC/AIC for exploratory analysis and\n");
    fmt::print("large candidate sets. Use CV when forecast performance is\n");
    fmt::print("critical and computational resources allow.\n\n");

    fmt::print("✓ Example complete!\n");

    return 0;
}
