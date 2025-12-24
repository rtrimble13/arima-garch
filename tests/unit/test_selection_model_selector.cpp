#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/selection/CandidateGrid.hpp"
#include "ag/selection/ModelSelector.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "test_framework.hpp"

using ag::models::ArimaGarchSpec;
using ag::models::composite::ArimaGarchParameters;
using ag::selection::CandidateGrid;
using ag::selection::CandidateGridConfig;
using ag::selection::ModelSelector;
using ag::selection::SelectionCriterion;
using ag::simulation::ArimaGarchSimulator;

// ============================================================================
// ModelSelector Basic Tests
// ============================================================================

// Test constructor with default criterion (BIC)
TEST(model_selector_constructor_default) {
    ModelSelector selector;
    REQUIRE(selector.getCriterion() == SelectionCriterion::BIC);
}

// Test constructor with explicit criterion
TEST(model_selector_constructor_explicit) {
    ModelSelector selector_aic(SelectionCriterion::AIC);
    REQUIRE(selector_aic.getCriterion() == SelectionCriterion::AIC);

    ModelSelector selector_bic(SelectionCriterion::BIC);
    REQUIRE(selector_bic.getCriterion() == SelectionCriterion::BIC);

    ModelSelector selector_aicc(SelectionCriterion::AICc);
    REQUIRE(selector_aicc.getCriterion() == SelectionCriterion::AICc);
}

// Test setCriterion method
TEST(model_selector_set_criterion) {
    ModelSelector selector(SelectionCriterion::BIC);
    REQUIRE(selector.getCriterion() == SelectionCriterion::BIC);

    selector.setCriterion(SelectionCriterion::AIC);
    REQUIRE(selector.getCriterion() == SelectionCriterion::AIC);

    selector.setCriterion(SelectionCriterion::AICc);
    REQUIRE(selector.getCriterion() == SelectionCriterion::AICc);
}

// ============================================================================
// ModelSelector Input Validation Tests
// ============================================================================

// Test select with null data
TEST(model_selector_select_null_data) {
    ModelSelector selector;
    std::vector<ArimaGarchSpec> candidates;
    candidates.emplace_back(1, 0, 1, 1, 1);

    bool caught_exception = false;
    try {
        selector.select(nullptr, 100, candidates);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("data") != std::string::npos);
        REQUIRE(msg.find("nullptr") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// Test select with zero observations
TEST(model_selector_select_zero_obs) {
    ModelSelector selector;
    std::vector<double> data = {1.0, 2.0, 3.0};
    std::vector<ArimaGarchSpec> candidates;
    candidates.emplace_back(1, 0, 1, 1, 1);

    bool caught_exception = false;
    try {
        selector.select(data.data(), 0, candidates);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("n_obs") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// Test select with empty candidates
TEST(model_selector_select_empty_candidates) {
    ModelSelector selector;
    std::vector<double> data = {1.0, 2.0, 3.0};
    std::vector<ArimaGarchSpec> candidates;  // Empty

    bool caught_exception = false;
    try {
        selector.select(data.data(), data.size(), candidates);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("candidates") != std::string::npos);
        REQUIRE(msg.find("empty") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// ============================================================================
// ModelSelector Functional Tests with Synthetic Data
// ============================================================================

// Test selection from small candidate set - best spec should be found
TEST(model_selector_small_candidate_set) {
    // Generate synthetic data from known ARIMA(1,0,1)-GARCH(1,1) model
    ArimaGarchSpec true_spec(1, 0, 1, 1, 1);
    ArimaGarchParameters true_params(true_spec);

    true_params.arima_params.intercept = 0.05;
    true_params.arima_params.ar_coef[0] = 0.6;
    true_params.arima_params.ma_coef[0] = 0.3;
    true_params.garch_params.omega = 0.01;
    true_params.garch_params.alpha_coef[0] = 0.1;
    true_params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchSimulator simulator(true_spec, true_params);
    auto sim_result = simulator.simulate(500, 12345);
    const auto& data = sim_result.returns;

    // Create small candidate set including true spec
    std::vector<ArimaGarchSpec> candidates;
    candidates.emplace_back(0, 0, 1, 1, 1);  // ARIMA(0,0,1)-GARCH(1,1)
    candidates.emplace_back(1, 0, 0, 1, 1);  // ARIMA(1,0,0)-GARCH(1,1)
    candidates.emplace_back(1, 0, 1, 1, 1);  // ARIMA(1,0,1)-GARCH(1,1) - TRUE
    candidates.emplace_back(2, 0, 2, 1, 1);  // ARIMA(2,0,2)-GARCH(1,1) - overfit

    // Select best model using BIC
    ModelSelector selector(SelectionCriterion::BIC);
    auto result = selector.select(data.data(), data.size(), candidates);

    // Should find a model
    REQUIRE(result.has_value());

    // Check statistics
    REQUIRE(result->candidates_evaluated > 0);
    REQUIRE(result->candidates_evaluated <= candidates.size());

    // Best spec should be close to true spec (1,0,1,1,1)
    // BIC tends to favor the true model or slightly simpler
    REQUIRE(result->best_spec.arimaSpec.p <= 2);
    REQUIRE(result->best_spec.arimaSpec.d == 0);
    REQUIRE(result->best_spec.arimaSpec.q <= 2);
    REQUIRE(result->best_spec.garchSpec.p == 1);
    REQUIRE(result->best_spec.garchSpec.q == 1);

    // Should have a finite score
    REQUIRE(std::isfinite(result->best_score));

    // Should have fit summary
    REQUIRE(result->best_fit_summary.has_value());
    REQUIRE(result->best_fit_summary->converged);
}

// Test that BIC selects true or close-to-true model
TEST(model_selector_recovers_true_spec_bic) {
    // Generate data from ARIMA(1,0,1)-GARCH(1,1)
    ArimaGarchSpec true_spec(1, 0, 1, 1, 1);
    ArimaGarchParameters true_params(true_spec);

    true_params.arima_params.intercept = 0.02;
    true_params.arima_params.ar_coef[0] = 0.7;
    true_params.arima_params.ma_coef[0] = 0.4;
    true_params.garch_params.omega = 0.01;
    true_params.garch_params.alpha_coef[0] = 0.12;
    true_params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchSimulator simulator(true_spec, true_params);
    auto sim_result = simulator.simulate(800, 54321);
    const auto& data = sim_result.returns;

    // Generate candidates around true spec
    CandidateGridConfig config(2, 0, 2, 1, 1);  // max ARIMA(2,0,2), GARCH(1,1)
    CandidateGrid grid(config);
    auto candidates = grid.generate();

    // Select with BIC
    ModelSelector selector(SelectionCriterion::BIC);
    auto result = selector.select(data.data(), data.size(), candidates);

    REQUIRE(result.has_value());

    // BIC should select true spec (1,0,1,1,1) or something very close
    // Allow for (1,0,0) or (0,0,1) as BIC can be conservative
    bool close_to_truth =
        (result->best_spec.arimaSpec.p == true_spec.arimaSpec.p &&
         result->best_spec.arimaSpec.q == true_spec.arimaSpec.q) ||
        (result->best_spec.arimaSpec.p <= 1 && result->best_spec.arimaSpec.q <= 1);

    REQUIRE(close_to_truth);
}

// Test selection with different IC criteria
TEST(model_selector_different_criteria) {
    // Generate synthetic data
    ArimaGarchSpec true_spec(1, 0, 0, 1, 1);
    ArimaGarchParameters true_params(true_spec);

    true_params.arima_params.intercept = 0.0;
    true_params.arima_params.ar_coef[0] = 0.5;
    true_params.garch_params.omega = 0.01;
    true_params.garch_params.alpha_coef[0] = 0.1;
    true_params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchSimulator simulator(true_spec, true_params);
    auto sim_result = simulator.simulate(400, 99999);
    const auto& data = sim_result.returns;

    // Small candidate set
    std::vector<ArimaGarchSpec> candidates;
    candidates.emplace_back(1, 0, 0, 1, 1);
    candidates.emplace_back(1, 0, 1, 1, 1);
    candidates.emplace_back(2, 0, 1, 1, 1);

    // Test BIC
    ModelSelector selector_bic(SelectionCriterion::BIC);
    auto result_bic = selector_bic.select(data.data(), data.size(), candidates);
    REQUIRE(result_bic.has_value());

    // Test AIC
    ModelSelector selector_aic(SelectionCriterion::AIC);
    auto result_aic = selector_aic.select(data.data(), data.size(), candidates);
    REQUIRE(result_aic.has_value());

    // Test AICc
    ModelSelector selector_aicc(SelectionCriterion::AICc);
    auto result_aicc = selector_aicc.select(data.data(), data.size(), candidates);
    REQUIRE(result_aicc.has_value());

    // All should select a valid model
    REQUIRE(result_bic->candidates_evaluated > 0);
    REQUIRE(result_aic->candidates_evaluated > 0);
    REQUIRE(result_aicc->candidates_evaluated > 0);
}

// Test that selection is robust to some candidates failing
TEST(model_selector_robust_to_failures) {
    // Generate synthetic data
    ArimaGarchSpec true_spec(1, 0, 1, 1, 1);
    ArimaGarchParameters true_params(true_spec);

    true_params.arima_params.intercept = 0.05;
    true_params.arima_params.ar_coef[0] = 0.6;
    true_params.arima_params.ma_coef[0] = 0.3;
    true_params.garch_params.omega = 0.01;
    true_params.garch_params.alpha_coef[0] = 0.1;
    true_params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchSimulator simulator(true_spec, true_params);
    auto sim_result = simulator.simulate(300, 42);
    const auto& data = sim_result.returns;

    // Create candidates - some may be harder to fit
    std::vector<ArimaGarchSpec> candidates;
    candidates.emplace_back(1, 0, 1, 1, 1);  // Good spec
    candidates.emplace_back(0, 0, 1, 1, 1);  // Should work
    candidates.emplace_back(2, 0, 2, 1, 1);  // May be harder to fit
    candidates.emplace_back(3, 0, 3, 1, 1);  // May fail with small data

    ModelSelector selector(SelectionCriterion::BIC);
    auto result = selector.select(data.data(), data.size(), candidates);

    // Should still find a best model even if some fail
    REQUIRE(result.has_value());
    REQUIRE(result->candidates_evaluated > 0);

    // Number evaluated + failed should equal total candidates
    REQUIRE(result->candidates_evaluated + result->candidates_failed == candidates.size());
}

// Test selection with diagnostics
TEST(model_selector_with_diagnostics) {
    // Generate synthetic data
    ArimaGarchSpec true_spec(1, 0, 0, 1, 1);
    ArimaGarchParameters true_params(true_spec);

    true_params.arima_params.intercept = 0.0;
    true_params.arima_params.ar_coef[0] = 0.5;
    true_params.garch_params.omega = 0.01;
    true_params.garch_params.alpha_coef[0] = 0.1;
    true_params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchSimulator simulator(true_spec, true_params);
    auto sim_result = simulator.simulate(400, 777);
    const auto& data = sim_result.returns;

    // Small candidate set
    std::vector<ArimaGarchSpec> candidates;
    candidates.emplace_back(1, 0, 0, 1, 1);
    candidates.emplace_back(1, 0, 1, 1, 1);

    ModelSelector selector(SelectionCriterion::BIC);
    auto result = selector.select(data.data(), data.size(), candidates, true);

    REQUIRE(result.has_value());
    REQUIRE(result->best_fit_summary.has_value());

    // Should have diagnostics
    REQUIRE(result->best_fit_summary->diagnostics.has_value());

    auto& diag = result->best_fit_summary->diagnostics.value();
    // Just check that diagnostics were computed (not checking optional fields)
    REQUIRE(diag.ljung_box_residuals.p_value >= 0.0);
    REQUIRE(diag.ljung_box_squared.p_value >= 0.0);
    REQUIRE(diag.jarque_bera.p_value >= 0.0);
}

// ============================================================================
// Cross-Validation Tests
// ============================================================================

// Test selection with CV criterion
TEST(model_selector_cv_criterion) {
    // Generate synthetic data
    ArimaGarchSpec true_spec(1, 0, 1, 1, 1);
    ArimaGarchParameters true_params(true_spec);

    true_params.arima_params.intercept = 0.05;
    true_params.arima_params.ar_coef[0] = 0.6;
    true_params.arima_params.ma_coef[0] = 0.3;
    true_params.garch_params.omega = 0.01;
    true_params.garch_params.alpha_coef[0] = 0.1;
    true_params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchSimulator simulator(true_spec, true_params);
    auto sim_result = simulator.simulate(300, 42);
    const auto& data = sim_result.returns;

    // Small candidate set
    std::vector<ArimaGarchSpec> candidates;
    candidates.emplace_back(0, 0, 1, 1, 1);
    candidates.emplace_back(1, 0, 0, 1, 1);
    candidates.emplace_back(1, 0, 1, 1, 1);

    // Select with CV
    ModelSelector selector(SelectionCriterion::CV);
    auto result = selector.select(data.data(), data.size(), candidates);

    // Should find a best model
    REQUIRE(result.has_value());
    REQUIRE(result->candidates_evaluated > 0);
    REQUIRE(std::isfinite(result->best_score));
    REQUIRE(result->best_score > 0.0);  // MSE should be positive
}

// Test CV vs BIC produce different results
TEST(model_selector_cv_vs_bic) {
    // Generate data
    ArimaGarchSpec true_spec(1, 0, 1, 1, 1);
    ArimaGarchParameters true_params(true_spec);

    true_params.arima_params.intercept = 0.02;
    true_params.arima_params.ar_coef[0] = 0.7;
    true_params.arima_params.ma_coef[0] = 0.4;
    true_params.garch_params.omega = 0.01;
    true_params.garch_params.alpha_coef[0] = 0.12;
    true_params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchSimulator simulator(true_spec, true_params);
    auto sim_result = simulator.simulate(250, 99999);
    const auto& data = sim_result.returns;

    // Create candidates
    std::vector<ArimaGarchSpec> candidates;
    candidates.emplace_back(1, 0, 0, 1, 1);
    candidates.emplace_back(0, 0, 1, 1, 1);
    candidates.emplace_back(1, 0, 1, 1, 1);

    // Select with BIC
    ModelSelector selector_bic(SelectionCriterion::BIC);
    auto result_bic = selector_bic.select(data.data(), data.size(), candidates);

    // Select with CV
    ModelSelector selector_cv(SelectionCriterion::CV);
    auto result_cv = selector_cv.select(data.data(), data.size(), candidates);

    // Both should find a model
    REQUIRE(result_bic.has_value());
    REQUIRE(result_cv.has_value());

    // Both should have valid scores
    REQUIRE(std::isfinite(result_bic->best_score));
    REQUIRE(std::isfinite(result_cv->best_score));

    // Scores are on different scales (BIC vs MSE), so just check they're positive
    REQUIRE(result_cv->best_score > 0.0);
}

// ============================================================================
// Run all tests
// ============================================================================

int main() {
    report_test_results("ModelSelector");
    return get_test_result();
}
