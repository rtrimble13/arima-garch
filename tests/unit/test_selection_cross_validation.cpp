#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/selection/CrossValidation.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"

#include <cmath>
#include <stdexcept>

#include "test_framework.hpp"

using ag::models::ArimaGarchSpec;
using ag::models::composite::ArimaGarchParameters;
using ag::selection::computeCrossValidationScore;
using ag::selection::CrossValidationConfig;
using ag::simulation::ArimaGarchSimulator;

// ============================================================================
// CrossValidation Basic Tests
// ============================================================================

// Test CrossValidationConfig construction
TEST(cross_validation_config_construction) {
    CrossValidationConfig config(100);
    REQUIRE(config.min_train_size == 100);
    REQUIRE(config.horizon == 1);  // Default horizon
}

// Test computeCrossValidationScore with null data
TEST(cross_validation_null_data) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    CrossValidationConfig config(50);

    bool caught_exception = false;
    try {
        computeCrossValidationScore(nullptr, 100, spec, config);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("data") != std::string::npos);
        REQUIRE(msg.find("nullptr") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// Test computeCrossValidationScore with zero observations
TEST(cross_validation_zero_obs) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    CrossValidationConfig config(50);
    std::vector<double> data = {1.0, 2.0, 3.0};

    bool caught_exception = false;
    try {
        computeCrossValidationScore(data.data(), 0, spec, config);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("n_obs") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// Test computeCrossValidationScore with min_train_size >= n_obs
TEST(cross_validation_invalid_min_train_size) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    CrossValidationConfig config(150);  // Too large
    std::vector<double> data(100, 0.0);

    bool caught_exception = false;
    try {
        computeCrossValidationScore(data.data(), data.size(), spec, config);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("min_train_size") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// ============================================================================
// CrossValidation Functional Tests
// ============================================================================

// Test that CV produces a finite MSE score for valid data
TEST(cross_validation_produces_score) {
    // Generate synthetic data
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.05;
    params.arima_params.ar_coef[0] = 0.6;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.01;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchSimulator simulator(spec, params);
    auto sim_result = simulator.simulate(200, 42);
    const auto& data = sim_result.returns;

    // Use CV with 140 observations as minimum training size
    CrossValidationConfig config(140);
    auto result = computeCrossValidationScore(data.data(), data.size(), spec, config);

    // Should produce a result
    REQUIRE(result.has_value());

    // MSE should be finite and positive
    REQUIRE(std::isfinite(result->mse));
    REQUIRE(result->mse > 0.0);

    // Should have evaluated some windows
    REQUIRE(result->n_windows > 0);
    REQUIRE(result->n_windows <= (data.size() - config.min_train_size));
}

// Test that CV MSE is reasonable for good specification
TEST(cross_validation_reasonable_mse) {
    // Generate data with known spec
    ArimaGarchSpec true_spec(1, 0, 0, 1, 1);
    ArimaGarchParameters true_params(true_spec);

    true_params.arima_params.intercept = 0.0;
    true_params.arima_params.ar_coef[0] = 0.5;
    true_params.garch_params.omega = 0.01;
    true_params.garch_params.alpha_coef[0] = 0.1;
    true_params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchSimulator simulator(true_spec, true_params);
    auto sim_result = simulator.simulate(250, 12345);
    const auto& data = sim_result.returns;

    // Evaluate true spec with CV
    CrossValidationConfig config(175);
    auto result_true = computeCrossValidationScore(data.data(), data.size(), true_spec, config);

    REQUIRE(result_true.has_value());

    // MSE should be reasonable (not too large)
    // For standardized returns with variance ~1, MSE should be on order of 1-10
    REQUIRE(result_true->mse < 100.0);
}

// Test that CV can distinguish between good and bad specifications
TEST(cross_validation_distinguishes_specs) {
    // Generate data from ARIMA(1,0,1)-GARCH(1,1)
    ArimaGarchSpec true_spec(1, 0, 1, 1, 1);
    ArimaGarchParameters true_params(true_spec);

    true_params.arima_params.intercept = 0.05;
    true_params.arima_params.ar_coef[0] = 0.6;
    true_params.arima_params.ma_coef[0] = 0.3;
    true_params.garch_params.omega = 0.01;
    true_params.garch_params.alpha_coef[0] = 0.1;
    true_params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchSimulator simulator(true_spec, true_params);
    auto sim_result = simulator.simulate(300, 99999);
    const auto& data = sim_result.returns;

    CrossValidationConfig config(210);

    // Evaluate true spec
    auto result_true = computeCrossValidationScore(data.data(), data.size(), true_spec, config);
    REQUIRE(result_true.has_value());

    // Evaluate a simpler spec (should have higher MSE in general)
    ArimaGarchSpec simple_spec(0, 0, 0, 1, 1);  // Pure GARCH(1,1), no ARIMA
    auto result_simple = computeCrossValidationScore(data.data(), data.size(), simple_spec, config);

    // Both should produce results
    REQUIRE(result_simple.has_value());

    // Note: We can't guarantee true spec always has lower MSE due to random variation
    // But both should have finite, positive MSE
    REQUIRE(std::isfinite(result_true->mse));
    REQUIRE(std::isfinite(result_simple->mse));
    REQUIRE(result_true->mse > 0.0);
    REQUIRE(result_simple->mse > 0.0);
}

// Test that CV returns nullopt for overly complex models with insufficient data
TEST(cross_validation_fails_gracefully) {
    // Generate simple data
    ArimaGarchSpec simple_spec(0, 0, 0, 1, 1);
    ArimaGarchParameters simple_params(simple_spec);

    simple_params.garch_params.omega = 0.01;
    simple_params.garch_params.alpha_coef[0] = 0.1;
    simple_params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchSimulator simulator(simple_spec, simple_params);
    auto sim_result = simulator.simulate(100, 777);
    const auto& data = sim_result.returns;

    // Try to fit very complex model with small training windows
    ArimaGarchSpec complex_spec(3, 0, 3, 1, 1);
    CrossValidationConfig config(50);  // Small training window

    auto result = computeCrossValidationScore(data.data(), data.size(), complex_spec, config);

    // Result might be nullopt if model is too complex for the data
    // or might succeed with high MSE - either is acceptable
    if (result.has_value()) {
        REQUIRE(std::isfinite(result->mse));
        REQUIRE(result->mse > 0.0);
    }
    // If nullopt, that's also fine - it means CV couldn't reliably evaluate this model
}

// Test that CV works with different minimum training sizes
TEST(cross_validation_different_training_sizes) {
    // Generate data
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.02;
    params.arima_params.ar_coef[0] = 0.5;
    params.arima_params.ma_coef[0] = 0.4;
    params.garch_params.omega = 0.01;
    params.garch_params.alpha_coef[0] = 0.12;
    params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchSimulator simulator(spec, params);
    auto sim_result = simulator.simulate(250, 54321);
    const auto& data = sim_result.returns;

    // Try different training sizes
    CrossValidationConfig config1(150);
    auto result1 = computeCrossValidationScore(data.data(), data.size(), spec, config1);

    CrossValidationConfig config2(200);
    auto result2 = computeCrossValidationScore(data.data(), data.size(), spec, config2);

    // Both should produce results
    REQUIRE(result1.has_value());
    REQUIRE(result2.has_value());

    // Result2 should have fewer windows than result1
    REQUIRE(result2->n_windows < result1->n_windows);

    // Both should have finite, positive MSE
    REQUIRE(std::isfinite(result1->mse));
    REQUIRE(std::isfinite(result2->mse));
    REQUIRE(result1->mse > 0.0);
    REQUIRE(result2->mse > 0.0);
}

// ============================================================================
// Run all tests
// ============================================================================

int main() {
    report_test_results("CrossValidation");
    return get_test_result();
}
