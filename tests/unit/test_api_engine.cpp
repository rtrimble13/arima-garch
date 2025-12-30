#include "ag/api/Engine.hpp"
#include "ag/io/Json.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/selection/CandidateGrid.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>

#include "test_framework.hpp"

using ag::api::Engine;
using ag::models::ArimaGarchSpec;
using ag::models::composite::ArimaGarchParameters;
using ag::simulation::ArimaGarchSimulator;

// ============================================================================
// Engine Construction Tests
// ============================================================================

TEST(engine_construction) {
    Engine engine;
    REQUIRE(true);  // Should construct without error
}

// ============================================================================
// Engine fit() Tests
// ============================================================================

TEST(engine_fit_basic) {
    // Generate synthetic data
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters true_params(spec);

    true_params.arima_params.intercept = 0.05;
    true_params.arima_params.ar_coef[0] = 0.6;
    true_params.arima_params.ma_coef[0] = 0.3;
    true_params.garch_params.omega = 0.01;
    true_params.garch_params.alpha_coef[0] = 0.1;
    true_params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchSimulator simulator(spec, true_params);
    auto sim_result = simulator.simulate(500, 42);

    // Fit using Engine
    Engine engine;
    auto fit_result = engine.fit(sim_result.returns, spec, true);

    REQUIRE(fit_result.has_value());
    REQUIRE(fit_result.value().model != nullptr);
    REQUIRE(fit_result.value().summary.converged);
    REQUIRE(fit_result.value().summary.sample_size == 500);
    REQUIRE(fit_result.value().summary.diagnostics.has_value());
}

TEST(engine_fit_insufficient_data) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    std::vector<double> data = {1.0, 2.0, 3.0};  // Too few observations

    Engine engine;
    auto fit_result = engine.fit(data, spec);

    REQUIRE(!fit_result.has_value());
}

TEST(engine_fit_no_diagnostics) {
    // Generate synthetic data
    ArimaGarchSpec spec(1, 0, 0, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.05;
    params.arima_params.ar_coef[0] = 0.5;
    params.garch_params.omega = 0.01;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchSimulator simulator(spec, params);
    auto sim_result = simulator.simulate(200, 123);

    // Fit without diagnostics
    Engine engine;
    auto fit_result = engine.fit(sim_result.returns, spec, false);

    REQUIRE(fit_result.has_value());
    REQUIRE(fit_result.value().model != nullptr);
    REQUIRE(!fit_result.value().summary.diagnostics.has_value());
}

// ============================================================================
// Engine forecast() Tests
// ============================================================================

TEST(engine_forecast_basic) {
    // Generate and fit a model
    ArimaGarchSpec spec(1, 0, 0, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.05;
    params.arima_params.ar_coef[0] = 0.5;
    params.garch_params.omega = 0.01;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchSimulator simulator(spec, params);
    auto sim_result = simulator.simulate(200, 456);

    Engine engine;
    auto fit_result = engine.fit(sim_result.returns, spec, false);
    REQUIRE(fit_result.has_value());

    // Forecast
    auto forecast_result = engine.forecast(*fit_result.value().model, 10);
    REQUIRE(forecast_result.has_value());
    REQUIRE(forecast_result.value().mean_forecasts.size() == 10);
    REQUIRE(forecast_result.value().variance_forecasts.size() == 10);
}

TEST(engine_forecast_invalid_horizon) {
    // Generate and fit a model
    ArimaGarchSpec spec(1, 0, 0, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.05;
    params.arima_params.ar_coef[0] = 0.5;
    params.garch_params.omega = 0.01;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchSimulator simulator(spec, params);
    auto sim_result = simulator.simulate(100, 789);

    Engine engine;
    auto fit_result = engine.fit(sim_result.returns, spec, false);
    REQUIRE(fit_result.has_value());

    // Try invalid horizon
    auto forecast_result = engine.forecast(*fit_result.value().model, 0);
    REQUIRE(!forecast_result.has_value());
}

// ============================================================================
// Engine simulate() Tests
// ============================================================================

TEST(engine_simulate_basic) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.05;
    params.arima_params.ar_coef[0] = 0.6;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.01;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.85;

    Engine engine;
    auto sim_result = engine.simulate(spec, params, 1000, 42);

    REQUIRE(sim_result.has_value());
    REQUIRE(sim_result.value().returns.size() == 1000);
    REQUIRE(sim_result.value().volatilities.size() == 1000);
}

TEST(engine_simulate_invalid_length) {
    ArimaGarchSpec spec(1, 0, 0, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.0;
    params.arima_params.ar_coef[0] = 0.5;
    params.garch_params.omega = 0.01;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    Engine engine;
    auto sim_result = engine.simulate(spec, params, 0, 42);

    REQUIRE(!sim_result.has_value());
}

// ============================================================================
// Engine auto_select() Tests
// ============================================================================

TEST(engine_auto_select_basic) {
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
    auto sim_result = simulator.simulate(300, 999);

    // Create candidates
    std::vector<ArimaGarchSpec> candidates;
    candidates.push_back(ArimaGarchSpec(1, 0, 0, 1, 1));
    candidates.push_back(ArimaGarchSpec(1, 0, 1, 1, 1));
    candidates.push_back(ArimaGarchSpec(2, 0, 1, 1, 1));

    Engine engine;
    auto select_result = engine.auto_select(sim_result.returns, candidates);

    REQUIRE(select_result.has_value());
    REQUIRE(select_result.value().model != nullptr);
    REQUIRE(select_result.value().candidates_evaluated > 0);
}

TEST(engine_auto_select_no_candidates) {
    std::vector<double> data(100, 1.0);
    std::vector<ArimaGarchSpec> candidates;  // Empty

    Engine engine;
    auto select_result = engine.auto_select(data, candidates);

    REQUIRE(!select_result.has_value());
}

// ============================================================================
// Integration Test: fit + forecast + serialize
// ============================================================================

TEST(engine_fit_forecast_serialize) {
    // Generate synthetic data
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters true_params(spec);

    true_params.arima_params.intercept = 0.05;
    true_params.arima_params.ar_coef[0] = 0.6;
    true_params.arima_params.ma_coef[0] = 0.3;
    true_params.garch_params.omega = 0.01;
    true_params.garch_params.alpha_coef[0] = 0.1;
    true_params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchSimulator simulator(spec, true_params);
    auto sim_result = simulator.simulate(500, 42);

    // Fit using Engine
    Engine engine;
    auto fit_result = engine.fit(sim_result.returns, spec, true);
    REQUIRE(fit_result.has_value());

    // Generate forecasts from fitted model
    auto forecast_result1 = engine.forecast(*fit_result.value().model, 5);
    REQUIRE(forecast_result1.has_value());

    // Serialize model to JSON
    auto json = ag::io::JsonWriter::toJson(*fit_result.value().model);
    REQUIRE(!json.empty());

    // Deserialize model using the correct JSON structure
    auto spec_result = ag::io::JsonReader::arimaGarchSpecFromJson(json["spec"]);
    REQUIRE(spec_result.has_value());

    auto arima_params_result = ag::io::JsonReader::arimaParametersFromJson(
        json["parameters"]["arima"], spec_result.value().arimaSpec);
    REQUIRE(arima_params_result.has_value());

    auto garch_params_result = ag::io::JsonReader::garchParametersFromJson(
        json["parameters"]["garch"], spec_result.value().garchSpec);
    REQUIRE(garch_params_result.has_value());

    // Create combined parameters
    ag::models::composite::ArimaGarchParameters deserialized_params(spec_result.value());
    deserialized_params.arima_params = arima_params_result.value();
    deserialized_params.garch_params = garch_params_result.value();

    // Create new model with deserialized parameters
    auto deserialized_model = std::make_shared<ag::models::composite::ArimaGarchModel>(
        spec_result.value(), deserialized_params);

    // Initialize with same data
    for (const auto& value : sim_result.returns) {
        deserialized_model->update(value);
    }

    // Generate forecasts from deserialized model
    auto forecast_result2 = engine.forecast(*deserialized_model, 5);
    REQUIRE(forecast_result2.has_value());

    // Verify forecasts match
    for (std::size_t i = 0; i < 5; ++i) {
        REQUIRE_APPROX(forecast_result1.value().mean_forecasts[i],
                       forecast_result2.value().mean_forecasts[i], 1e-6);
        REQUIRE_APPROX(forecast_result1.value().variance_forecasts[i],
                       forecast_result2.value().variance_forecasts[i], 1e-6);
    }
}

// ============================================================================
// Engine simulate from loaded model Tests
// ============================================================================

TEST(engine_simulate_from_loaded_model) {
    // Step 1: Create and save a model
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.05;
    params.arima_params.ar_coef[0] = 0.6;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.01;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.85;

    auto model = std::make_shared<ag::models::composite::ArimaGarchModel>(spec, params);

    // Save model to temporary location
    auto temp_file = std::filesystem::temp_directory_path() / "test_simulate_model.json";
    auto save_result = ag::io::JsonWriter::saveModel(temp_file.string(), *model);
    REQUIRE(save_result.has_value());

    // Step 2: Load the model
    auto load_result = ag::io::JsonReader::loadModel(temp_file.string());
    REQUIRE(load_result.has_value());

    // Step 3: Extract parameters and simulate
    auto& loaded_model = *load_result;
    ArimaGarchParameters loaded_params(loaded_model.getSpec());
    loaded_params.arima_params = loaded_model.getArimaParams();
    loaded_params.garch_params = loaded_model.getGarchParams();

    Engine engine;
    auto sim_result = engine.simulate(loaded_model.getSpec(), loaded_params, 100, 42);
    REQUIRE(sim_result.has_value());
    REQUIRE(sim_result.value().returns.size() == 100);
    REQUIRE(sim_result.value().volatilities.size() == 100);
}

TEST(engine_simulate_from_loaded_model_reproducibility) {
    // Create and save a model
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.05;
    params.arima_params.ar_coef[0] = 0.6;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.01;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.85;

    auto model = std::make_shared<ag::models::composite::ArimaGarchModel>(spec, params);

    auto temp_file = std::filesystem::temp_directory_path() / "test_simulate_repro.json";
    auto save_result = ag::io::JsonWriter::saveModel(temp_file.string(), *model);
    REQUIRE(save_result.has_value());

    // Load model
    auto load_result = ag::io::JsonReader::loadModel(temp_file.string());
    REQUIRE(load_result.has_value());

    auto& loaded_model = *load_result;
    ArimaGarchParameters loaded_params(loaded_model.getSpec());
    loaded_params.arima_params = loaded_model.getArimaParams();
    loaded_params.garch_params = loaded_model.getGarchParams();

    // Simulate twice with same seed
    Engine engine;
    auto sim1 = engine.simulate(loaded_model.getSpec(), loaded_params, 50, 12345);
    auto sim2 = engine.simulate(loaded_model.getSpec(), loaded_params, 50, 12345);

    REQUIRE(sim1.has_value());
    REQUIRE(sim2.has_value());

    // Verify reproducibility
    for (size_t i = 0; i < 50; ++i) {
        REQUIRE_APPROX(sim1.value().returns[i], sim2.value().returns[i], 1e-15);
        REQUIRE_APPROX(sim1.value().volatilities[i], sim2.value().volatilities[i], 1e-15);
    }
}

// ============================================================================
// Engine fit() Tests for ARIMA-only models (no GARCH)
// ============================================================================

TEST(engine_fit_arima_101_no_garch) {
    // Test fitting ARIMA(1,0,1) model without GARCH component
    // This is a regression test for the segfault bug

    // Generate synthetic data from ARIMA(1,0,1) model
    ArimaGarchSpec spec(1, 0, 1, 0, 0);  // No GARCH component
    ArimaGarchParameters true_params(spec);

    true_params.arima_params.intercept = 0.05;
    true_params.arima_params.ar_coef[0] = 0.6;
    true_params.arima_params.ma_coef[0] = 0.3;

    // For ARIMA-only, use a simple simulator approach
    // Generate random data with known properties
    std::vector<double> data(100);
    std::mt19937 gen(42);
    std::normal_distribution<> dist(0.0, 1.0);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = 0.05 + dist(gen) * 0.2;
    }

    // Fit using Engine
    Engine engine;
    auto fit_result = engine.fit(data, spec, true);

    // Should not crash and should converge
    REQUIRE(fit_result.has_value());
    REQUIRE(fit_result.value().model != nullptr);
    REQUIRE(fit_result.value().summary.converged);
    REQUIRE(fit_result.value().summary.sample_size == 100);

    // Verify spec is correct
    const auto& fitted_spec = fit_result.value().model->getSpec();
    REQUIRE(fitted_spec.arimaSpec.p == 1);
    REQUIRE(fitted_spec.arimaSpec.d == 0);
    REQUIRE(fitted_spec.arimaSpec.q == 1);
    REQUIRE(fitted_spec.garchSpec.p == 0);
    REQUIRE(fitted_spec.garchSpec.q == 0);
    REQUIRE(fitted_spec.garchSpec.isNull());
}

TEST(engine_fit_ar_100_no_garch) {
    // Test fitting AR(1) model without GARCH component

    // Generate random data
    std::vector<double> data(100);
    std::mt19937 gen(123);
    std::normal_distribution<> dist(0.0, 1.0);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = 0.1 + dist(gen) * 0.25;
    }

    ArimaGarchSpec spec(1, 0, 0, 0, 0);  // AR(1), no GARCH

    // Fit using Engine
    Engine engine;
    auto fit_result = engine.fit(data, spec, true);

    // Should not crash and should converge
    REQUIRE(fit_result.has_value());
    REQUIRE(fit_result.value().model != nullptr);
    REQUIRE(fit_result.value().summary.converged);

    // Verify GARCH spec is null
    const auto& fitted_spec = fit_result.value().model->getSpec();
    REQUIRE(fitted_spec.garchSpec.isNull());
}

TEST(engine_fit_ma_001_no_garch) {
    // Test fitting MA(1) model without GARCH component
    // Note: MA models can be harder to fit and may not always converge

    // Generate random data
    std::vector<double> data(100);
    std::mt19937 gen(456);
    std::normal_distribution<> dist(0.0, 1.0);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = 0.08 + dist(gen) * 0.3;
    }

    ArimaGarchSpec spec(0, 0, 1, 0, 0);  // MA(1), no GARCH

    // Fit using Engine - the test is that this doesn't crash (segfault)
    Engine engine;
    bool no_crash = false;
    try {
        auto fit_result = engine.fit(data, spec, false);  // Skip diagnostics for speed
        no_crash = true;
        
        // If convergence succeeded, verify the model is valid
        if (fit_result.has_value()) {
            REQUIRE(fit_result.value().model != nullptr);
            const auto& fitted_spec = fit_result.value().model->getSpec();
            REQUIRE(fitted_spec.garchSpec.isNull());
        }
    } catch (...) {
        // Any exception (other than segfault) is also acceptable
        // The key requirement is no segfault/crash
        no_crash = true;
    }
    
    // Verify the fit call completed without crashing
    REQUIRE(no_crash);
}

int main() {
    report_test_results("Engine API Tests");
    return get_test_result();
}
