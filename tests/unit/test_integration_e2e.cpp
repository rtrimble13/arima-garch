#include "ag/api/Engine.hpp"
#include "ag/io/CsvReader.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"

#include <filesystem>

#include "test_framework.hpp"

using ag::api::Engine;
using ag::io::CsvReader;
using ag::io::CsvReaderOptions;
using ag::models::ArimaGarchSpec;
using ag::models::composite::ArimaGarchParameters;
using ag::simulation::ArimaGarchSimulator;

// ============================================================================
// End-to-End Integration Test: fit → forecast → simulate
// ============================================================================

TEST(integration_e2e_fit_forecast_simulate) {
    // Step 1: Load data from tiny_returns.csv fixture
    std::filesystem::path fixture_path = "../../tests/fixtures/tiny_returns.csv";

    CsvReaderOptions options;
    options.has_header = true;

    auto data_result = CsvReader::read(fixture_path, options);
    REQUIRE(data_result.has_value());

    const auto& ts = *data_result;
    REQUIRE(ts.size() == 30);  // Expected 30 data points

    // Convert TimeSeries to std::vector<double>
    std::vector<double> data(ts.begin(), ts.end());

    // Step 2: Fit model with fixed specification ARIMA(1,0,1)-GARCH(1,1)
    ArimaGarchSpec spec(1, 0, 1, 1, 1);

    Engine engine;
    auto fit_result = engine.fit(data, spec, true);

    REQUIRE(fit_result.has_value());
    REQUIRE(fit_result.value().model != nullptr);
    REQUIRE(fit_result.value().summary.converged);
    REQUIRE(fit_result.value().summary.sample_size == 30);

    // Verify model parameters are within reasonable bounds
    const auto& fitted_model = *fit_result.value().model;
    const auto& fitted_spec = fitted_model.getSpec();
    REQUIRE(fitted_spec.arimaSpec.p == 1);
    REQUIRE(fitted_spec.arimaSpec.d == 0);
    REQUIRE(fitted_spec.arimaSpec.q == 1);
    REQUIRE(fitted_spec.garchSpec.p == 1);
    REQUIRE(fitted_spec.garchSpec.q == 1);

    // Step 3: Forecast with horizon=5
    const int forecast_horizon = 5;
    auto forecast_result = engine.forecast(fitted_model, forecast_horizon);

    REQUIRE(forecast_result.has_value());
    REQUIRE(forecast_result.value().mean_forecasts.size() == forecast_horizon);
    REQUIRE(forecast_result.value().variance_forecasts.size() == forecast_horizon);

    // Verify all forecast values are finite and reasonable
    for (int i = 0; i < forecast_horizon; ++i) {
        REQUIRE(std::isfinite(forecast_result.value().mean_forecasts[i]));
        REQUIRE(std::isfinite(forecast_result.value().variance_forecasts[i]));
        REQUIRE(forecast_result.value().variance_forecasts[i] > 0.0);
    }

    // Step 4: Simulate with steps=10, fixed seed
    const int sim_steps = 10;
    const unsigned int sim_seed = 42;

    // Extract fitted parameters for simulation
    ArimaGarchParameters fitted_params(spec);
    fitted_params.arima_params = fitted_model.getArimaParams();
    fitted_params.garch_params = fitted_model.getGarchParams();

    // Simulate first path
    auto sim_result1 = engine.simulate(spec, fitted_params, sim_steps, sim_seed);
    REQUIRE(sim_result1.has_value());
    REQUIRE(sim_result1.value().returns.size() == sim_steps);
    REQUIRE(sim_result1.value().volatilities.size() == sim_steps);

    // Verify all simulated values are finite
    for (int i = 0; i < sim_steps; ++i) {
        REQUIRE(std::isfinite(sim_result1.value().returns[i]));
        REQUIRE(std::isfinite(sim_result1.value().volatilities[i]));
        REQUIRE(sim_result1.value().volatilities[i] > 0.0);
    }

    // Simulate additional paths with same seed to verify determinism
    auto sim_result2 = engine.simulate(spec, fitted_params, sim_steps, sim_seed);
    REQUIRE(sim_result2.has_value());
    REQUIRE(sim_result2.value().returns.size() == sim_steps);

    // Verify deterministic output: same seed should produce identical results
    for (int i = 0; i < sim_steps; ++i) {
        REQUIRE_APPROX(sim_result1.value().returns[i], sim_result2.value().returns[i], 1e-15);
        REQUIRE_APPROX(sim_result1.value().volatilities[i], sim_result2.value().volatilities[i],
                       1e-15);
    }

    // Simulate third path with different seed
    auto sim_result3 = engine.simulate(spec, fitted_params, sim_steps, sim_seed + 1);
    REQUIRE(sim_result3.has_value());
    REQUIRE(sim_result3.value().returns.size() == sim_steps);

    // Verify different seed produces different results
    bool different = false;
    for (int i = 0; i < sim_steps; ++i) {
        if (std::abs(sim_result1.value().returns[i] - sim_result3.value().returns[i]) > 1e-10) {
            different = true;
            break;
        }
    }
    REQUIRE(different);  // With high probability, should be different

    // Step 5: Validate shape invariants
    // All outputs should have correct dimensions and valid values
    REQUIRE(fit_result.value().model != nullptr);
    REQUIRE(forecast_result.value().mean_forecasts.size() == forecast_horizon);
    REQUIRE(forecast_result.value().variance_forecasts.size() == forecast_horizon);
    REQUIRE(sim_result1.value().returns.size() == sim_steps);
    REQUIRE(sim_result1.value().volatilities.size() == sim_steps);
}

int main() {
    report_test_results("End-to-End Integration Tests");
    return get_test_result();
}
