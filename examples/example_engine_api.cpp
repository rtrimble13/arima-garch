/**
 * @file example_engine_api.cpp
 * @brief Demonstrates using the Engine API for ARIMA-GARCH modeling.
 *
 * This example shows how to use the ag::api::Engine facade for common operations:
 * 1. Fit a model to data
 * 2. Generate forecasts
 * 3. Automatic model selection
 * 4. Simulate synthetic data
 */

#include "ag/api/Engine.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/report/FitSummary.hpp"
#include "ag/selection/CandidateGrid.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"

#include <iostream>
#include <vector>

#include <fmt/core.h>

using ag::api::Engine;
using ag::models::ArimaGarchSpec;
using ag::models::composite::ArimaGarchParameters;
using ag::report::generateTextReport;
using ag::selection::CandidateGrid;
using ag::selection::CandidateGridConfig;
using ag::simulation::ArimaGarchSimulator;

int main() {
    fmt::print("=== Engine API Example ===\n\n");

    // Step 1: Generate synthetic data for demonstration
    fmt::print("Step 1: Generating synthetic data...\n");

    ArimaGarchSpec true_spec(1, 0, 1, 1, 1);  // ARIMA(1,0,1)-GARCH(1,1)
    ArimaGarchParameters true_params(true_spec);

    true_params.arima_params.intercept = 0.05;
    true_params.arima_params.ar_coef[0] = 0.6;
    true_params.arima_params.ma_coef[0] = 0.3;
    true_params.garch_params.omega = 0.01;
    true_params.garch_params.alpha_coef[0] = 0.1;
    true_params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchSimulator simulator(true_spec, true_params);
    auto sim_result = simulator.simulate(500, 42);

    fmt::print("  Generated {} observations\n\n", sim_result.returns.size());

    // Step 2: Fit a model using Engine
    fmt::print("Step 2: Fitting ARIMA-GARCH model...\n");

    Engine engine;
    auto fit_result = engine.fit(sim_result.returns, true_spec, true);

    if (!fit_result) {
        fmt::print("  ❌ Fit failed: {}\n", fit_result.error().message);
        return 1;
    }

    fmt::print("  ✅ Model fitted successfully\n");
    fmt::print("  Converged: {}\n", fit_result.value().summary.converged);
    fmt::print("  Iterations: {}\n", fit_result.value().summary.iterations);
    fmt::print("  AIC: {:.4f}\n", fit_result.value().summary.aic);
    fmt::print("  BIC: {:.4f}\n\n", fit_result.value().summary.bic);

    // Step 3: Generate forecasts
    fmt::print("Step 3: Generating forecasts...\n");

    auto forecast_result = engine.forecast(*fit_result.value().model, 10);

    if (!forecast_result) {
        fmt::print("  ❌ Forecast failed: {}\n", forecast_result.error().message);
        return 1;
    }

    fmt::print("  ✅ Generated 10-step ahead forecasts\n");
    fmt::print("  First 5 mean forecasts:\n");
    for (int i = 0; i < 5; ++i) {
        fmt::print("    t+{}: {:.6f} (volatility: {:.6f})\n", i + 1,
                   forecast_result.value().mean_forecasts[i],
                   std::sqrt(forecast_result.value().variance_forecasts[i]));
    }
    fmt::print("\n");

    // Step 4: Automatic model selection
    fmt::print("Step 4: Automatic model selection...\n");

    // Generate candidate models
    CandidateGridConfig grid_config(2, 1, 2, 1, 1);  // Max orders
    CandidateGrid grid(grid_config);
    auto candidates = grid.generate();

    fmt::print("  Evaluating {} candidate models...\n", candidates.size());

    auto select_result = engine.auto_select(sim_result.returns, candidates);

    if (!select_result) {
        fmt::print("  ❌ Selection failed: {}\n", select_result.error().message);
        return 1;
    }

    fmt::print("  ✅ Best model selected\n");
    fmt::print("  Selected: ARIMA({},{},{})-GARCH({},{})\n",
               select_result.value().selected_spec.arimaSpec.p,
               select_result.value().selected_spec.arimaSpec.d,
               select_result.value().selected_spec.arimaSpec.q,
               select_result.value().selected_spec.garchSpec.p,
               select_result.value().selected_spec.garchSpec.q);
    fmt::print("  Candidates evaluated: {}\n", select_result.value().candidates_evaluated);
    fmt::print("  Candidates failed: {}\n", select_result.value().candidates_failed);
    fmt::print("  BIC: {:.4f}\n\n", select_result.value().summary.bic);

    // Step 5: Simulate from the fitted model
    fmt::print("Step 5: Simulating from fitted model...\n");

    auto simulate_result = engine.simulate(select_result.value().selected_spec,
                                           select_result.value().summary.parameters, 100, 123);

    if (!simulate_result) {
        fmt::print("  ❌ Simulation failed: {}\n", simulate_result.error().message);
        return 1;
    }

    fmt::print("  ✅ Simulated {} observations\n", simulate_result.value().returns.size());
    fmt::print("  Mean of simulated returns: {:.6f}\n",
               std::accumulate(simulate_result.value().returns.begin(),
                               simulate_result.value().returns.end(), 0.0) /
                   simulate_result.value().returns.size());
    fmt::print("\n");

    // Step 6: Print full fit summary report
    fmt::print("Step 6: Full fit summary report:\n");
    fmt::print("{}\n", std::string(60, '-'));
    std::string report = generateTextReport(select_result.value().summary);
    std::cout << report << std::endl;

    fmt::print("\n✅ Example complete!\n");
    fmt::print("\nKey features demonstrated:\n");
    fmt::print("  • fit()         - Complete model fitting pipeline\n");
    fmt::print("  • forecast()    - Multi-step ahead forecasting\n");
    fmt::print("  • auto_select() - Automatic model selection\n");
    fmt::print("  • simulate()    - Synthetic data generation\n");

    return 0;
}
