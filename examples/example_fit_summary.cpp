/**
 * @file example_fit_summary.cpp
 * @brief Demonstrates generating a comprehensive fit summary report.
 *
 * This example shows how to:
 * 1. Fit an ARIMA-GARCH model to synthetic data
 * 2. Populate a FitSummary structure with results
 * 3. Compute diagnostic tests
 * 4. Generate a formatted text report
 * 5. Display the report to console (or write to file)
 */

#include "ag/diagnostics/DiagnosticReport.hpp"
#include "ag/estimation/Likelihood.hpp"
#include "ag/estimation/Optimizer.hpp"
#include "ag/estimation/ParameterInitialization.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/report/FitSummary.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#include <fmt/core.h>

using ag::diagnostics::computeDiagnostics;
using ag::estimation::ArimaGarchLikelihood;
using ag::estimation::NelderMeadOptimizer;
using ag::estimation::optimizeWithRestarts;
using ag::models::ArimaGarchSpec;
using ag::models::composite::ArimaGarchParameters;
using ag::report::FitSummary;
using ag::report::generateTextReport;
using ag::simulation::ArimaGarchSimulator;

int main() {
    fmt::print("=== ARIMA-GARCH Fit Summary Report Example ===\n\n");

    // Step 1: Generate synthetic data from a known model
    fmt::print("Step 1: Generating synthetic data...\n");

    ArimaGarchSpec true_spec(1, 0, 1, 1, 1);  // ARIMA(1,0,1)-GARCH(1,1)
    ArimaGarchParameters true_params(true_spec);

    // Set true parameters
    true_params.arima_params.intercept = 0.05;
    true_params.arima_params.ar_coef[0] = 0.6;
    true_params.arima_params.ma_coef[0] = 0.3;
    true_params.garch_params.omega = 0.01;
    true_params.garch_params.alpha_coef[0] = 0.1;
    true_params.garch_params.beta_coef[0] = 0.85;

    // Simulate data
    ArimaGarchSimulator simulator(true_spec, true_params);
    auto sim_result = simulator.simulate(1000, 42);
    const auto& data = sim_result.returns;

    fmt::print("  Generated {} observations\n\n", data.size());

    // Step 2: Initialize parameters
    fmt::print("Step 2: Initializing parameters...\n");

    auto [arima_init, garch_init] =
        ag::estimation::initializeArimaGarchParameters(data.data(), data.size(), true_spec);

    fmt::print("  Initialization complete\n\n");

    // Step 3: Set up optimization
    fmt::print("Step 3: Fitting model (this may take a moment)...\n");

    ArimaGarchLikelihood likelihood(true_spec);

    // Pack parameters into vector
    std::vector<double> initial_params;
    initial_params.push_back(arima_init.intercept);
    initial_params.push_back(arima_init.ar_coef[0]);
    initial_params.push_back(arima_init.ma_coef[0]);
    initial_params.push_back(garch_init.omega);
    initial_params.push_back(garch_init.alpha_coef[0]);
    initial_params.push_back(garch_init.beta_coef[0]);

    // Define objective function
    auto objective = [&](const std::vector<double>& params) -> double {
        ag::models::arima::ArimaParameters arima_p(1, 1);
        ag::models::garch::GarchParameters garch_p(1, 1);

        arima_p.intercept = params[0];
        arima_p.ar_coef[0] = params[1];
        arima_p.ma_coef[0] = params[2];
        garch_p.omega = params[3];
        garch_p.alpha_coef[0] = params[4];
        garch_p.beta_coef[0] = params[5];

        // Check constraints
        if (!garch_p.isPositive() || !garch_p.isStationary()) {
            return 1e10;
        }

        try {
            return likelihood.computeNegativeLogLikelihood(data.data(), data.size(), arima_p,
                                                           garch_p);
        } catch (...) {
            return 1e10;
        }
    };

    // Optimize with random restarts
    NelderMeadOptimizer optimizer(1e-6, 1e-6, 2000);
    auto result = optimizeWithRestarts(optimizer, objective, initial_params, 3, 0.15, 42);

    fmt::print("  Optimization complete\n\n");

    // Step 4: Create FitSummary and populate it
    fmt::print("Step 4: Creating fit summary...\n");

    FitSummary summary(true_spec);

    // Unpack optimized parameters
    summary.parameters.arima_params.intercept = result.parameters[0];
    summary.parameters.arima_params.ar_coef[0] = result.parameters[1];
    summary.parameters.arima_params.ma_coef[0] = result.parameters[2];
    summary.parameters.garch_params.omega = result.parameters[3];
    summary.parameters.garch_params.alpha_coef[0] = result.parameters[4];
    summary.parameters.garch_params.beta_coef[0] = result.parameters[5];

    // Set convergence information
    summary.converged = result.converged;
    summary.iterations = result.iterations;
    summary.message = result.message;
    summary.sample_size = data.size();

    // Set likelihood and information criteria
    summary.neg_log_likelihood = result.objective_value;

    std::size_t k = true_spec.totalParamCount();
    std::size_t n = data.size();
    summary.aic = 2.0 * k + 2.0 * summary.neg_log_likelihood;
    summary.bic = k * std::log(n) + 2.0 * summary.neg_log_likelihood;

    fmt::print("  Fit summary populated\n\n");

    // Step 5: Compute diagnostics
    fmt::print("Step 5: Computing diagnostic tests...\n");

    auto diagnostics = computeDiagnostics(true_spec, summary.parameters, data, 10, true);
    summary.diagnostics = diagnostics;

    fmt::print("  Diagnostics computed\n\n");

    // Step 6: Generate and display text report
    fmt::print("Step 6: Generating text report...\n\n");

    std::string report = generateTextReport(summary);

    // Display the report
    std::cout << report << std::endl;

    // Optional: Show comparison with true parameters
    fmt::print("\n=== Parameter Recovery Analysis ===\n\n");
    fmt::print("Comparison with true parameters:\n");
    fmt::print("  Parameter          True       Estimated   Error\n");
    fmt::print("  ------------------------------------------------\n");
    fmt::print(
        "  Intercept        {: .6f}   {: .6f}   {: .6f}\n", true_params.arima_params.intercept,
        summary.parameters.arima_params.intercept,
        std::abs(true_params.arima_params.intercept - summary.parameters.arima_params.intercept));
    fmt::print(
        "  AR(1)            {: .6f}   {: .6f}   {: .6f}\n", true_params.arima_params.ar_coef[0],
        summary.parameters.arima_params.ar_coef[0],
        std::abs(true_params.arima_params.ar_coef[0] - summary.parameters.arima_params.ar_coef[0]));
    fmt::print(
        "  MA(1)            {: .6f}   {: .6f}   {: .6f}\n", true_params.arima_params.ma_coef[0],
        summary.parameters.arima_params.ma_coef[0],
        std::abs(true_params.arima_params.ma_coef[0] - summary.parameters.arima_params.ma_coef[0]));
    fmt::print("  Omega            {: .6f}   {: .6f}   {: .6f}\n", true_params.garch_params.omega,
               summary.parameters.garch_params.omega,
               std::abs(true_params.garch_params.omega - summary.parameters.garch_params.omega));
    fmt::print("  Alpha            {: .6f}   {: .6f}   {: .6f}\n",
               true_params.garch_params.alpha_coef[0],
               summary.parameters.garch_params.alpha_coef[0],
               std::abs(true_params.garch_params.alpha_coef[0] -
                        summary.parameters.garch_params.alpha_coef[0]));
    fmt::print("  Beta             {: .6f}   {: .6f}   {: .6f}\n",
               true_params.garch_params.beta_coef[0], summary.parameters.garch_params.beta_coef[0],
               std::abs(true_params.garch_params.beta_coef[0] -
                        summary.parameters.garch_params.beta_coef[0]));
    fmt::print("\n");

    fmt::print("✓ Example complete! The FitSummary provides a comprehensive\n");
    fmt::print("  report of model fitting results that can be printed to console\n");
    fmt::print("  or saved to a file for documentation purposes.\n");

    return 0;
}
