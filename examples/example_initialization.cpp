/**
 * @file example_initialization.cpp
 * @brief Demonstrates parameter initialization and random restarts for ARIMA-GARCH models.
 *
 * This example shows how to:
 * 1. Generate synthetic AR(1)-GARCH(1,1) data
 * 2. Initialize parameters using heuristics
 * 3. Optimize with random restarts for robust convergence
 */

#include "ag/estimation/Likelihood.hpp"
#include "ag/estimation/Optimizer.hpp"
#include "ag/estimation/ParameterInitialization.hpp"
#include "ag/models/ArimaGarchSpec.hpp"

#include <cmath>
#include <random>
#include <vector>

#include <fmt/core.h>

// Generate synthetic AR(1)-GARCH(1,1) data
std::vector<double> generateSyntheticData(int n, double phi, double omega, double alpha,
                                          double beta, unsigned int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data;
    data.reserve(n);

    double y = 0.0;
    double h = omega / (1.0 - alpha - beta);  // Unconditional variance
    double eps_prev_sq = 0.0;

    for (int i = 0; i < n; ++i) {
        h = omega + alpha * eps_prev_sq + beta * h;
        double z = dist(rng);
        double eps = std::sqrt(h) * z;
        y = phi * y + eps;
        data.push_back(y);
        eps_prev_sq = eps * eps;
    }

    return data;
}

int main() {
    fmt::print("=== ARIMA-GARCH Parameter Initialization and Random Restarts Example ===\n\n");

    // True model parameters: AR(1)-GARCH(1,1)
    const double true_phi = 0.7;
    const double true_omega = 0.05;
    const double true_alpha = 0.1;
    const double true_beta = 0.85;

    fmt::print("True parameters:\n");
    fmt::print("  AR(1): phi = {:.3f}\n", true_phi);
    fmt::print("  GARCH(1,1): omega = {:.3f}, alpha = {:.3f}, beta = {:.3f}\n\n", true_omega,
               true_alpha, true_beta);

    // Generate synthetic data
    const int n_obs = 500;
    const unsigned int seed = 42;
    auto data = generateSyntheticData(n_obs, true_phi, true_omega, true_alpha, true_beta, seed);
    fmt::print("Generated {} observations\n\n", n_obs);

    // Define model specification
    ag::models::ArimaGarchSpec spec(1, 0, 0, 1, 1);  // AR(1)-GARCH(1,1)

    // Step 1: Initialize parameters using heuristics
    fmt::print("Step 1: Initializing parameters using heuristics...\n");
    auto [arima_init, garch_init] =
        ag::estimation::initializeArimaGarchParameters(data.data(), data.size(), spec);

    fmt::print("Initial ARIMA parameters:\n");
    fmt::print("  intercept = {:.6f}\n", arima_init.intercept);
    fmt::print("  AR(1) coefficient = {:.6f}\n", arima_init.ar_coef[0]);

    fmt::print("Initial GARCH parameters:\n");
    fmt::print("  omega = {:.6f}\n", garch_init.omega);
    fmt::print("  alpha = {:.6f}\n", garch_init.alpha_coef[0]);
    fmt::print("  beta = {:.6f}\n\n", garch_init.beta_coef[0]);

    // Step 2: Set up likelihood function
    fmt::print("Step 2: Setting up optimization...\n");
    ag::estimation::ArimaGarchLikelihood likelihood(spec);

    // Pack parameters into a single vector
    std::vector<double> initial_params;
    initial_params.push_back(arima_init.intercept);
    initial_params.push_back(arima_init.ar_coef[0]);
    initial_params.push_back(garch_init.omega);
    initial_params.push_back(garch_init.alpha_coef[0]);
    initial_params.push_back(garch_init.beta_coef[0]);

    // Define objective function
    auto objective = [&](const std::vector<double>& params) -> double {
        ag::models::arima::ArimaParameters arima_p(1, 0);
        ag::models::garch::GarchParameters garch_p(1, 1);

        arima_p.intercept = params[0];
        arima_p.ar_coef[0] = params[1];
        garch_p.omega = params[2];
        garch_p.alpha_coef[0] = params[3];
        garch_p.beta_coef[0] = params[4];

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

    // Step 3: Optimize with random restarts
    fmt::print("Step 3: Optimizing with random restarts...\n");
    ag::estimation::NelderMeadOptimizer optimizer(1e-6, 1e-6, 2000);
    const int num_restarts = 3;
    const double perturbation_scale = 0.15;

    auto result = ag::estimation::optimizeWithRestarts(optimizer, objective, initial_params,
                                                       num_restarts, perturbation_scale, seed);

    fmt::print("\nOptimization results:\n");
    fmt::print("  Converged: {}\n", result.converged ? "Yes" : "No");
    fmt::print("  Iterations: {}\n", result.iterations);
    fmt::print("  Restarts performed: {}\n", result.restarts_performed);
    fmt::print("  Successful restarts: {}\n", result.successful_restarts);
    fmt::print("  Final objective value: {:.6f}\n\n", result.objective_value);

    // Extract final parameters
    fmt::print("Estimated parameters:\n");
    fmt::print("  intercept = {:.6f}\n", result.parameters[0]);
    fmt::print("  AR(1) coefficient = {:.6f} (true: {:.3f})\n", result.parameters[1], true_phi);
    fmt::print("  omega = {:.6f} (true: {:.3f})\n", result.parameters[2], true_omega);
    fmt::print("  alpha = {:.6f} (true: {:.3f})\n", result.parameters[3], true_alpha);
    fmt::print("  beta = {:.6f} (true: {:.3f})\n\n", result.parameters[4], true_beta);

    // Compute estimation errors
    double ar_error = std::abs(result.parameters[1] - true_phi);
    double omega_error = std::abs(result.parameters[2] - true_omega);
    double alpha_error = std::abs(result.parameters[3] - true_alpha);
    double beta_error = std::abs(result.parameters[4] - true_beta);

    fmt::print("Parameter estimation errors:\n");
    fmt::print("  AR(1): {:.6f}\n", ar_error);
    fmt::print("  omega: {:.6f}\n", omega_error);
    fmt::print("  alpha: {:.6f}\n", alpha_error);
    fmt::print("  beta: {:.6f}\n\n", beta_error);

    if (result.converged) {
        fmt::print("✓ Optimization converged successfully!\n");
        fmt::print("  Random restarts helped find {} improved solutions\n",
                   result.successful_restarts);
        return 0;
    } else {
        fmt::print("✗ Optimization did not converge\n");
        return 1;
    }
}
