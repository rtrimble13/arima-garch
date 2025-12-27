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

    // Step 6: Compare Normal vs Student-T distributions
    fmt::print("Step 6: Comparing Normal vs Student-T distributions...\n");

    // Constants for Student-T distribution search
    constexpr double MIN_DF = 3.0;   // Minimum degrees of freedom to consider
    constexpr double MAX_DF = 30.0;  // Maximum degrees of freedom to consider
    constexpr double DF_STEP = 1.0;  // Step size for df grid search

    // Helper function to compute chi-square upper tail probability (p-value)
    // This is a simplified approximation for demonstration purposes.
    // For production use, consider using boost::math::chi_squared or exposing
    // the chi_square_ccdf function from JarqueBera.cpp as a public utility.
    auto chi_square_ccdf = [](double x, double k) -> double {
        if (x <= 0.0)
            return 1.0;
        if (k == 1.0) {
            // For df=1, chi-square is square of standard normal
            // P(χ²(1) > x) ≈ 2 * P(Z > sqrt(x)) where Z ~ N(0,1)
            double z = std::sqrt(x);
            return std::erfc(z / std::sqrt(2.0)) / 2.0;
        }
        // For other df, use Wilson-Hilferty approximation
        // Transforms chi-square to approximately standard normal
        double z = std::pow(x / k, 1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k));
        z /= std::sqrt(2.0 / (9.0 * k));
        return std::erfc(z / std::sqrt(2.0)) / 2.0;
    };

    // Fit with Student-T distribution
    // Try different df values and pick the one with best likelihood
    double best_df = MIN_DF;
    double best_nll = 1e10;
    ag::models::arima::ArimaParameters best_arima_t = summary.parameters.arima_params;
    ag::models::garch::GarchParameters best_garch_t = summary.parameters.garch_params;

    // Grid search over df values
    for (double df = MIN_DF; df <= MAX_DF; df += DF_STEP) {
        ArimaGarchLikelihood likelihood_t(true_spec,
                                          ag::estimation::InnovationDistribution::StudentT);

        // Pack parameters for optimization (including df as last parameter)
        std::vector<double> initial_params_t = initial_params;
        initial_params_t.push_back(df);

        auto objective_t = [&](const std::vector<double>& params) -> double {
            ag::models::arima::ArimaParameters arima_p(1, 1);
            ag::models::garch::GarchParameters garch_p(1, 1);

            arima_p.intercept = params[0];
            arima_p.ar_coef[0] = params[1];
            arima_p.ma_coef[0] = params[2];
            garch_p.omega = params[3];
            garch_p.alpha_coef[0] = params[4];
            garch_p.beta_coef[0] = params[5];
            double df_param = params[6];

            if (!garch_p.isPositive() || !garch_p.isStationary() || df_param <= 2.0) {
                return 1e10;
            }

            try {
                return likelihood_t.computeNegativeLogLikelihood(data.data(), data.size(), arima_p,
                                                                 garch_p, df_param);
            } catch (...) {
                return 1e10;
            }
        };

        // Quick local optimization for this df
        NelderMeadOptimizer optimizer_t(1e-4, 1e-4, 500);
        auto result_t = optimizer_t.minimize(objective_t, initial_params_t);

        if (result_t.converged && result_t.objective_value < best_nll) {
            best_nll = result_t.objective_value;
            best_df = result_t.parameters[6];
            best_arima_t.intercept = result_t.parameters[0];
            best_arima_t.ar_coef[0] = result_t.parameters[1];
            best_arima_t.ma_coef[0] = result_t.parameters[2];
            best_garch_t.omega = result_t.parameters[3];
            best_garch_t.alpha_coef[0] = result_t.parameters[4];
            best_garch_t.beta_coef[0] = result_t.parameters[5];
        }
    }

    // Create distribution comparison
    ag::report::DistributionComparison dc;
    dc.normal_log_likelihood = -summary.neg_log_likelihood;
    dc.student_t_log_likelihood = -best_nll;
    dc.student_t_df = best_df;

    // Likelihood ratio test: LR = 2 * (LL_studentT - LL_normal)
    dc.lr_statistic = 2.0 * (dc.student_t_log_likelihood - dc.normal_log_likelihood);
    // Under H0, LR ~ chi-square(1) because Student-T has 1 extra parameter (df)
    dc.lr_p_value = chi_square_ccdf(dc.lr_statistic, 1.0);

    // Information criteria
    std::size_t k_normal = k;
    std::size_t k_student = k + 1;  // +1 for df parameter
    std::size_t n_obs = n;
    dc.normal_aic = summary.aic;
    dc.normal_bic = summary.bic;
    dc.student_t_aic = 2.0 * k_student + 2.0 * best_nll;
    dc.student_t_bic = k_student * std::log(n_obs) + 2.0 * best_nll;

    // Decide preference: Student-T is preferred if LR test is significant AND
    // information criteria favor it
    dc.prefer_student_t = (dc.lr_p_value < 0.05) && (dc.student_t_bic < dc.normal_bic);

    summary.distribution_comparison = dc;

    fmt::print("  Distribution comparison complete\n");
    fmt::print("    Normal log-likelihood:    {:.2f}\n", dc.normal_log_likelihood);
    fmt::print("    Student-T log-likelihood: {:.2f}\n", dc.student_t_log_likelihood);
    fmt::print("    Student-T df:             {:.2f}\n", dc.student_t_df);
    fmt::print("    LR test p-value:          {:.4f}\n", dc.lr_p_value);
    fmt::print("    Prefer Student-T:         {}\n\n", dc.prefer_student_t ? "Yes" : "No");

    // Step 7: Generate and display text report
    fmt::print("Step 7: Generating text report...\n\n");

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
