#include "ag/diagnostics/Residuals.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"
#include "ag/stats/Descriptive.hpp"

#include <iostream>

#include <fmt/core.h>

using ag::diagnostics::computeResiduals;
using ag::models::ArimaGarchSpec;
using ag::models::composite::ArimaGarchParameters;
using ag::simulation::ArimaGarchSimulator;

int main() {
    fmt::print("=== ARIMA-GARCH Residual Diagnostics Example ===\n\n");

    // Define a well-behaved ARIMA(1,0,1)-GARCH(1,1) model
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.05;
    params.arima_params.ar_coef[0] = 0.6;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.01;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.85;

    fmt::print("Model Specification: ARIMA({},{},{})-GARCH({},{})\n", spec.arimaSpec.p,
               spec.arimaSpec.d, spec.arimaSpec.q, spec.garchSpec.p, spec.garchSpec.q);
    fmt::print("\nARIMA Parameters:\n");
    fmt::print("  Intercept: {:.4f}\n", params.arima_params.intercept);
    fmt::print("  AR(1): {:.4f}\n", params.arima_params.ar_coef[0]);
    fmt::print("  MA(1): {:.4f}\n", params.arima_params.ma_coef[0]);
    fmt::print("\nGARCH Parameters:\n");
    fmt::print("  ω (omega): {:.4f}\n", params.garch_params.omega);
    fmt::print("  α (alpha): {:.4f}\n", params.garch_params.alpha_coef[0]);
    fmt::print("  β (beta): {:.4f}\n", params.garch_params.beta_coef[0]);

    // Simulate data from this model
    fmt::print("\nSimulating {} observations...\n", 1000);
    ArimaGarchSimulator simulator(spec, params);
    auto sim_result = simulator.simulate(1000, 42);

    // Compute residuals using the same (correct) parameters
    fmt::print("Computing residuals...\n");
    auto residuals = computeResiduals(spec, params, sim_result.returns);

    fmt::print("\nResidual Series Summary:\n");
    fmt::print("  Number of observations: {}\n", residuals.eps_t.size());

    // Compute statistics for raw residuals
    double mean_eps = 0.0;
    for (double val : residuals.eps_t) {
        mean_eps += val;
    }
    mean_eps /= residuals.eps_t.size();

    double var_eps = 0.0;
    for (double val : residuals.eps_t) {
        double diff = val - mean_eps;
        var_eps += diff * diff;
    }
    var_eps /= (residuals.eps_t.size() - 1);  // Sample variance (n-1)

    fmt::print("\nRaw Residuals (eps_t):\n");
    fmt::print("  Mean: {:.6f}\n", mean_eps);
    fmt::print("  Variance: {:.6f}\n", var_eps);
    fmt::print("  Std Dev: {:.6f}\n", std::sqrt(var_eps));

    // Compute statistics for conditional variances
    double mean_h = 0.0;
    double min_h = residuals.h_t[0];
    double max_h = residuals.h_t[0];

    for (double val : residuals.h_t) {
        mean_h += val;
        if (val < min_h)
            min_h = val;
        if (val > max_h)
            max_h = val;
    }
    mean_h /= residuals.h_t.size();

    fmt::print("\nConditional Variances (h_t):\n");
    fmt::print("  Mean: {:.6f}\n", mean_h);
    fmt::print("  Min: {:.6f}\n", min_h);
    fmt::print("  Max: {:.6f}\n", max_h);
    fmt::print("  Range: {:.6f}\n", max_h - min_h);

    // Compute statistics for standardized residuals
    double mean_std = 0.0;
    for (double val : residuals.std_eps_t) {
        mean_std += val;
    }
    mean_std /= residuals.std_eps_t.size();

    double var_std = 0.0;
    for (double val : residuals.std_eps_t) {
        double diff = val - mean_std;
        var_std += diff * diff;
    }
    var_std /= (residuals.std_eps_t.size() - 1);  // Sample variance (n-1)

    fmt::print("\nStandardized Residuals (std_eps_t = eps_t / sqrt(h_t)):\n");
    fmt::print("  Mean: {:.6f}\n", mean_std);
    fmt::print("  Variance: {:.6f}\n", var_std);
    fmt::print("  Std Dev: {:.6f}\n", std::sqrt(var_std));

    // For a correctly specified model, standardized residuals should be approximately N(0,1)
    fmt::print("\n=== Model Diagnostics ===\n");
    fmt::print("For a correctly specified model:\n");
    fmt::print("  - Standardized residuals should have mean ≈ 0\n");
    fmt::print("  - Standardized residuals should have variance ≈ 1\n\n");

    if (std::abs(mean_std) < 0.1 && std::abs(var_std - 1.0) < 0.15) {
        fmt::print("✓ Diagnostics look good! The model appears to be correctly specified.\n");
        fmt::print("  Mean is close to 0: {:.6f}\n", mean_std);
        fmt::print("  Variance is close to 1: {:.6f}\n", var_std);
    } else {
        fmt::print("⚠ Diagnostics indicate potential model misspecification.\n");
        if (std::abs(mean_std) >= 0.1) {
            fmt::print("  Warning: Mean is not close to 0: {:.6f}\n", mean_std);
        }
        if (std::abs(var_std - 1.0) >= 0.15) {
            fmt::print("  Warning: Variance is not close to 1: {:.6f}\n", var_std);
        }
    }

    fmt::print("\n=== First 10 observations ===\n");
    fmt::print("{:>6} {:>12} {:>12} {:>12}\n", "t", "y_t", "eps_t", "std_eps_t");
    fmt::print("{}\n", std::string(48, '-'));
    for (int t = 0; t < 10; ++t) {
        fmt::print("{:6d} {:12.6f} {:12.6f} {:12.6f}\n", t, sim_result.returns[t],
                   residuals.eps_t[t], residuals.std_eps_t[t]);
    }

    return 0;
}
