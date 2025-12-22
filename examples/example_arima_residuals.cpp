#include "ag/models/ArimaSpec.hpp"
#include "ag/models/arima/ArimaModel.hpp"

#include <iostream>
#include <random>
#include <vector>

#include <fmt/core.h>

using ag::models::ArimaSpec;
using ag::models::arima::ArimaModel;
using ag::models::arima::ArimaParameters;

/**
 * @brief Simulate an AR(1) process: y_t = c + φ*y_{t-1} + ε_t
 */
std::vector<double> simulate_ar1(double phi, double intercept,
                                 const std::vector<double>& innovations) {
    std::vector<double> series;
    series.reserve(innovations.size());

    series.push_back(intercept + innovations[0]);
    for (std::size_t t = 1; t < innovations.size(); ++t) {
        series.push_back(intercept + phi * series[t - 1] + innovations[t]);
    }

    return series;
}

int main() {
    fmt::print("=== ARIMA Residual Computation Example ===\n\n");

    // Set up a simple AR(1) model
    const double phi = 0.7;        // AR coefficient
    const double intercept = 2.0;  // Intercept term
    const int n_obs = 10;

    // Generate synthetic innovations (white noise)
    std::vector<double> true_innovations = {0.5, -0.3, 0.8, -0.2, 0.4, -0.1, 0.6, -0.4, 0.2, 0.1};

    fmt::print("Simulating AR(1) process:\n");
    fmt::print("  φ (AR coefficient) = {:.2f}\n", phi);
    fmt::print("  c (intercept) = {:.2f}\n", intercept);
    fmt::print("  n_obs = {}\n\n", n_obs);

    // Simulate the AR(1) series
    auto series = simulate_ar1(phi, intercept, true_innovations);

    fmt::print("Generated time series:\n  ");
    for (std::size_t t = 0; t < series.size(); ++t) {
        fmt::print("{:.3f} ", series[t]);
    }
    fmt::print("\n\n");

    // Set up ARIMA(1,0,0) model
    ArimaSpec spec(1, 0, 0);
    ArimaModel model(spec);

    // Set parameters to the true values used in simulation
    ArimaParameters params(1, 0);
    params.intercept = intercept;
    params.ar_coef[0] = phi;

    // Compute residuals
    auto residuals = model.computeResiduals(series.data(), series.size(), params);

    fmt::print("Computed residuals (should match innovations):\n  ");
    for (std::size_t t = 0; t < residuals.size(); ++t) {
        fmt::print("{:.3f} ", residuals[t]);
    }
    fmt::print("\n\n");

    fmt::print("True innovations:\n  ");
    for (std::size_t t = 0; t < true_innovations.size(); ++t) {
        fmt::print("{:.3f} ", true_innovations[t]);
    }
    fmt::print("\n\n");

    // Compute error between residuals and true innovations
    double max_error = 0.0;
    for (std::size_t t = 0; t < residuals.size(); ++t) {
        double error = std::abs(residuals[t] - true_innovations[t]);
        max_error = std::max(max_error, error);
    }

    fmt::print("Maximum error: {:.2e}\n", max_error);

    if (max_error < 1e-10) {
        fmt::print("✓ Residuals match innovations perfectly!\n");
    } else {
        fmt::print("✗ Residuals do not match innovations\n");
    }

    fmt::print("\n=== Example with Random Walk (ARIMA(0,1,0)) ===\n\n");

    // Generate a random walk
    std::vector<double> rw_innovations = {1.0, 0.5, -0.5, 0.8, -0.3};
    std::vector<double> rw_series;
    rw_series.push_back(rw_innovations[0]);
    for (std::size_t t = 1; t < rw_innovations.size(); ++t) {
        rw_series.push_back(rw_series[t - 1] + rw_innovations[t]);
    }

    fmt::print("Random walk series: ");
    for (auto v : rw_series) {
        fmt::print("{:.3f} ", v);
    }
    fmt::print("\n");

    // Set up ARIMA(0,1,0) model
    ArimaSpec rw_spec(0, 1, 0);
    ArimaModel rw_model(rw_spec);

    ArimaParameters rw_params(0, 0);
    rw_params.intercept = 0.0;

    auto rw_residuals = rw_model.computeResiduals(rw_series.data(), rw_series.size(), rw_params);

    fmt::print("Residuals (differenced series): ");
    for (auto v : rw_residuals) {
        fmt::print("{:.3f} ", v);
    }
    fmt::print("\n");

    fmt::print("Expected (innovations[1:]): ");
    for (std::size_t t = 1; t < rw_innovations.size(); ++t) {
        fmt::print("{:.3f} ", rw_innovations[t]);
    }
    fmt::print("\n");

    fmt::print("\nNote: After differencing, we lose one observation.\n");
    fmt::print("The residuals represent the first differences of the series.\n");

    return 0;
}
