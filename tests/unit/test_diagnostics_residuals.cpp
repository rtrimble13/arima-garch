#include "ag/diagnostics/Residuals.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"
#include "ag/stats/Descriptive.hpp"

#include <algorithm>
#include <cmath>
#include <random>

#include "test_framework.hpp"

using ag::diagnostics::computeResiduals;
using ag::diagnostics::ResidualSeries;
using ag::models::ArimaGarchSpec;
using ag::models::composite::ArimaGarchParameters;
using ag::simulation::ArimaGarchSimulator;

// ============================================================================
// Basic Residual Computation Tests
// ============================================================================

// Test basic residual computation with white noise mean and constant variance
TEST(residuals_white_noise_constant_variance) {
    // ARIMA(0,0,0)-GARCH(1,1) - white noise mean with near-constant variance
    // Use very small alpha and beta to approximate constant variance
    ArimaGarchSpec spec(0, 0, 0, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.0;
    params.garch_params.omega = 1.0;            // Constant component
    params.garch_params.alpha_coef[0] = 0.001;  // Minimal ARCH effect
    params.garch_params.beta_coef[0] = 0.001;   // Minimal GARCH effect

    // Simple data
    std::vector<double> data = {1.0, -0.5, 0.8, -0.3, 0.4};

    auto residuals = computeResiduals(spec, params, data);

    // Verify sizes
    REQUIRE(residuals.eps_t.size() == data.size());
    REQUIRE(residuals.h_t.size() == data.size());
    REQUIRE(residuals.std_eps_t.size() == data.size());

    // For white noise with zero intercept, eps_t should equal data
    for (std::size_t t = 0; t < data.size(); ++t) {
        REQUIRE_APPROX(residuals.eps_t[t], data[t], 1e-10);
    }

    // For near-constant variance, h_t should be approximately constant (~1.0)
    for (std::size_t t = 0; t < data.size(); ++t) {
        REQUIRE(residuals.h_t[t] > 0.99);
        REQUIRE(residuals.h_t[t] < 1.01);
    }

    // Standardized residuals should be approximately equal to raw residuals
    for (std::size_t t = 0; t < data.size(); ++t) {
        REQUIRE_APPROX(residuals.std_eps_t[t], residuals.eps_t[t], 0.02);
    }
}

// Test that all outputs are finite (no NaNs or Infs)
TEST(residuals_no_nans) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.5;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    std::vector<double> data = {1.0, 2.0, 1.5, 2.2, 1.8, 2.5, 1.3, 2.0};

    auto residuals = computeResiduals(spec, params, data);

    // Verify all values are finite
    for (std::size_t t = 0; t < data.size(); ++t) {
        REQUIRE(std::isfinite(residuals.eps_t[t]));
        REQUIRE(std::isfinite(residuals.h_t[t]));
        REQUIRE(std::isfinite(residuals.std_eps_t[t]));
    }
}

// ============================================================================
// Tests with Simulated Data
// ============================================================================

// Test that standardized residuals have variance approximately 1
// for a correctly specified model on simulated data
TEST(residuals_standardized_variance_one) {
    // Create a well-behaved ARIMA(1,0,1)-GARCH(1,1) model
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.05;
    params.arima_params.ar_coef[0] = 0.6;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.01;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.85;

    // Simulate data from this model
    ArimaGarchSimulator simulator(spec, params);
    auto sim_result = simulator.simulate(1000, 42);

    // Compute residuals using the same (correct) parameters
    auto residuals = computeResiduals(spec, params, sim_result.returns);

    // Compute variance of standardized residuals
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
    var_std /= residuals.std_eps_t.size();

    // For a correctly specified model, standardized residuals should have variance ~1
    // Allow some tolerance due to finite sample effects
    REQUIRE(var_std > 0.8);              // Lower bound
    REQUIRE(var_std < 1.2);              // Upper bound
    REQUIRE_APPROX(var_std, 1.0, 0.15);  // Should be close to 1.0
}

// Test with larger sample size for tighter variance check
TEST(residuals_standardized_variance_large_sample) {
    // Create a stationary ARIMA(1,0,0)-GARCH(1,1) model
    ArimaGarchSpec spec(1, 0, 0, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.02;
    params.arima_params.ar_coef[0] = 0.7;
    params.garch_params.omega = 0.02;
    params.garch_params.alpha_coef[0] = 0.15;
    params.garch_params.beta_coef[0] = 0.80;

    // Simulate a larger sample
    ArimaGarchSimulator simulator(spec, params);
    auto sim_result = simulator.simulate(5000, 123);

    // Compute residuals
    auto residuals = computeResiduals(spec, params, sim_result.returns);

    // Compute variance of standardized residuals
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
    var_std /= residuals.std_eps_t.size();

    // With larger sample, variance should be closer to 1
    REQUIRE(var_std > 0.9);
    REQUIRE(var_std < 1.1);
    REQUIRE_APPROX(var_std, 1.0, 0.08);
}

// Test that conditional variances are positive
TEST(residuals_positive_conditional_variance) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.5;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    // Simulate data
    ArimaGarchSimulator simulator(spec, params);
    auto sim_result = simulator.simulate(500, 999);

    auto residuals = computeResiduals(spec, params, sim_result.returns);

    // All conditional variances should be positive
    for (std::size_t t = 0; t < residuals.h_t.size(); ++t) {
        REQUIRE(residuals.h_t[t] > 0.0);
    }
}

// ============================================================================
// GARCH Effect Tests
// ============================================================================

// Test that GARCH model produces time-varying conditional variance
TEST(residuals_time_varying_variance) {
    // GARCH(1,1) with significant parameters to ensure variance changes
    ArimaGarchSpec spec(0, 0, 0, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.0;
    params.garch_params.omega = 0.05;
    params.garch_params.alpha_coef[0] = 0.3;  // Strong ARCH effect
    params.garch_params.beta_coef[0] = 0.6;   // Strong GARCH effect

    // Create data with varying magnitude to trigger GARCH dynamics
    std::vector<double> data = {0.1, 0.2, 3.0, 0.1, 0.2, 0.1, -2.5, 0.1, 0.1, 0.2};

    auto residuals = computeResiduals(spec, params, data);

    // Conditional variance should vary over time
    double min_h = *std::min_element(residuals.h_t.begin(), residuals.h_t.end());
    double max_h = *std::max_element(residuals.h_t.begin(), residuals.h_t.end());

    // Should have meaningful variation
    REQUIRE(max_h > min_h * 1.5);  // At least 50% variation
}

// ============================================================================
// Error Handling Tests
// ============================================================================

// Test with null pointer
TEST(residuals_null_pointer) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.5;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    bool caught_exception = false;
    try {
        computeResiduals(spec, params, nullptr, 10);
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }

    REQUIRE(caught_exception);
}

// Test with empty data
TEST(residuals_empty_data) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.5;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    std::vector<double> empty_data;
    bool caught_exception = false;
    try {
        computeResiduals(spec, params, empty_data);
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }

    REQUIRE(caught_exception);
}

// Test with invalid GARCH parameters (negative omega)
TEST(residuals_invalid_garch_params) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.5;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = -0.1;  // Invalid!
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    std::vector<double> data = {1.0, 2.0, 1.5};

    bool caught_exception = false;
    try {
        computeResiduals(spec, params, data);
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }

    REQUIRE(caught_exception);
}

// ============================================================================
// Main test runner
// ============================================================================

int main() {
    report_test_results("Diagnostics: Residuals");
    return get_test_result();
}
