#include "ag/diagnostics/DiagnosticReport.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"

#include <algorithm>
#include <cmath>
#include <random>

#include "test_framework.hpp"

using ag::diagnostics::computeDiagnostics;
using ag::models::ArimaGarchSpec;
using ag::models::composite::ArimaGarchParameters;
using ag::simulation::ArimaGarchSimulator;

// ============================================================================
// Basic Diagnostic Report Tests
// ============================================================================

// Test that diagnostic report is generated successfully for white noise
TEST(diagnostic_report_white_noise) {
    // ARIMA(0,0,0)-GARCH(1,1) - white noise mean with simple GARCH
    ArimaGarchSpec spec(0, 0, 0, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.0;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    // Generate white noise data
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(500);
    for (auto& val : data) {
        val = dist(gen);
    }

    // Compute diagnostics
    auto report = computeDiagnostics(spec, params, data, 10, false);

    // Verify that all fields are present and valid
    REQUIRE(report.ljung_box_residuals.lags == 10);
    REQUIRE(report.ljung_box_residuals.dof > 0);
    REQUIRE(std::isfinite(report.ljung_box_residuals.statistic));
    REQUIRE(std::isfinite(report.ljung_box_residuals.p_value));
    REQUIRE(report.ljung_box_residuals.p_value >= 0.0);
    REQUIRE(report.ljung_box_residuals.p_value <= 1.0);

    REQUIRE(report.ljung_box_squared.lags == 10);
    REQUIRE(report.ljung_box_squared.dof > 0);
    REQUIRE(std::isfinite(report.ljung_box_squared.statistic));
    REQUIRE(std::isfinite(report.ljung_box_squared.p_value));
    REQUIRE(report.ljung_box_squared.p_value >= 0.0);
    REQUIRE(report.ljung_box_squared.p_value <= 1.0);

    REQUIRE(std::isfinite(report.jarque_bera.statistic));
    REQUIRE(std::isfinite(report.jarque_bera.p_value));
    REQUIRE(report.jarque_bera.p_value >= 0.0);
    REQUIRE(report.jarque_bera.p_value <= 1.0);

    // ADF should not be included by default
    REQUIRE(!report.adf.has_value());
}

// Test that white noise passes Ljung-Box test more often than not
TEST(diagnostic_report_white_noise_passes_ljung_box) {
    // Run multiple trials with white noise
    ArimaGarchSpec spec(0, 0, 0, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.0;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.05;  // Small ARCH effect
    params.garch_params.beta_coef[0] = 0.90;   // High persistence

    int num_trials = 10;
    int num_passes = 0;
    double alpha = 0.05;  // Significance level

    for (int trial = 0; trial < num_trials; ++trial) {
        std::mt19937 gen(100 + trial);  // Different seed for each trial
        std::normal_distribution<double> dist(0.0, 1.0);

        std::vector<double> data(300);
        for (auto& val : data) {
            val = dist(gen);
        }

        auto report = computeDiagnostics(spec, params, data, 10, false);

        // Check if Ljung-Box test on residuals passes (p-value > alpha)
        if (report.ljung_box_residuals.p_value > alpha) {
            num_passes++;
        }
    }

    // White noise should pass more often than not (at least 50% of the time)
    // With alpha = 0.05, we expect about 95% to pass, but allow for some variation
    REQUIRE(num_passes >= num_trials / 2);
}

// Test with simulated ARIMA-GARCH data
TEST(diagnostic_report_simulated_arima_garch) {
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

    // Compute diagnostics using the same (correct) parameters
    auto report = computeDiagnostics(spec, params, sim_result.returns, 10, false);

    // For a correctly specified model, we expect:
    // - Ljung-Box tests should generally have high p-values (no autocorrelation)
    // - Jarque-Bera may or may not pass (depends on innovation distribution)

    // Verify all fields are valid
    REQUIRE(std::isfinite(report.ljung_box_residuals.statistic));
    REQUIRE(std::isfinite(report.ljung_box_residuals.p_value));
    REQUIRE(report.ljung_box_residuals.p_value >= 0.0);
    REQUIRE(report.ljung_box_residuals.p_value <= 1.0);

    REQUIRE(std::isfinite(report.ljung_box_squared.statistic));
    REQUIRE(std::isfinite(report.ljung_box_squared.p_value));
    REQUIRE(report.ljung_box_squared.p_value >= 0.0);
    REQUIRE(report.ljung_box_squared.p_value <= 1.0);

    REQUIRE(std::isfinite(report.jarque_bera.statistic));
    REQUIRE(std::isfinite(report.jarque_bera.p_value));

    // For correctly specified model, Ljung-Box p-value on residuals should typically be > 0.01
    // This is a lenient check to allow for some random variation
    REQUIRE(report.ljung_box_residuals.p_value > 0.01);

    // Note: Ljung-Box squared can occasionally fail even for correctly specified models
    // due to random variation, so we just verify it's a valid value
    REQUIRE(std::isfinite(report.ljung_box_squared.p_value));
}

// Test with ADF test included
TEST(diagnostic_report_with_adf) {
    ArimaGarchSpec spec(1, 0, 0, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.02;
    params.arima_params.ar_coef[0] = 0.7;
    params.garch_params.omega = 0.02;
    params.garch_params.alpha_coef[0] = 0.15;
    params.garch_params.beta_coef[0] = 0.80;

    // Simulate data
    ArimaGarchSimulator simulator(spec, params);
    auto sim_result = simulator.simulate(500, 123);

    // Compute diagnostics WITH ADF test
    auto report = computeDiagnostics(spec, params, sim_result.returns, 10, true);

    // Verify ADF test is included
    REQUIRE(report.adf.has_value());

    // Verify ADF result is valid
    REQUIRE(std::isfinite(report.adf->statistic));
    REQUIRE(std::isfinite(report.adf->p_value));
    REQUIRE(report.adf->p_value >= 0.0);
    REQUIRE(report.adf->p_value <= 1.0);
    // Note: ADF lags can be 0 with automatic selection
    REQUIRE(report.adf->lags >= 0);
}

// Test custom number of lags
TEST(diagnostic_report_custom_lags) {
    ArimaGarchSpec spec(0, 0, 0, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.0;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(500);
    for (auto& val : data) {
        val = dist(gen);
    }

    // Test with 20 lags
    auto report = computeDiagnostics(spec, params, data, 20, false);

    REQUIRE(report.ljung_box_residuals.lags == 20);
    REQUIRE(report.ljung_box_squared.lags == 20);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

// Test with empty data
TEST(diagnostic_report_empty_data) {
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
        (void)computeDiagnostics(spec, params, empty_data);
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }

    REQUIRE(caught_exception);
}

// Test with invalid number of lags (zero)
TEST(diagnostic_report_zero_lags) {
    ArimaGarchSpec spec(0, 0, 0, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.0;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    std::vector<double> data(100, 1.0);

    bool caught_exception = false;
    try {
        (void)computeDiagnostics(spec, params, data, 0, false);
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }

    REQUIRE(caught_exception);
}

// Test with too many lags
TEST(diagnostic_report_too_many_lags) {
    ArimaGarchSpec spec(0, 0, 0, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.0;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    std::vector<double> data(50, 1.0);

    bool caught_exception = false;
    try {
        (void)computeDiagnostics(spec, params, data, 100, false);  // More lags than data
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }

    REQUIRE(caught_exception);
}

// Test with insufficient lags (lags <= number of parameters)
TEST(diagnostic_report_insufficient_lags) {
    // ARIMA(2,0,2)-GARCH(1,1)
    // Total parameters via spec.totalParamCount():
    //   ARIMA: p=2 AR + q=2 MA + 1 intercept = 5
    //   GARCH: p=1 GARCH + q=1 ARCH + 1 omega = 3
    //   Total = 8
    ArimaGarchSpec spec(2, 0, 2, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.0;
    params.arima_params.ar_coef[0] = 0.3;
    params.arima_params.ar_coef[1] = 0.2;
    params.arima_params.ma_coef[0] = 0.1;
    params.arima_params.ma_coef[1] = 0.1;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    std::vector<double> data(100, 1.0);

    bool caught_exception = false;
    try {
        // Use only 8 lags, which equals the number of parameters
        // This should fail since DOF would be 0
        (void)computeDiagnostics(spec, params, data, 8, false);
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }

    REQUIRE(caught_exception);
}

// ============================================================================
// Main test runner
// ============================================================================

int main() {
    report_test_results("Diagnostics: DiagnosticReport");
    return get_test_result();
}
