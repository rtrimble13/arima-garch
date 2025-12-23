#include "ag/forecasting/Forecaster.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/arima/ArimaModel.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/models/garch/GarchModel.hpp"

#include <cmath>
#include <iostream>

#include "test_framework.hpp"

using ag::forecasting::Forecaster;
using ag::forecasting::ForecastResult;
using ag::models::ArimaGarchSpec;
using ag::models::ArimaSpec;
using ag::models::GarchSpec;
using ag::models::arima::ArimaParameters;
using ag::models::composite::ArimaGarchModel;
using ag::models::composite::ArimaGarchParameters;
using ag::models::garch::GarchParameters;

// ============================================================================
// ForecastResult Tests
// ============================================================================

TEST(forecast_result_construction) {
    ForecastResult result(10);
    REQUIRE(result.mean_forecasts.size() == 10);
    REQUIRE(result.variance_forecasts.size() == 10);
}

// ============================================================================
// Forecaster Basic Tests
// ============================================================================

TEST(forecaster_construction) {
    // Create a simple ARIMA(1,0,0)-GARCH(1,1) model
    ArimaSpec arima_spec(1, 0, 0);
    GarchSpec garch_spec(1, 1);
    ArimaGarchSpec spec(arima_spec, garch_spec);

    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.5;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchModel model(spec, params);

    // Initialize model with some data
    model.update(1.0);
    model.update(1.1);
    model.update(0.9);

    // Create forecaster
    Forecaster forecaster(model);

    // Should construct without error
    REQUIRE(true);
}

TEST(forecaster_invalid_horizon) {
    ArimaSpec arima_spec(1, 0, 0);
    GarchSpec garch_spec(1, 1);
    ArimaGarchSpec spec(arima_spec, garch_spec);

    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.5;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchModel model(spec, params);
    model.update(1.0);

    Forecaster forecaster(model);

    // Test zero horizon
    bool caught_exception = false;
    try {
        forecaster.forecast(0);
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }
    REQUIRE(caught_exception);

    // Test negative horizon
    caught_exception = false;
    try {
        forecaster.forecast(-5);
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }
    REQUIRE(caught_exception);
}

// ============================================================================
// Mean Forecast Tests
// ============================================================================

TEST(forecast_mean_ar1_simple) {
    // AR(1) model: y_t = 0.1 + 0.5*y_{t-1} + ε_t
    ArimaSpec arima_spec(1, 0, 0);
    GarchSpec garch_spec(1, 1);
    ArimaGarchSpec spec(arima_spec, garch_spec);

    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.5;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchModel model(spec, params);

    // Initialize with y_0 = 1.0
    model.update(1.0);

    Forecaster forecaster(model);

    // Forecast 3 steps ahead
    auto result = forecaster.forecast(3);

    // Manual calculation:
    // ŷ_1 = 0.1 + 0.5 * 1.0 = 0.6
    double y1_expected = 0.1 + 0.5 * 1.0;
    REQUIRE_APPROX(result.mean_forecasts[0], y1_expected, 1e-10);

    // ŷ_2 = 0.1 + 0.5 * 0.6 = 0.4
    double y2_expected = 0.1 + 0.5 * 0.6;
    REQUIRE_APPROX(result.mean_forecasts[1], y2_expected, 1e-10);

    // ŷ_3 = 0.1 + 0.5 * 0.4 = 0.3
    double y3_expected = 0.1 + 0.5 * 0.4;
    REQUIRE_APPROX(result.mean_forecasts[2], y3_expected, 1e-10);
}

TEST(forecast_mean_ar1_convergence) {
    // Test that AR(1) forecast converges to unconditional mean
    // For AR(1): y_t = c + φ*y_{t-1} + ε_t
    // Unconditional mean: μ = c / (1 - φ)
    ArimaSpec arima_spec(1, 0, 0);
    GarchSpec garch_spec(1, 1);
    ArimaGarchSpec spec(arima_spec, garch_spec);

    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.2;   // c
    params.arima_params.ar_coef[0] = 0.6;  // φ
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    double unconditional_mean = 0.2 / (1.0 - 0.6);  // = 0.5

    ArimaGarchModel model(spec, params);
    model.update(2.0);  // Start far from unconditional mean

    Forecaster forecaster(model);

    // Forecast many steps ahead
    auto result = forecaster.forecast(50);

    // Last forecast should be close to unconditional mean
    REQUIRE(std::abs(result.mean_forecasts[49] - unconditional_mean) < 0.01);
}

TEST(forecast_mean_ma1) {
    // MA(1) model: y_t = 0.1 + ε_t + 0.5*ε_{t-1}
    ArimaSpec arima_spec(0, 0, 1);
    GarchSpec garch_spec(1, 1);
    ArimaGarchSpec spec(arima_spec, garch_spec);

    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.1;
    params.arima_params.ma_coef[0] = 0.5;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchModel model(spec, params);

    // Update with a value to get a residual
    model.update(1.0);  // ε_0 = 1.0 - 0.1 = 0.9

    Forecaster forecaster(model);

    // Forecast 3 steps ahead
    auto result = forecaster.forecast(3);

    // Manual calculation:
    // ŷ_1 = 0.1 + 0 + 0.5 * 0.9 = 0.55 (uses last residual)
    double y1_expected = 0.1 + 0.5 * 0.9;
    REQUIRE_APPROX(result.mean_forecasts[0], y1_expected, 1e-10);

    // ŷ_2 = 0.1 + 0 + 0.5 * 0 = 0.1 (future residuals are zero)
    double y2_expected = 0.1;
    REQUIRE_APPROX(result.mean_forecasts[1], y2_expected, 1e-10);

    // ŷ_3 = 0.1 + 0 + 0.5 * 0 = 0.1
    double y3_expected = 0.1;
    REQUIRE_APPROX(result.mean_forecasts[2], y3_expected, 1e-10);
}

TEST(forecast_mean_arma11) {
    // ARMA(1,1) model: y_t = 0.2 + 0.7*y_{t-1} + ε_t + 0.3*ε_{t-1}
    ArimaSpec arima_spec(1, 0, 1);
    GarchSpec garch_spec(1, 1);
    ArimaGarchSpec spec(arima_spec, garch_spec);

    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.2;
    params.arima_params.ar_coef[0] = 0.7;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchModel model(spec, params);

    // Initialize with specific values
    auto out1 = model.update(1.0);
    double eps_0 = 1.0 - out1.mu_t;

    Forecaster forecaster(model);

    // Forecast 2 steps ahead
    auto result = forecaster.forecast(2);

    // Manual calculation:
    // ŷ_1 = 0.2 + 0.7 * 1.0 + 0.3 * eps_0
    double y1_expected = 0.2 + 0.7 * 1.0 + 0.3 * eps_0;
    REQUIRE_APPROX(result.mean_forecasts[0], y1_expected, 1e-10);

    // ŷ_2 = 0.2 + 0.7 * ŷ_1 + 0.3 * 0
    double y2_expected = 0.2 + 0.7 * y1_expected;
    REQUIRE_APPROX(result.mean_forecasts[1], y2_expected, 1e-10);
}

// ============================================================================
// Variance Forecast Tests
// ============================================================================

TEST(forecast_variance_garch11_simple) {
    // GARCH(1,1): h_t = 0.1 + 0.1*ε²_{t-1} + 0.8*h_{t-1}
    ArimaSpec arima_spec(0, 0, 0);  // Just constant mean
    GarchSpec garch_spec(1, 1);
    ArimaGarchSpec spec(arima_spec, garch_spec);

    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.0;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchModel model(spec, params);

    // Initialize with a value to get variance and residual
    auto out = model.update(1.0);
    double h_0 = out.h_t;
    double eps_0 = 1.0 - out.mu_t;
    double eps2_0 = eps_0 * eps_0;

    Forecaster forecaster(model);

    // Forecast 3 steps ahead
    auto result = forecaster.forecast(3);

    // Manual calculation:
    // ĥ_1 = 0.1 + 0.1 * eps2_0 + 0.8 * h_0
    double h1_expected = 0.1 + 0.1 * eps2_0 + 0.8 * h_0;
    REQUIRE_APPROX(result.variance_forecasts[0], h1_expected, 1e-10);

    // ĥ_2 = 0.1 + 0.1 * ĥ_1 + 0.8 * ĥ_1 = 0.1 + 0.9 * ĥ_1
    double h2_expected = 0.1 + 0.1 * h1_expected + 0.8 * h1_expected;
    REQUIRE_APPROX(result.variance_forecasts[1], h2_expected, 1e-10);

    // ĥ_3 = 0.1 + 0.1 * ĥ_2 + 0.8 * ĥ_2 = 0.1 + 0.9 * ĥ_2
    double h3_expected = 0.1 + 0.1 * h2_expected + 0.8 * h2_expected;
    REQUIRE_APPROX(result.variance_forecasts[2], h3_expected, 1e-10);
}

TEST(forecast_variance_garch11_convergence_to_unconditional) {
    // Test that GARCH(1,1) variance forecast converges to unconditional variance
    // For GARCH(1,1): h_t = ω + α*ε²_{t-1} + β*h_{t-1}
    // Unconditional variance: σ² = ω / (1 - α - β)
    ArimaSpec arima_spec(0, 0, 0);
    GarchSpec garch_spec(1, 1);
    ArimaGarchSpec spec(arima_spec, garch_spec);

    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.0;
    params.garch_params.omega = 0.05;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.85;

    // Check stationarity
    REQUIRE(params.garch_params.isStationary());

    double unconditional_var = params.garch_params.unconditionalVariance();
    // = 0.05 / (1 - 0.1 - 0.85) = 0.05 / 0.05 = 1.0

    ArimaGarchModel model(spec, params);

    // Initialize with several values
    for (int i = 0; i < 10; ++i) {
        model.update(0.5 * (i % 2 == 0 ? 1 : -1));
    }

    Forecaster forecaster(model);

    // Forecast many steps ahead
    auto result = forecaster.forecast(100);

    // Variance should converge to unconditional variance
    // With α + β = 0.95, convergence is slow
    // After 100 steps, should be reasonably close
    // Check last 10 forecasts are approaching unconditional variance
    for (int i = 90; i < 100; ++i) {
        double diff = std::abs(result.variance_forecasts[i] - unconditional_var);
        REQUIRE(diff < 0.05);  // Relaxed tolerance for slow convergence
    }

    // Last forecast should be closer than first
    double diff_first = std::abs(result.variance_forecasts[0] - unconditional_var);
    double diff_last = std::abs(result.variance_forecasts[99] - unconditional_var);
    REQUIRE(diff_last < diff_first);
}

TEST(forecast_variance_garch11_convergence_rate) {
    // Test convergence behavior more precisely
    // For GARCH(1,1), variance converges geometrically to unconditional variance
    ArimaSpec arima_spec(0, 0, 0);
    GarchSpec garch_spec(1, 1);
    ArimaGarchSpec spec(arima_spec, garch_spec);

    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.0;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.15;
    params.garch_params.beta_coef[0] = 0.75;

    double unconditional_var = params.garch_params.unconditionalVariance();
    // = 0.1 / (1 - 0.15 - 0.75) = 0.1 / 0.1 = 1.0

    ArimaGarchModel model(spec, params);
    model.update(2.0);  // Start with high variance

    Forecaster forecaster(model);

    auto result = forecaster.forecast(50);

    // Check monotonic convergence
    for (int i = 1; i < 50; ++i) {
        double diff_prev = std::abs(result.variance_forecasts[i - 1] - unconditional_var);
        double diff_curr = std::abs(result.variance_forecasts[i] - unconditional_var);
        // Distance to unconditional variance should decrease
        REQUIRE(diff_curr <= diff_prev + 1e-10);  // Allow tiny numerical error
    }
}

TEST(forecast_variance_garch22) {
    // Test GARCH(2,2) forecasting
    ArimaSpec arima_spec(0, 0, 0);
    GarchSpec garch_spec(2, 2);
    ArimaGarchSpec spec(arima_spec, garch_spec);

    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.0;
    params.garch_params.omega = 0.05;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.alpha_coef[1] = 0.05;
    params.garch_params.beta_coef[0] = 0.6;
    params.garch_params.beta_coef[1] = 0.2;

    // Check stationarity: 0.1 + 0.05 + 0.6 + 0.2 = 0.95 < 1 ✓
    REQUIRE(params.garch_params.isStationary());

    ArimaGarchModel model(spec, params);

    // Initialize with several observations
    model.update(1.0);
    model.update(1.2);
    model.update(0.8);

    Forecaster forecaster(model);

    // Forecast 10 steps ahead
    auto result = forecaster.forecast(10);

    // All variances should be positive
    for (double h : result.variance_forecasts) {
        REQUIRE(h > 0.0);
    }

    // Should converge toward unconditional variance
    double unconditional_var = params.garch_params.unconditionalVariance();
    REQUIRE(std::abs(result.variance_forecasts[9] - unconditional_var) <
            std::abs(result.variance_forecasts[0] - unconditional_var));
}

// ============================================================================
// Combined ARIMA-GARCH Forecast Tests
// ============================================================================

TEST(forecast_combined_arima_garch) {
    // Test full ARIMA(1,0,1)-GARCH(1,1) forecasting
    ArimaSpec arima_spec(1, 0, 1);
    GarchSpec garch_spec(1, 1);
    ArimaGarchSpec spec(arima_spec, garch_spec);

    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.6;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.05;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchModel model(spec, params);

    // Initialize with some data
    std::vector<double> data = {1.0, 1.1, 0.9, 1.2, 0.8, 1.0};
    for (double y : data) {
        model.update(y);
    }

    Forecaster forecaster(model);

    // Forecast 20 steps ahead
    auto result = forecaster.forecast(20);

    REQUIRE(result.mean_forecasts.size() == 20);
    REQUIRE(result.variance_forecasts.size() == 20);

    // All variances should be positive
    for (double h : result.variance_forecasts) {
        REQUIRE(h > 0.0);
    }

    // Mean should converge to unconditional mean
    double unconditional_mean = 0.1 / (1.0 - 0.6);  // c / (1 - φ) = 0.25
    REQUIRE(std::abs(result.mean_forecasts[19] - unconditional_mean) < 0.05);

    // Variance should be approaching unconditional variance
    // With α + β = 0.95, convergence is slow, so just check it's reasonable
    double unconditional_var = params.garch_params.unconditionalVariance();
    // After 20 steps, should be getting closer (but may not be converged)
    double diff_first = std::abs(result.variance_forecasts[0] - unconditional_var);
    double diff_last = std::abs(result.variance_forecasts[19] - unconditional_var);
    REQUIRE(diff_last <= diff_first + 0.1);  // At least not diverging
}

int main() {
    report_test_results("Forecasting");
    return get_test_result();
}
