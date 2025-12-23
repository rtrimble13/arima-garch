#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"

#include <cmath>
#include <random>

#include "test_framework.hpp"

using ag::models::ArimaGarchSpec;
using ag::models::composite::ArimaGarchModel;
using ag::models::composite::ArimaGarchParameters;

// ============================================================================
// ArimaGarchModel Construction Tests
// ============================================================================

// Test basic construction
TEST(arimagarch_model_construction) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    // Set some valid parameters
    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.5;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchModel model(spec, params);

    // Verify spec is stored correctly
    REQUIRE(model.getSpec().arimaSpec.p == 1);
    REQUIRE(model.getSpec().arimaSpec.d == 0);
    REQUIRE(model.getSpec().arimaSpec.q == 1);
    REQUIRE(model.getSpec().garchSpec.p == 1);
    REQUIRE(model.getSpec().garchSpec.q == 1);
}

// Test construction with invalid GARCH parameters
TEST(arimagarch_model_invalid_garch_params) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.5;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = -0.1;  // Invalid: omega must be > 0
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    bool caught_exception = false;
    try {
        ArimaGarchModel model(spec, params);
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }

    REQUIRE(caught_exception);
}

// ============================================================================
// ArimaGarchModel Update Tests - White Noise Mean, Simple Variance
// ============================================================================

// Test ARIMA(0,0,0)-GARCH(1,1) - white noise mean, GARCH(1,1) variance
TEST(arimagarch_update_white_noise_garch11) {
    ArimaGarchSpec spec(0, 0, 0, 1, 1);
    ArimaGarchParameters params(spec);

    // White noise mean with zero intercept
    params.arima_params.intercept = 0.0;

    // GARCH(1,1): h_t = 0.5 + 0.3*ε²_{t-1} + 0.1*h_{t-1}
    params.garch_params.omega = 0.5;
    params.garch_params.alpha_coef[0] = 0.3;
    params.garch_params.beta_coef[0] = 0.1;

    ArimaGarchModel model(spec, params);

    // Process first observation
    double y_1 = 1.0;
    auto output_1 = model.update(y_1);

    // μ_1 should be 0 (white noise with zero intercept)
    REQUIRE_APPROX(output_1.mu_t, 0.0, 1e-10);

    // h_1 should be initialized (omega / (1 - alpha) if stationary, or sample variance)
    // Since it's the first update, h_1 = 0.5 + 0.3 * 0 = 0.5 (initialized with 0 squared residual)
    REQUIRE(output_1.h_t > 0.0);

    // Process second observation
    double y_2 = 2.0;
    auto output_2 = model.update(y_2);

    // μ_2 should still be 0
    REQUIRE_APPROX(output_2.mu_t, 0.0, 1e-10);

    // h_2 = 0.5 + 0.3 * ε²_1 + 0.1 * h_1 where ε_1 = y_1 - μ_1 = 1.0 - 0.0 = 1.0
    // h_2 = 0.5 + 0.3 * 1.0 + 0.1 * h_1
    double expected_h_2 = 0.5 + 0.3 * (1.0 * 1.0) + 0.1 * output_1.h_t;
    REQUIRE_APPROX(output_2.h_t, expected_h_2, 1e-10);

    // Process third observation
    double y_3 = 1.5;
    auto output_3 = model.update(y_3);

    // μ_3 should still be 0
    REQUIRE_APPROX(output_3.mu_t, 0.0, 1e-10);

    // h_3 = 0.5 + 0.3 * ε²_2 + 0.1 * h_2 where ε_2 = y_2 - μ_2 = 2.0 - 0.0 = 2.0
    // h_3 = 0.5 + 0.3 * 4.0 + 0.1 * h_2
    double expected_h_3 = 0.5 + 0.3 * (2.0 * 2.0) + 0.1 * output_2.h_t;
    REQUIRE_APPROX(output_3.h_t, expected_h_3, 1e-10);
}

// ============================================================================
// ArimaGarchModel Update Tests - AR(1)-GARCH(1,1)
// ============================================================================

// Test AR(1)-GARCH(1,1) model update
TEST(arimagarch_update_ar1_garch11) {
    ArimaGarchSpec spec(1, 0, 0, 1, 1);
    ArimaGarchParameters params(spec);

    // AR(1): y_t = 0.2 + 0.6*y_{t-1} + ε_t
    params.arima_params.intercept = 0.2;
    params.arima_params.ar_coef[0] = 0.6;

    // GARCH(1,1): h_t = 0.1 + 0.15*ε²_{t-1} + 0.75*h_{t-1}
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.15;
    params.garch_params.beta_coef[0] = 0.75;

    ArimaGarchModel model(spec, params);

    // Process first observation
    double y_1 = 1.0;
    auto output_1 = model.update(y_1);

    // μ_1 = 0.2 + 0.6 * 0 = 0.2 (no history yet)
    REQUIRE_APPROX(output_1.mu_t, 0.2, 1e-10);

    // ε_1 = 1.0 - 0.2 = 0.8
    double eps_1 = y_1 - output_1.mu_t;
    REQUIRE_APPROX(eps_1, 0.8, 1e-10);

    // h_1 should be positive
    REQUIRE(output_1.h_t > 0.0);

    // Process second observation
    double y_2 = 2.0;
    auto output_2 = model.update(y_2);

    // μ_2 = 0.2 + 0.6 * 1.0 = 0.8
    REQUIRE_APPROX(output_2.mu_t, 0.8, 1e-10);

    // ε_2 = 2.0 - 0.8 = 1.2
    double eps_2 = y_2 - output_2.mu_t;
    REQUIRE_APPROX(eps_2, 1.2, 1e-10);

    // h_2 = 0.1 + 0.15 * ε²_1 + 0.75 * h_1
    double expected_h_2 = 0.1 + 0.15 * (eps_1 * eps_1) + 0.75 * output_1.h_t;
    REQUIRE_APPROX(output_2.h_t, expected_h_2, 1e-10);

    // Process third observation
    double y_3 = 1.5;
    auto output_3 = model.update(y_3);

    // μ_3 = 0.2 + 0.6 * 2.0 = 1.4
    REQUIRE_APPROX(output_3.mu_t, 1.4, 1e-10);

    // ε_3 = 1.5 - 1.4 = 0.1
    double eps_3 = y_3 - output_3.mu_t;
    REQUIRE_APPROX(eps_3, 0.1, 1e-10);

    // h_3 = 0.1 + 0.15 * ε²_2 + 0.75 * h_2
    double expected_h_3 = 0.1 + 0.15 * (eps_2 * eps_2) + 0.75 * output_2.h_t;
    REQUIRE_APPROX(output_3.h_t, expected_h_3, 1e-10);
}

// ============================================================================
// ArimaGarchModel Update Tests - ARMA(1,1)-GARCH(1,1)
// ============================================================================

// Test ARMA(1,1)-GARCH(1,1) model update
TEST(arimagarch_update_arma11_garch11) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    // ARMA(1,1): y_t = 0.1 + 0.5*y_{t-1} + ε_t + 0.3*ε_{t-1}
    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.5;
    params.arima_params.ma_coef[0] = 0.3;

    // GARCH(1,1): h_t = 0.05 + 0.1*ε²_{t-1} + 0.85*h_{t-1}
    params.garch_params.omega = 0.05;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchModel model(spec, params);

    // Process first observation
    double y_1 = 1.0;
    auto output_1 = model.update(y_1);

    // μ_1 = 0.1 + 0.5 * 0 + 0.3 * 0 = 0.1 (no history)
    REQUIRE_APPROX(output_1.mu_t, 0.1, 1e-10);

    // ε_1 = 1.0 - 0.1 = 0.9
    double eps_1 = y_1 - output_1.mu_t;

    // Process second observation
    double y_2 = 1.5;
    auto output_2 = model.update(y_2);

    // μ_2 = 0.1 + 0.5 * 1.0 + 0.3 * 0.9 = 0.1 + 0.5 + 0.27 = 0.87
    double expected_mu_2 = 0.1 + 0.5 * y_1 + 0.3 * eps_1;
    REQUIRE_APPROX(output_2.mu_t, expected_mu_2, 1e-10);

    // Both outputs should have positive variance
    REQUIRE(output_1.h_t > 0.0);
    REQUIRE(output_2.h_t > 0.0);
}

// ============================================================================
// ArimaGarchModel Sequential Update Tests
// ============================================================================

// Test that sequential updates don't cause reallocations or crashes
TEST(arimagarch_sequential_updates_stability) {
    ArimaGarchSpec spec(2, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    // ARMA(2,1): y_t = 0.1 + 0.4*y_{t-1} + 0.3*y_{t-2} + ε_t + 0.2*ε_{t-1}
    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.4;
    params.arima_params.ar_coef[1] = 0.3;
    params.arima_params.ma_coef[0] = 0.2;

    // GARCH(1,1): h_t = 0.05 + 0.1*ε²_{t-1} + 0.85*h_{t-1}
    params.garch_params.omega = 0.05;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchModel model(spec, params);

    // Generate a longer series
    std::vector<double> observations(100);
    std::mt19937 gen(42);
    std::normal_distribution<> d(1.0, 0.5);
    for (auto& obs : observations) {
        obs = d(gen);
    }

    // Process all observations
    for (std::size_t i = 0; i < observations.size(); ++i) {
        auto output = model.update(observations[i]);

        // Verify that both mu_t and h_t are computed
        // h_t should always be positive
        REQUIRE(output.h_t > 0.0);

        // Variance should remain bounded (stability check)
        REQUIRE(output.h_t < 100.0);
    }
}

// Test that variances remain positive throughout updates
TEST(arimagarch_variance_positivity) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.0;
    params.arima_params.ar_coef[0] = 0.5;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchModel model(spec, params);

    // Process random observations
    std::vector<double> observations = {1.0, -0.5, 2.0, -1.5, 0.8, 1.2, -0.3, 0.5};

    for (double y_t : observations) {
        auto output = model.update(y_t);
        REQUIRE(output.h_t > 0.0);
    }
}

// ============================================================================
// ArimaGarchModel State Access Tests
// ============================================================================

// Test that states can be accessed
TEST(arimagarch_state_access) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.5;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchModel model(spec, params);

    // Access states
    const auto& arima_state = model.getArimaState();
    const auto& garch_state = model.getGarchState();

    // Verify states are initialized
    REQUIRE(arima_state.isInitialized());
    REQUIRE(garch_state.isInitialized());

    // Process an observation
    model.update(1.0);

    // States should still be accessible
    const auto& arima_state_after = model.getArimaState();
    const auto& garch_state_after = model.getGarchState();

    REQUIRE(arima_state_after.isInitialized());
    REQUIRE(garch_state_after.isInitialized());
}

int main() {
    report_test_results("ARIMA-GARCH Composite Model");
    return get_test_result();
}
