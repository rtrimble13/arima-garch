#include "ag/estimation/Likelihood.hpp"

#include <cmath>
#include <random>

#include "test_framework.hpp"

using ag::estimation::ArimaGarchLikelihood;
using ag::models::ArimaGarchSpec;
using ag::models::arima::ArimaParameters;
using ag::models::garch::GarchParameters;

// ============================================================================
// ArimaGarchLikelihood Construction Tests
// ============================================================================

// Test construction with valid spec
TEST(likelihood_construction) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchLikelihood likelihood(spec);

    REQUIRE(likelihood.getSpec().arimaSpec.p == 1);
    REQUIRE(likelihood.getSpec().arimaSpec.d == 0);
    REQUIRE(likelihood.getSpec().arimaSpec.q == 1);
    REQUIRE(likelihood.getSpec().garchSpec.p == 1);
    REQUIRE(likelihood.getSpec().garchSpec.q == 1);
}

// ============================================================================
// Likelihood Computation Tests - White Noise with Constant Variance
// ============================================================================

// Test ARIMA(0,0,0)-GARCH(1,1) with small coefficients (nearly constant variance)
TEST(likelihood_white_noise_constant_variance) {
    ArimaGarchSpec spec(0, 0, 0, 1, 1);
    ArimaGarchLikelihood likelihood(spec);

    // Simple white noise data with zero mean
    std::vector<double> data = {0.5, -0.3, 0.2, -0.1, 0.4, -0.6, 0.1, 0.3, -0.2, 0.5};

    ArimaParameters arima_params(0, 0);
    arima_params.intercept = 0.0;

    GarchParameters garch_params(1, 1);
    garch_params.omega = 0.095;          // Nearly constant variance h_t ≈ 0.1
    garch_params.alpha_coef[0] = 0.001;  // Very small ARCH coefficient
    garch_params.beta_coef[0] = 0.001;   // Very small GARCH coefficient

    double nll = likelihood.computeNegativeLogLikelihood(data.data(), data.size(), arima_params,
                                                         garch_params);

    // Verify NLL is finite (can be positive or negative)
    REQUIRE(std::isfinite(nll));
}

// ============================================================================
// Likelihood Computation Tests - ARIMA with Constant Variance
// ============================================================================

// Test ARIMA(1,0,0)-GARCH(1,1) - AR(1) with nearly constant variance
TEST(likelihood_ar1_constant_variance) {
    ArimaGarchSpec spec(1, 0, 0, 1, 1);
    ArimaGarchLikelihood likelihood(spec);

    // Generate AR(1) data: y_t = 0.5 + 0.7*y_{t-1} + ε_t
    std::vector<double> data = {1.0, 1.2, 1.35, 1.445, 1.5115, 1.55805, 1.590635};

    ArimaParameters arima_params(1, 0);
    arima_params.intercept = 0.5;
    arima_params.ar_coef[0] = 0.7;

    GarchParameters garch_params(1, 1);
    garch_params.omega = 0.048;  // Nearly constant variance
    garch_params.alpha_coef[0] = 0.001;
    garch_params.beta_coef[0] = 0.001;

    double nll = likelihood.computeNegativeLogLikelihood(data.data(), data.size(), arima_params,
                                                         garch_params);

    // NLL should be finite (can be positive or negative)
    REQUIRE(std::isfinite(nll));
}

// ============================================================================
// Likelihood Computation Tests - GARCH Effects
// ============================================================================

// Test ARIMA(0,0,0)-GARCH(1,1) - White noise with time-varying variance
TEST(likelihood_white_noise_garch11) {
    ArimaGarchSpec spec(0, 0, 0, 1, 1);
    ArimaGarchLikelihood likelihood(spec);

    // White noise data with zero mean
    std::vector<double> data = {0.5, -0.8, 0.3, -0.4, 0.6, -0.5, 0.2, 0.7, -0.3, 0.4};

    ArimaParameters arima_params(0, 0);
    arima_params.intercept = 0.0;

    GarchParameters garch_params(1, 1);
    garch_params.omega = 0.01;
    garch_params.alpha_coef[0] = 0.1;  // ARCH(1) coefficient
    garch_params.beta_coef[0] = 0.85;  // GARCH(1) coefficient

    double nll = likelihood.computeNegativeLogLikelihood(data.data(), data.size(), arima_params,
                                                         garch_params);

    // NLL should be finite (can be positive or negative)
    REQUIRE(std::isfinite(nll));
}

// ============================================================================
// Likelihood Comparison Tests - Parameter Sensitivity
// ============================================================================

// Test that NLL decreases when variance matches data better
TEST(likelihood_variance_sensitivity) {
    ArimaGarchSpec spec(0, 0, 0, 1, 1);
    ArimaGarchLikelihood likelihood(spec);

    // Data with moderate variance
    std::vector<double> data = {0.3, -0.2, 0.4, -0.3, 0.5, -0.4, 0.2, 0.3, -0.1, 0.4};

    ArimaParameters arima_params(0, 0);
    arima_params.intercept = 0.0;

    // Compute sample variance
    double mean = 0.0;
    double variance = 0.0;
    for (double x : data) {
        variance += x * x;
    }
    variance /= data.size();

    // Test with variance much smaller than true (high persistence)
    GarchParameters params_low(1, 1);
    params_low.omega = variance * 0.002;  // Very small omega
    params_low.alpha_coef[0] = 0.05;
    params_low.beta_coef[0] = 0.93;  // High persistence
    double nll_low =
        likelihood.computeNegativeLogLikelihood(data.data(), data.size(), arima_params, params_low);

    // Test with variance close to true (low persistence)
    GarchParameters params_good(1, 1);
    params_good.omega = variance * 0.9;  // Close to sample variance
    params_good.alpha_coef[0] = 0.05;
    params_good.beta_coef[0] = 0.05;  // Low persistence
    double nll_good = likelihood.computeNegativeLogLikelihood(data.data(), data.size(),
                                                              arima_params, params_good);

    // Test with variance much larger than true (high persistence)
    GarchParameters params_high(1, 1);
    params_high.omega = variance * 0.1;  // Large omega
    params_high.alpha_coef[0] = 0.05;
    params_high.beta_coef[0] = 0.93;  // High persistence
    double nll_high = likelihood.computeNegativeLogLikelihood(data.data(), data.size(),
                                                              arima_params, params_high);

    // NLL should be smallest for variance close to true
    REQUIRE(nll_good < nll_low);
    REQUIRE(nll_good < nll_high);
}

// Test that NLL decreases near true ARIMA parameters
TEST(likelihood_arima_parameter_sensitivity) {
    ArimaGarchSpec spec(1, 0, 0, 1, 1);
    ArimaGarchLikelihood likelihood(spec);

    // Generate synthetic AR(1) data: y_t = 1.0 + 0.6*y_{t-1} + ε_t
    std::mt19937 rng(12345);
    std::normal_distribution<double> dist(0.0, 0.3);

    std::vector<double> data;
    data.reserve(100);
    double y = 0.0;
    for (int i = 0; i < 100; ++i) {
        y = 1.0 + 0.6 * y + dist(rng);
        data.push_back(y);
    }

    GarchParameters garch_params(1, 1);
    garch_params.omega = 0.09;
    garch_params.alpha_coef[0] = 0.005;
    garch_params.beta_coef[0] = 0.005;

    // Test with true parameters
    ArimaParameters params_true(1, 0);
    params_true.intercept = 1.0;
    params_true.ar_coef[0] = 0.6;
    double nll_true = likelihood.computeNegativeLogLikelihood(data.data(), data.size(), params_true,
                                                              garch_params);

    // Test with incorrect parameters
    ArimaParameters params_wrong1(1, 0);
    params_wrong1.intercept = 0.5;
    params_wrong1.ar_coef[0] = 0.6;
    double nll_wrong1 = likelihood.computeNegativeLogLikelihood(data.data(), data.size(),
                                                                params_wrong1, garch_params);

    ArimaParameters params_wrong2(1, 0);
    params_wrong2.intercept = 1.0;
    params_wrong2.ar_coef[0] = 0.3;
    double nll_wrong2 = likelihood.computeNegativeLogLikelihood(data.data(), data.size(),
                                                                params_wrong2, garch_params);

    // NLL should be smaller for true parameters
    REQUIRE(nll_true < nll_wrong1);
    REQUIRE(nll_true < nll_wrong2);
}

// Test that NLL decreases near true GARCH parameters
TEST(likelihood_garch_parameter_sensitivity) {
    ArimaGarchSpec spec(0, 0, 0, 1, 1);
    ArimaGarchLikelihood likelihood(spec);

    // Generate synthetic GARCH(1,1) data
    std::mt19937 rng(54321);
    std::normal_distribution<double> std_normal(0.0, 1.0);

    double omega_true = 0.02;
    double alpha_true = 0.15;
    double beta_true = 0.8;

    std::vector<double> data;
    data.reserve(200);
    double h = omega_true / (1.0 - alpha_true - beta_true);  // Unconditional variance
    for (int i = 0; i < 200; ++i) {
        double z = std_normal(rng);
        double eps = std::sqrt(h) * z;
        data.push_back(eps);
        h = omega_true + alpha_true * eps * eps + beta_true * h;
    }

    ArimaParameters arima_params(0, 0);
    arima_params.intercept = 0.0;

    // Test with true parameters
    GarchParameters params_true(1, 1);
    params_true.omega = omega_true;
    params_true.alpha_coef[0] = alpha_true;
    params_true.beta_coef[0] = beta_true;
    double nll_true = likelihood.computeNegativeLogLikelihood(data.data(), data.size(),
                                                              arima_params, params_true);

    // Test with incorrect parameters
    GarchParameters params_wrong1(1, 1);
    params_wrong1.omega = omega_true * 2.0;
    params_wrong1.alpha_coef[0] = alpha_true;
    params_wrong1.beta_coef[0] = beta_true * 0.5;
    double nll_wrong1 = likelihood.computeNegativeLogLikelihood(data.data(), data.size(),
                                                                arima_params, params_wrong1);

    GarchParameters params_wrong2(1, 1);
    params_wrong2.omega = omega_true;
    params_wrong2.alpha_coef[0] = alpha_true * 0.5;
    params_wrong2.beta_coef[0] = beta_true;
    double nll_wrong2 = likelihood.computeNegativeLogLikelihood(data.data(), data.size(),
                                                                arima_params, params_wrong2);

    // NLL should be smaller for true parameters
    REQUIRE(nll_true < nll_wrong1);
    REQUIRE(nll_true < nll_wrong2);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

// Test null data pointer
TEST(likelihood_null_data) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchLikelihood likelihood(spec);

    ArimaParameters arima_params(1, 1);
    GarchParameters garch_params(1, 1);
    garch_params.omega = 0.1;

    bool caught = false;
    try {
        likelihood.computeNegativeLogLikelihood(nullptr, 10, arima_params, garch_params);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test zero data size
TEST(likelihood_zero_size) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchLikelihood likelihood(spec);

    std::vector<double> data = {1.0};
    ArimaParameters arima_params(1, 1);
    GarchParameters garch_params(1, 1);
    garch_params.omega = 0.1;

    bool caught = false;
    try {
        likelihood.computeNegativeLogLikelihood(data.data(), 0, arima_params, garch_params);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test invalid GARCH parameters (negative omega)
TEST(likelihood_invalid_garch_params) {
    ArimaGarchSpec spec(0, 0, 0, 1, 1);
    ArimaGarchLikelihood likelihood(spec);

    std::vector<double> data = {0.5, -0.3, 0.2};
    ArimaParameters arima_params(0, 0);
    GarchParameters garch_params(1, 1);
    garch_params.omega = -0.1;  // Invalid: omega must be > 0
    garch_params.alpha_coef[0] = 0.1;
    garch_params.beta_coef[0] = 0.8;

    bool caught = false;
    try {
        likelihood.computeNegativeLogLikelihood(data.data(), data.size(), arima_params,
                                                garch_params);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

int main() {
    report_test_results("Likelihood Tests");
    return get_test_result();
}
