#include "ag/stats/PACF.hpp"

#include <cmath>
#include <random>
#include <vector>

#include "test_framework.hpp"

// Test PACF error handling - lag 0 is not valid
TEST(pacf_lag_zero_invalid) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    bool caught = false;
    try {
        [[maybe_unused]] double result = ag::stats::pacf_at_lag(data, 0);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test PACF for a constant series
TEST(pacf_constant_series) {
    std::vector<double> data = {5.0, 5.0, 5.0, 5.0, 5.0};
    std::vector<double> result = ag::stats::pacf(data, 3);

    // For constant series, PACF should be 0 at all lags
    REQUIRE_APPROX(result[0], 0.0, 1e-10);  // Lag 1
    REQUIRE_APPROX(result[1], 0.0, 1e-10);  // Lag 2
    REQUIRE_APPROX(result[2], 0.0, 1e-10);  // Lag 3
}

// Test PACF for white noise (should have near-zero PACF at all lags)
TEST(pacf_white_noise) {
    // Generate white noise with fixed seed for reproducibility
    std::mt19937 gen(12345);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(1000);
    for (auto& val : data) {
        val = dist(gen);
    }

    std::vector<double> result = ag::stats::pacf(data, 10);

    // For white noise, PACF at all lags should be close to 0
    // With 1000 samples, standard error is approximately 1/sqrt(1000) ≈ 0.032
    // We expect values within ~3 standard errors (about 0.1)
    for (std::size_t lag = 0; lag < 10; ++lag) {
        REQUIRE(std::abs(result[lag]) < 0.15);
    }
}

// Test PACF with known AR(1) process properties
// For AR(1): X_t = φ * X_{t-1} + ε_t
// PACF should have: PACF(1) = φ, PACF(k) ≈ 0 for k > 1
TEST(pacf_ar1_process) {
    // Simulate AR(1) with φ = 0.7
    const double phi = 0.7;
    const int n = 500;

    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(n);
    data[0] = dist(gen);
    for (int i = 1; i < n; ++i) {
        data[i] = phi * data[i - 1] + dist(gen);
    }

    std::vector<double> result = ag::stats::pacf(data, 5);

    // PACF at lag 1 should be approximately φ
    REQUIRE(std::abs(result[0] - phi) < 0.1);

    // PACF at lags > 1 should be close to 0 for AR(1)
    for (std::size_t k = 1; k < 5; ++k) {
        REQUIRE(std::abs(result[k]) < 0.15);
    }
}

// Test PACF with known AR(2) process properties
// For AR(2): X_t = φ1 * X_{t-1} + φ2 * X_{t-2} + ε_t
// PACF should have non-zero values at lags 1 and 2, then cut off
TEST(pacf_ar2_process) {
    // Simulate AR(2) with φ1 = 0.5, φ2 = 0.3
    const double phi1 = 0.5;
    const double phi2 = 0.3;
    const int n = 500;

    std::mt19937 gen(123);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(n);
    data[0] = dist(gen);
    data[1] = phi1 * data[0] + dist(gen);
    for (int i = 2; i < n; ++i) {
        data[i] = phi1 * data[i - 1] + phi2 * data[i - 2] + dist(gen);
    }

    std::vector<double> result = ag::stats::pacf(data, 6);

    // PACF at lags 1 and 2 should be significant
    REQUIRE(std::abs(result[0]) > 0.15);  // Lag 1
    REQUIRE(std::abs(result[1]) > 0.15);  // Lag 2

    // PACF at lags > 2 should be close to 0 for AR(2)
    for (std::size_t k = 2; k < 6; ++k) {
        REQUIRE(std::abs(result[k]) < 0.2);
    }
}

// Test PACF error handling - empty data
TEST(pacf_empty_data) {
    std::vector<double> data = {};
    bool caught = false;
    try {
        [[maybe_unused]] auto result = ag::stats::pacf(data, 1);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test PACF error handling - lag too large
TEST(pacf_lag_too_large) {
    std::vector<double> data = {1.0, 2.0, 3.0};
    bool caught = false;
    try {
        [[maybe_unused]] auto result = ag::stats::pacf(data, 5);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test PACF error handling - lag equals data size
TEST(pacf_lag_equals_size) {
    std::vector<double> data = {1.0, 2.0, 3.0};
    bool caught = false;
    try {
        [[maybe_unused]] auto result = ag::stats::pacf(data, 3);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test PACF with very short series (edge case)
TEST(pacf_short_series) {
    std::vector<double> data = {1.0, 2.0, 3.0};
    std::vector<double> result = ag::stats::pacf(data, 1);

    // Should compute PACF at lag 1
    REQUIRE(result.size() == 1);
    REQUIRE(!std::isnan(result[0]));
}

// Test PACF with max_lag = 0 (edge case)
TEST(pacf_zero_max_lag) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> result = ag::stats::pacf(data, 0);

    // Should return empty vector
    REQUIRE(result.size() == 0);
}

// Test pacf_at_lag function
TEST(pacf_at_lag_function) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    double lag1 = ag::stats::pacf_at_lag(data, 1);
    double lag2 = ag::stats::pacf_at_lag(data, 2);

    // Compare with full PACF computation
    std::vector<double> full_pacf = ag::stats::pacf(data, 2);
    REQUIRE_APPROX(lag1, full_pacf[0], 1e-10);
    REQUIRE_APPROX(lag2, full_pacf[1], 1e-10);
}

// Test PACF with negative numbers
TEST(pacf_negative_values) {
    std::vector<double> data = {-5.0, -3.0, -1.0, 1.0, 3.0, 5.0};
    std::vector<double> result = ag::stats::pacf(data, 2);

    // PACF should still be computable with negative values
    REQUIRE(!std::isnan(result[0]));
    REQUIRE(!std::isnan(result[1]));
}

// Test PACF result size
TEST(pacf_result_size) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> result = ag::stats::pacf(data, 5);

    // Should return max_lag values (lag 1 through max_lag)
    REQUIRE(result.size() == 5);
}

// Test PACF for simple trend
TEST(pacf_simple_trend) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> result = ag::stats::pacf(data, 3);

    // For a trend, PACF at lag 1 should be high
    REQUIRE(result[0] > 0.5);

    // Should compute all requested lags
    REQUIRE(result.size() == 3);
}

// Test numerical stability with near-singular case
TEST(pacf_numerical_stability) {
    // Create data that might cause numerical issues
    std::vector<double> data(20);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = 1.0 + 0.001 * i;  // Very gentle slope
    }

    std::vector<double> result = ag::stats::pacf(data, 5);

    // Should not produce NaN or Inf
    for (const auto& val : result) {
        REQUIRE(!std::isnan(val));
        REQUIRE(!std::isinf(val));
    }
}

int main() {
    report_test_results("PACF Tests");
    return get_test_result();
}
