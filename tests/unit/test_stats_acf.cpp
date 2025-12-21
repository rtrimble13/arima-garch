#include "ag/stats/ACF.hpp"

#include <cmath>
#include <random>
#include <vector>

#include "test_framework.hpp"

// Test ACF at lag 0 is always 1
TEST(acf_lag_zero) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    double result = ag::stats::acf_at_lag(data, 0);
    REQUIRE_APPROX(result, 1.0, 1e-10);
}

// Test ACF for a constant series
TEST(acf_constant_series) {
    std::vector<double> data = {5.0, 5.0, 5.0, 5.0, 5.0};
    std::vector<double> result = ag::stats::acf(data, 3);
    REQUIRE_APPROX(result[0], 1.0, 1e-10);  // Lag 0
    REQUIRE_APPROX(result[1], 0.0, 1e-10);  // Lag 1
    REQUIRE_APPROX(result[2], 0.0, 1e-10);  // Lag 2
    REQUIRE_APPROX(result[3], 0.0, 1e-10);  // Lag 3
}

// Test ACF for a simple trend
TEST(acf_simple_trend) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> result = ag::stats::acf(data, 2);

    REQUIRE_APPROX(result[0], 1.0, 1e-10);  // Lag 0 is always 1
    // For a linear trend with short series, ACF should be positive
    REQUIRE(result[1] > 0.0);
    // We can't guarantee exact values for such a short series
}

// Test ACF for white noise (should have near-zero ACF at all lags except 0)
TEST(acf_white_noise) {
    // Generate white noise with fixed seed for reproducibility
    std::mt19937 gen(12345);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(1000);
    for (auto& val : data) {
        val = dist(gen);
    }

    std::vector<double> result = ag::stats::acf(data, 10);

    REQUIRE_APPROX(result[0], 1.0, 1e-10);  // Lag 0 is always 1

    // For white noise, ACF at non-zero lags should be close to 0
    // With 1000 samples, standard error is approximately 1/sqrt(1000) ≈ 0.032
    // We expect values within ~3 standard errors (about 0.1)
    for (std::size_t lag = 1; lag <= 10; ++lag) {
        REQUIRE(std::abs(result[lag]) < 0.15);
    }
}

// Test ACF for a simple periodic signal
TEST(acf_periodic_signal) {
    // Create a simple sine wave
    std::vector<double> data(100);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = std::sin(2.0 * M_PI * i / 10.0);  // Period of 10
    }

    std::vector<double> result = ag::stats::acf(data, 20);

    REQUIRE_APPROX(result[0], 1.0, 1e-10);  // Lag 0

    // For a periodic signal with period 10, ACF should show periodicity
    // At lag 5 (half period), should be negative (opposite phase)
    REQUIRE(result[5] < -0.5);

    // At lag 10 (full period), should be positive again (same phase)
    REQUIRE(result[10] > 0.5);
}

// Test ACF with known AR(1) process properties
// For AR(1): X_t = φ * X_{t-1} + ε_t
// ACF should decay exponentially: ACF(k) = φ^k
TEST(acf_ar1_process) {
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

    std::vector<double> result = ag::stats::acf(data, 5);

    REQUIRE_APPROX(result[0], 1.0, 1e-10);

    // Check exponential decay pattern
    for (std::size_t k = 1; k <= 5; ++k) {
        double expected = std::pow(phi, k);
        // Allow some tolerance due to finite sample and randomness
        REQUIRE(std::abs(result[k] - expected) < 0.15);
    }
}

// Test ACF error handling - empty data
TEST(acf_empty_data) {
    std::vector<double> data = {};
    bool caught = false;
    try {
        [[maybe_unused]] auto result = ag::stats::acf(data, 1);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test ACF error handling - lag too large
TEST(acf_lag_too_large) {
    std::vector<double> data = {1.0, 2.0, 3.0};
    bool caught = false;
    try {
        [[maybe_unused]] auto result = ag::stats::acf(data, 5);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test ACF error handling - lag equals data size
TEST(acf_lag_equals_size) {
    std::vector<double> data = {1.0, 2.0, 3.0};
    bool caught = false;
    try {
        [[maybe_unused]] auto result = ag::stats::acf(data, 3);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test ACF with very short series (edge case)
TEST(acf_short_series) {
    std::vector<double> data = {1.0, 2.0, 3.0};
    std::vector<double> result = ag::stats::acf(data, 1);

    REQUIRE_APPROX(result[0], 1.0, 1e-10);
    // Should still compute ACF at lag 1
    REQUIRE(result.size() == 2);
}

// Test acf_at_lag function
TEST(acf_at_lag_function) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};

    double lag0 = ag::stats::acf_at_lag(data, 0);
    REQUIRE_APPROX(lag0, 1.0, 1e-10);

    double lag1 = ag::stats::acf_at_lag(data, 1);
    REQUIRE(lag1 > 0.0);  // Should be positive for increasing sequence

    // Compare with full ACF computation
    std::vector<double> full_acf = ag::stats::acf(data, 2);
    REQUIRE_APPROX(lag1, full_acf[1], 1e-10);
}

// Test ACF with negative numbers
TEST(acf_negative_values) {
    std::vector<double> data = {-5.0, -3.0, -1.0, 1.0, 3.0, 5.0};
    std::vector<double> result = ag::stats::acf(data, 2);

    REQUIRE_APPROX(result[0], 1.0, 1e-10);
    // ACF should still be computable with negative values
    REQUIRE(!std::isnan(result[1]));
    REQUIRE(!std::isnan(result[2]));
}

// Test ACF result size
TEST(acf_result_size) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> result = ag::stats::acf(data, 5);

    // Should return max_lag + 1 values (lag 0 through max_lag)
    REQUIRE(result.size() == 6);
}

int main() {
    report_test_results("ACF Tests");
    return get_test_result();
}
