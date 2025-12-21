#include "ag/stats/Descriptive.hpp"

#include <vector>

#include "test_framework.hpp"

// Test mean calculation with a simple vector
TEST(mean_simple) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    double result = ag::stats::mean(data);
    REQUIRE_APPROX(result, 3.0, 1e-10);
}

// Test mean with negative values
TEST(mean_negative) {
    std::vector<double> data = {-2.0, -1.0, 0.0, 1.0, 2.0};
    double result = ag::stats::mean(data);
    REQUIRE_APPROX(result, 0.0, 1e-10);
}

// Test mean with single value
TEST(mean_single) {
    std::vector<double> data = {42.0};
    double result = ag::stats::mean(data);
    REQUIRE_APPROX(result, 42.0, 1e-10);
}

// Test variance calculation
// For data {1, 2, 3, 4, 5}:
// mean = 3
// squared deviations: (1-3)^2=4, (2-3)^2=1, (3-3)^2=0, (4-3)^2=1, (5-3)^2=4
// sum = 10
// sample variance = 10 / (5-1) = 2.5
TEST(variance_simple) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    double result = ag::stats::variance(data);
    REQUIRE_APPROX(result, 2.5, 1e-10);
}

// Test variance with constant values (should be 0)
TEST(variance_constant) {
    std::vector<double> data = {5.0, 5.0, 5.0, 5.0};
    double result = ag::stats::variance(data);
    REQUIRE_APPROX(result, 0.0, 1e-10);
}

// Test variance with two values
// For data {1, 3}:
// mean = 2
// squared deviations: (1-2)^2=1, (3-2)^2=1
// sample variance = 2 / (2-1) = 2.0
TEST(variance_two_values) {
    std::vector<double> data = {1.0, 3.0};
    double result = ag::stats::variance(data);
    REQUIRE_APPROX(result, 2.0, 1e-10);
}

// Test skewness with symmetric distribution (should be near 0)
TEST(skewness_symmetric) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    double result = ag::stats::skewness(data);
    REQUIRE_APPROX(result, 0.0, 1e-10);
}

// Test skewness with right-skewed distribution
// For data {1, 2, 2, 3, 10}:
// This is right-skewed (positive skewness)
TEST(skewness_right_skewed) {
    std::vector<double> data = {1.0, 2.0, 2.0, 3.0, 10.0};
    double result = ag::stats::skewness(data);
    // Expected to be positive
    REQUIRE(result > 0.5);
}

// Test skewness with left-skewed distribution
TEST(skewness_left_skewed) {
    std::vector<double> data = {1.0, 8.0, 9.0, 9.0, 10.0};
    double result = ag::stats::skewness(data);
    // Expected to be negative
    REQUIRE(result < -0.5);
}

// Test skewness with constant values (should be 0)
TEST(skewness_constant) {
    std::vector<double> data = {5.0, 5.0, 5.0, 5.0, 5.0};
    double result = ag::stats::skewness(data);
    REQUIRE_APPROX(result, 0.0, 1e-10);
}

// Test kurtosis with normal-like distribution
// For a uniform distribution, excess kurtosis should be negative
TEST(kurtosis_uniform_like) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double result = ag::stats::kurtosis(data);
    // Uniform distribution has excess kurtosis around -1.2
    REQUIRE(result < 0.0);
    REQUIRE(result > -2.0);
}

// Test kurtosis with heavy tails (should have positive excess kurtosis)
// Data with outliers should have higher kurtosis
TEST(kurtosis_heavy_tails) {
    std::vector<double> data = {1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 20.0};
    double result = ag::stats::kurtosis(data);
    // Should have positive excess kurtosis due to outliers
    REQUIRE(result > 0.0);
}

// Test kurtosis with constant values (should be 0)
TEST(kurtosis_constant) {
    std::vector<double> data = {5.0, 5.0, 5.0, 5.0, 5.0};
    double result = ag::stats::kurtosis(data);
    REQUIRE_APPROX(result, 0.0, 1e-10);
}

// Test with known statistical values
// For standard normal-like data, we can verify against expected values
TEST(descriptive_known_values) {
    // Data: {2, 4, 4, 4, 5, 5, 7, 9}
    // Mean = 40/8 = 5.0
    // Variance = sum((x-5)^2) / (n-1) = (9+1+1+1+0+0+4+16) / 7 = 32/7 ≈ 4.571
    std::vector<double> data = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    
    double m = ag::stats::mean(data);
    REQUIRE_APPROX(m, 5.0, 1e-10);
    
    double v = ag::stats::variance(data);
    REQUIRE_APPROX(v, 32.0 / 7.0, 1e-10);
}

// Test error handling - empty data
TEST(mean_empty_error) {
    std::vector<double> data = {};
    bool caught = false;
    try {
        [[maybe_unused]] double result = ag::stats::mean(data);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test error handling - variance with insufficient data
TEST(variance_insufficient_data) {
    std::vector<double> data = {1.0};
    bool caught = false;
    try {
        [[maybe_unused]] double result = ag::stats::variance(data);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test error handling - skewness with insufficient data
TEST(skewness_insufficient_data) {
    std::vector<double> data = {1.0, 2.0};
    bool caught = false;
    try {
        [[maybe_unused]] double result = ag::stats::skewness(data);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test error handling - kurtosis with insufficient data
TEST(kurtosis_insufficient_data) {
    std::vector<double> data = {1.0, 2.0, 3.0};
    bool caught = false;
    try {
        [[maybe_unused]] double result = ag::stats::kurtosis(data);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

int main() {
    report_test_results("Descriptive Statistics Tests");
    return get_test_result();
}
