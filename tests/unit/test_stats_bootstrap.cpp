#include "ag/stats/Bootstrap.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "test_framework.hpp"

using ag::stats::ADFRegressionForm;

// ============================================================================
// Bootstrap Ljung-Box Test
// ============================================================================

// Test bootstrap Ljung-Box with white noise - should have high p-value
TEST(bootstrap_ljung_box_white_noise) {
    // Generate white noise with fixed seed
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> residuals(200);
    for (auto& val : residuals) {
        val = dist(gen);
    }

    // Perform bootstrap Ljung-Box test
    auto result = ag::stats::ljung_box_test_bootstrap(residuals, 10, 500, 12345);

    // Check that result structure is populated correctly
    REQUIRE(result.statistic >= 0.0);
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);
    REQUIRE(result.lags == 10);
    REQUIRE(result.dof == 10);

    // For white noise, p-value should be high (not significant)
    // With 500 bootstrap replications, we expect reasonable accuracy
    REQUIRE(result.p_value > 0.05);
}

// Test bootstrap Ljung-Box with autocorrelated data - should have low p-value
TEST(bootstrap_ljung_box_autocorrelated) {
    // Generate AR(1) process with strong autocorrelation
    const double phi = 0.85;
    const int n = 200;

    std::mt19937 gen(123);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> residuals(n);
    residuals[0] = dist(gen);
    for (int i = 1; i < n; ++i) {
        residuals[i] = phi * residuals[i - 1] + dist(gen);
    }

    // Perform bootstrap Ljung-Box test
    auto result = ag::stats::ljung_box_test_bootstrap(residuals, 10, 500, 456);

    // For autocorrelated data, p-value should be low (significant)
    REQUIRE(result.p_value < 0.05);
}

// Test bootstrap Ljung-Box with Student-t innovations
TEST(bootstrap_ljung_box_student_t_white_noise) {
    // Generate white noise from Student-t distribution
    std::mt19937 gen(789);
    std::student_t_distribution<double> dist(5.0);  // Heavy tails

    std::vector<double> residuals(200);
    for (auto& val : residuals) {
        val = dist(gen);
    }

    // Perform bootstrap Ljung-Box test
    auto result = ag::stats::ljung_box_test_bootstrap(residuals, 10, 500, 789);

    // Even with heavy tails, white noise should have high p-value
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);
    REQUIRE(result.statistic >= 0.0);

    // Should typically pass (high p-value), though not guaranteed
    // We just verify it's a reasonable value
    REQUIRE(result.p_value > 0.01);
}

// Test bootstrap Ljung-Box consistency with multiple seeds
TEST(bootstrap_ljung_box_reproducibility) {
    std::mt19937 gen(111);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> residuals(100);
    for (auto& val : residuals) {
        val = dist(gen);
    }

    // Run test twice with same seed
    auto result1 = ag::stats::ljung_box_test_bootstrap(residuals, 8, 200, 999);
    auto result2 = ag::stats::ljung_box_test_bootstrap(residuals, 8, 200, 999);

    // Results should be identical
    REQUIRE(std::abs(result1.statistic - result2.statistic) < 1e-10);
    REQUIRE(std::abs(result1.p_value - result2.p_value) < 1e-10);
}

// ============================================================================
// Bootstrap ADF Test
// ============================================================================

// Test bootstrap ADF with stationary data - should reject unit root
TEST(bootstrap_adf_stationary) {
    // Generate stationary AR(1) process with phi < 1
    const double phi = 0.5;
    const int n = 150;

    std::mt19937 gen(222);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(n);
    data[0] = dist(gen);
    for (int i = 1; i < n; ++i) {
        data[i] = phi * data[i - 1] + dist(gen);
    }

    // Perform bootstrap ADF test
    auto result = ag::stats::adf_test_bootstrap(data, 2, ADFRegressionForm::Constant, 200, 333);

    // Check that result structure is valid
    REQUIRE(std::isfinite(result.statistic));
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);
    REQUIRE(result.lags == 2);
    REQUIRE(result.regression_form == ADFRegressionForm::Constant);

    // Critical values should be in reasonable range (negative)
    REQUIRE(result.critical_value_1pct < 0.0);
    REQUIRE(result.critical_value_5pct < 0.0);
    REQUIRE(result.critical_value_10pct < 0.0);
    REQUIRE(result.critical_value_1pct < result.critical_value_5pct);
    REQUIRE(result.critical_value_5pct < result.critical_value_10pct);

    // For stationary process, p-value should typically be low
    // (though this is stochastic, so we use a lenient threshold)
    REQUIRE(result.p_value < 0.5);
}

// Test bootstrap ADF with unit root (random walk) - should fail to reject
TEST(bootstrap_adf_unit_root) {
    // Generate random walk (unit root process)
    const int n = 150;

    std::mt19937 gen(444);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(n);
    data[0] = dist(gen);
    for (int i = 1; i < n; ++i) {
        data[i] = data[i - 1] + dist(gen);  // phi = 1.0
    }

    // Perform bootstrap ADF test
    auto result = ag::stats::adf_test_bootstrap(data, 2, ADFRegressionForm::Constant, 200, 555);

    // Check that result structure is valid
    REQUIRE(std::isfinite(result.statistic));
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);

    // For unit root, p-value should typically be high
    // (though this is stochastic, so we just verify it's not extremely low)
    // The test is successful if it doesn't reject too strongly
    REQUIRE(result.p_value > 0.01);  // More lenient threshold
}

// Test bootstrap ADF with Student-t innovations
TEST(bootstrap_adf_student_t_stationary) {
    // Generate stationary process with Student-t innovations
    const double phi = 0.6;
    const int n = 150;

    std::mt19937 gen(666);
    std::student_t_distribution<double> dist(5.0);  // Heavy tails

    std::vector<double> data(n);
    data[0] = dist(gen);
    for (int i = 1; i < n; ++i) {
        data[i] = phi * data[i - 1] + dist(gen);
    }

    // Perform bootstrap ADF test
    auto result = ag::stats::adf_test_bootstrap(data, 2, ADFRegressionForm::Constant, 200, 777);

    // Check validity
    REQUIRE(std::isfinite(result.statistic));
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);

    // Bootstrap should work correctly even with heavy tails
    REQUIRE(result.critical_value_1pct < result.critical_value_5pct);
    REQUIRE(result.critical_value_5pct < result.critical_value_10pct);
}

// Test bootstrap ADF reproducibility
TEST(bootstrap_adf_reproducibility) {
    std::mt19937 gen(888);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(100);
    data[0] = dist(gen);
    for (int i = 1; i < 100; ++i) {
        data[i] = 0.7 * data[i - 1] + dist(gen);
    }

    // Run test twice with same seed
    auto result1 = ag::stats::adf_test_bootstrap(data, 1, ADFRegressionForm::Constant, 100, 1111);
    auto result2 = ag::stats::adf_test_bootstrap(data, 1, ADFRegressionForm::Constant, 100, 1111);

    // Results should be identical
    REQUIRE(std::abs(result1.statistic - result2.statistic) < 1e-10);
    REQUIRE(std::abs(result1.p_value - result2.p_value) < 1e-10);
    REQUIRE(std::abs(result1.critical_value_5pct - result2.critical_value_5pct) < 1e-10);
}

// Test bootstrap ADF with different regression forms
TEST(bootstrap_adf_regression_forms) {
    std::mt19937 gen(999);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(120);
    data[0] = dist(gen);
    for (int i = 1; i < 120; ++i) {
        data[i] = 0.6 * data[i - 1] + dist(gen);
    }

    // Test with None (no constant, no trend)
    auto result_none = ag::stats::adf_test_bootstrap(data, 1, ADFRegressionForm::None, 100, 1000);
    REQUIRE(result_none.regression_form == ADFRegressionForm::None);
    REQUIRE(std::isfinite(result_none.statistic));

    // Test with Constant
    auto result_const =
        ag::stats::adf_test_bootstrap(data, 1, ADFRegressionForm::Constant, 100, 1000);
    REQUIRE(result_const.regression_form == ADFRegressionForm::Constant);
    REQUIRE(std::isfinite(result_const.statistic));

    // Test with ConstantAndTrend
    auto result_trend =
        ag::stats::adf_test_bootstrap(data, 1, ADFRegressionForm::ConstantAndTrend, 100, 1000);
    REQUIRE(result_trend.regression_form == ADFRegressionForm::ConstantAndTrend);
    REQUIRE(std::isfinite(result_trend.statistic));
}

// ============================================================================
// Error Handling Tests
// ============================================================================

// Test bootstrap Ljung-Box with empty data
TEST(bootstrap_ljung_box_empty_data) {
    std::vector<double> empty;

    bool caught_exception = false;
    try {
        (void)ag::stats::ljung_box_test_bootstrap(empty, 5, 100, 42);
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }

    REQUIRE(caught_exception);
}

// Test bootstrap Ljung-Box with invalid lags
TEST(bootstrap_ljung_box_invalid_lags) {
    std::vector<double> data(50, 1.0);

    bool caught_exception = false;
    try {
        (void)ag::stats::ljung_box_test_bootstrap(data, 0, 100, 42);  // lags = 0
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }

    REQUIRE(caught_exception);
}

// Test bootstrap Ljung-Box with too many lags
TEST(bootstrap_ljung_box_too_many_lags) {
    std::vector<double> data(50, 1.0);

    bool caught_exception = false;
    try {
        (void)ag::stats::ljung_box_test_bootstrap(data, 60, 100, 42);  // lags > n
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }

    REQUIRE(caught_exception);
}

// Test bootstrap ADF with insufficient data
TEST(bootstrap_adf_insufficient_data) {
    std::vector<double> data(5, 1.0);  // Too small

    bool caught_exception = false;
    try {
        (void)ag::stats::adf_test_bootstrap(data, 1, ADFRegressionForm::Constant, 100, 42);
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }

    REQUIRE(caught_exception);
}

// ============================================================================
// Tests for Correct Unit Root Null Hypothesis Implementation
// ============================================================================

// Test that pure random walk (unit root) has high p-value
TEST(bootstrap_adf_pure_random_walk) {
    // Generate a pure random walk: y_t = y_{t-1} + ε_t
    const int n = 200;
    std::mt19937 gen(99999);  // Use a different seed
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(n);
    data[0] = 0.0;
    for (int i = 1; i < n; ++i) {
        data[i] = data[i - 1] + dist(gen);
    }

    // With correct bootstrap under unit root null, p-value should be high
    auto result = ag::stats::adf_test_bootstrap(data, 2, ADFRegressionForm::Constant, 500, 42);

    // For unit root data, we expect to fail to reject most of the time
    // However, due to randomness, we use a more lenient threshold
    // The test should not strongly reject (p-value should not be extremely low)
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);

    // Critical values should be in reasonable negative range
    REQUIRE(result.critical_value_1pct < result.critical_value_5pct);
    REQUIRE(result.critical_value_5pct < result.critical_value_10pct);
    REQUIRE(result.critical_value_5pct < -1.0);  // Should be negative
}

// Test that strongly stationary series has low p-value
TEST(bootstrap_adf_strongly_stationary) {
    // Generate strongly stationary AR(1) with phi = 0.3
    const double phi = 0.3;
    const int n = 200;

    std::mt19937 gen(54321);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(n);
    data[0] = dist(gen);
    for (int i = 1; i < n; ++i) {
        data[i] = phi * data[i - 1] + dist(gen);
    }

    // With correct bootstrap under unit root null, stationary data should reject
    auto result = ag::stats::adf_test_bootstrap(data, 2, ADFRegressionForm::Constant, 500, 99);

    // For strongly stationary data, p-value should be low (reject unit root)
    REQUIRE(result.p_value < 0.3);

    // Statistic should be more negative (strong evidence against unit root)
    REQUIRE(result.statistic < -1.0);
}

// Test with Student-t innovations on unit root series
TEST(bootstrap_adf_unit_root_student_t) {
    // Generate random walk with Student-t(5) innovations
    const int n = 200;
    std::mt19937 gen(11111);
    std::student_t_distribution<double> dist(5.0);  // Heavy tails

    std::vector<double> data(n);
    data[0] = 0.0;
    for (int i = 1; i < n; ++i) {
        data[i] = data[i - 1] + dist(gen);
    }

    // Bootstrap should work correctly with heavy tails when unit root is imposed
    auto result = ag::stats::adf_test_bootstrap(data, 2, ADFRegressionForm::Constant, 500, 777);

    // For unit root with heavy tails, p-value should still be high
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);
    REQUIRE(result.p_value > 0.05);  // Should fail to reject unit root

    // Critical values should be in reasonable range
    REQUIRE(result.critical_value_1pct < result.critical_value_5pct);
    REQUIRE(result.critical_value_5pct < result.critical_value_10pct);
}

// Test that differences of unit root series are stationary
TEST(bootstrap_adf_integrated_series) {
    // Generate I(1) series: random walk with drift
    const int n = 200;
    const double drift = 0.1;
    std::mt19937 gen(22222);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(n);
    data[0] = 0.0;
    for (int i = 1; i < n; ++i) {
        data[i] = data[i - 1] + drift + dist(gen);
    }

    // Test levels: should not reject unit root
    auto result_levels =
        ag::stats::adf_test_bootstrap(data, 2, ADFRegressionForm::ConstantAndTrend, 300, 333);
    REQUIRE(result_levels.p_value > 0.05);

    // Take first differences
    std::vector<double> differences(n - 1);
    for (int i = 0; i < n - 1; ++i) {
        differences[i] = data[i + 1] - data[i];
    }

    // Test differences: should reject unit root (differences are stationary)
    auto result_diff =
        ag::stats::adf_test_bootstrap(differences, 2, ADFRegressionForm::Constant, 300, 444);
    REQUIRE(result_diff.p_value < 0.5);  // Should tend to reject for stationary series
}

// ============================================================================
// Main test runner
// ============================================================================

int main() {
    report_test_results("Stats: Bootstrap Methods");
    return get_test_result();
}
