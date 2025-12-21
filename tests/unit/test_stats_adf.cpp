#include "ag/stats/ADF.hpp"

#include <cmath>
#include <random>
#include <vector>

#include "test_framework.hpp"

// Test ADF on stationary white noise - should reject unit root
TEST(adf_test_white_noise_stationary) {
    // Generate white noise with fixed seed for reproducibility
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(200);
    for (auto& val : data) {
        val = dist(gen);
    }

    // Perform ADF test with constant
    auto result = ag::stats::adf_test(data, 1, ag::stats::ADFRegressionForm::Constant);

    // Check that result structure is populated
    REQUIRE(result.lags >= 0);
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);
    REQUIRE(result.regression_form == ag::stats::ADFRegressionForm::Constant);

    // White noise should be stationary (reject null hypothesis of unit root)
    // Low p-value indicates stationarity
    REQUIRE(result.p_value < 0.1);  // Should strongly reject unit root

    // Test statistic should be more negative than critical values
    REQUIRE(result.statistic < result.critical_value_5pct);
}

// Test ADF on random walk - should fail to reject unit root
TEST(adf_test_random_walk_nonstationary) {
    // Generate random walk: y_t = y_{t-1} + ε_t
    std::mt19937 gen(123);
    std::normal_distribution<double> dist(0.0, 1.0);

    const int n = 200;
    std::vector<double> data(n);
    data[0] = dist(gen);
    for (int i = 1; i < n; ++i) {
        data[i] = data[i - 1] + dist(gen);
    }

    // Perform ADF test
    auto result = ag::stats::adf_test(data, 1, ag::stats::ADFRegressionForm::Constant);

    // Check result structure
    REQUIRE(result.lags >= 0);
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);

    // Random walk should be non-stationary (fail to reject unit root)
    // High p-value indicates unit root
    REQUIRE(result.p_value > 0.1);  // Should fail to reject unit root

    // Test statistic should be closer to zero (less negative)
    REQUIRE(result.statistic > result.critical_value_10pct);
}

// Test ADF on stationary AR(1) process
TEST(adf_test_stationary_ar1) {
    // Generate stationary AR(1): y_t = 0.5*y_{t-1} + ε_t
    const double phi = 0.5;  // |phi| < 1 ensures stationarity
    const int n = 250;

    std::mt19937 gen(456);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(n);
    data[0] = dist(gen);
    for (int i = 1; i < n; ++i) {
        data[i] = phi * data[i - 1] + dist(gen);
    }

    // Test with automatic lag selection
    auto result = ag::stats::adf_test(data, 0, ag::stats::ADFRegressionForm::Constant);

    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);

    // Stationary AR(1) should reject unit root
    REQUIRE(result.p_value < 0.15);
}

// Test ADF with trend
TEST(adf_test_with_trend) {
    // Generate series with deterministic trend
    const int n = 200;
    std::mt19937 gen(789);
    std::normal_distribution<double> dist(0.0, 0.5);

    std::vector<double> data(n);
    for (int i = 0; i < n; ++i) {
        data[i] = 0.1 * i + dist(gen);  // Linear trend + noise
    }

    // Test with constant and trend
    auto result = ag::stats::adf_test(data, 1, ag::stats::ADFRegressionForm::ConstantAndTrend);

    REQUIRE(result.regression_form == ag::stats::ADFRegressionForm::ConstantAndTrend);
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);

    // Trend-stationary series should reject unit root when trend is included
    REQUIRE(result.p_value < 0.2);
}

// Test ADF with no constant or trend
TEST(adf_test_no_deterministics) {
    std::mt19937 gen(321);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(150);
    for (auto& val : data) {
        val = dist(gen);
    }

    auto result = ag::stats::adf_test(data, 1, ag::stats::ADFRegressionForm::None);

    REQUIRE(result.regression_form == ag::stats::ADFRegressionForm::None);
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);

    // Should still detect stationarity
    REQUIRE(result.statistic < 0.0);  // Should be negative
}

// Test automatic lag selection
TEST(adf_test_auto_lag_selection) {
    std::mt19937 gen(555);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(300);
    for (auto& val : data) {
        val = dist(gen);
    }

    // Test with lags=0 (automatic selection)
    auto result = ag::stats::adf_test(data, 0, ag::stats::ADFRegressionForm::Constant);

    // Should have selected some number of lags
    REQUIRE(result.lags >= 0);
    REQUIRE(result.lags < 50);  // Reasonable upper bound

    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);
}

// Test automatic regression form selection
TEST(adf_test_auto_regression_form) {
    std::mt19937 gen(666);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(200);
    for (auto& val : data) {
        val = dist(gen);
    }

    auto result = ag::stats::adf_test_auto(data, 1);

    // Should select one of the three forms
    bool valid_form = (result.regression_form == ag::stats::ADFRegressionForm::None) ||
                      (result.regression_form == ag::stats::ADFRegressionForm::Constant) ||
                      (result.regression_form == ag::stats::ADFRegressionForm::ConstantAndTrend);
    REQUIRE(valid_form);

    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);
}

// Test critical values are ordered correctly
TEST(adf_test_critical_values_ordered) {
    std::mt19937 gen(777);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(150);
    for (auto& val : data) {
        val = dist(gen);
    }

    auto result = ag::stats::adf_test(data, 1, ag::stats::ADFRegressionForm::Constant);

    // Critical values should be ordered: 1% < 5% < 10% (all negative)
    REQUIRE(result.critical_value_1pct < result.critical_value_5pct);
    REQUIRE(result.critical_value_5pct < result.critical_value_10pct);
    REQUIRE(result.critical_value_10pct < 0.0);
}

// Test different regression forms give different critical values
TEST(adf_test_different_forms_different_cvs) {
    std::mt19937 gen(888);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(150);
    for (auto& val : data) {
        val = dist(gen);
    }

    auto result_none = ag::stats::adf_test(data, 1, ag::stats::ADFRegressionForm::None);
    auto result_const = ag::stats::adf_test(data, 1, ag::stats::ADFRegressionForm::Constant);
    auto result_trend =
        ag::stats::adf_test(data, 1, ag::stats::ADFRegressionForm::ConstantAndTrend);

    // Critical values should differ across forms
    // More restrictive forms (with fewer parameters) have less negative CVs
    REQUIRE(result_none.critical_value_5pct > result_const.critical_value_5pct);
    REQUIRE(result_const.critical_value_5pct > result_trend.critical_value_5pct);
}

// Test ADF handles nearly constant series
TEST(adf_test_nearly_constant) {
    std::vector<double> data(100);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = 10.0 + (i % 2) * 0.001;  // Nearly constant with tiny variation
    }

    auto result = ag::stats::adf_test(data, 1, ag::stats::ADFRegressionForm::Constant);

    // Should complete without error
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);
}

// Test ADF with multiple lags
TEST(adf_test_multiple_lags) {
    // Generate AR(2) process
    const double phi1 = 0.6;
    const double phi2 = 0.3;
    const int n = 300;

    std::mt19937 gen(999);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(n);
    data[0] = dist(gen);
    data[1] = dist(gen);
    for (int i = 2; i < n; ++i) {
        data[i] = phi1 * data[i - 1] + phi2 * data[i - 2] + dist(gen);
    }

    // Test with 2 lags to capture AR(2) structure
    auto result = ag::stats::adf_test(data, 2, ag::stats::ADFRegressionForm::Constant);

    REQUIRE(result.lags == 2);
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);

    // Should detect stationarity
    REQUIRE(result.p_value < 0.2);
}

// Test error handling - too few observations
TEST(adf_test_too_few_observations) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};

    bool caught = false;
    try {
        [[maybe_unused]] auto result =
            ag::stats::adf_test(data, 1, ag::stats::ADFRegressionForm::Constant);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test error handling - too many lags
TEST(adf_test_too_many_lags) {
    std::vector<double> data(50, 1.0);

    bool caught = false;
    try {
        [[maybe_unused]] auto result =
            ag::stats::adf_test(data, 30, ag::stats::ADFRegressionForm::Constant);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test ADF returns consistent results
TEST(adf_test_consistency) {
    std::vector<double> data = {1.2, 1.5, 1.3, 1.8, 1.6, 2.0, 1.9, 2.2, 2.1, 2.5,
                                2.3, 2.7, 2.6, 2.9, 2.8, 3.1, 3.0, 3.3, 3.2, 3.5,
                                3.4, 3.7, 3.6, 3.9, 3.8, 4.0, 4.1, 4.2, 4.3, 4.4};

    // Run test twice - should get same result
    auto result1 = ag::stats::adf_test(data, 1, ag::stats::ADFRegressionForm::Constant);
    auto result2 = ag::stats::adf_test(data, 1, ag::stats::ADFRegressionForm::Constant);

    REQUIRE_APPROX(result1.statistic, result2.statistic, 1e-10);
    REQUIRE_APPROX(result1.p_value, result2.p_value, 1e-10);
    REQUIRE(result1.lags == result2.lags);
}

// Test that statistic is reasonable
TEST(adf_test_statistic_reasonable_range) {
    std::mt19937 gen(1111);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(200);
    for (auto& val : data) {
        val = dist(gen);
    }

    auto result = ag::stats::adf_test(data, 1, ag::stats::ADFRegressionForm::Constant);

    // Test statistic should be in reasonable range (typically -10 to 5)
    REQUIRE(result.statistic > -20.0);
    REQUIRE(result.statistic < 10.0);
}

// Test with larger sample size
TEST(adf_test_large_sample) {
    std::mt19937 gen(2222);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(1000);
    for (auto& val : data) {
        val = dist(gen);
    }

    auto result = ag::stats::adf_test(data, 0, ag::stats::ADFRegressionForm::Constant);

    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);

    // With large sample, white noise should strongly reject unit root
    REQUIRE(result.p_value < 0.05);
}

// Test auto selection on non-stationary series
TEST(adf_test_auto_on_random_walk) {
    std::mt19937 gen(3333);
    std::normal_distribution<double> dist(0.0, 1.0);

    const int n = 200;
    std::vector<double> data(n);
    data[0] = 0.0;
    for (int i = 1; i < n; ++i) {
        data[i] = data[i - 1] + dist(gen);
    }

    auto result = ag::stats::adf_test_auto(data, 0);

    // Should fail to reject regardless of form selected
    REQUIRE(result.p_value > 0.05);
}

int main() {
    report_test_results("ADF Tests");
    return get_test_result();
}
