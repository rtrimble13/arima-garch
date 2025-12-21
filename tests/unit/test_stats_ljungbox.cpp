#include "ag/stats/LjungBox.hpp"

#include <cmath>
#include <random>
#include <vector>

#include "test_framework.hpp"

// Test Ljung-Box statistic calculation for white noise
TEST(ljung_box_white_noise) {
    // Generate white noise with fixed seed for reproducibility
    std::mt19937 gen(12345);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> residuals(500);
    for (auto& val : residuals) {
        val = dist(gen);
    }

    // Compute Ljung-Box statistic for 10 lags
    double q = ag::stats::ljung_box_statistic(residuals, 10);

    // For white noise, Q should be approximately chi-square(10)
    // Mean of chi-square(k) is k, std dev is sqrt(2k)
    // So for k=10, mean=10, std=sqrt(20)≈4.47
    // Q should roughly be in range [0, 25] with high probability
    REQUIRE(q >= 0.0);
    REQUIRE(q < 30.0);  // Very lenient upper bound
}

// Test Ljung-Box test with white noise - should have high p-value
TEST(ljung_box_test_white_noise_high_pvalue) {
    // Generate white noise with fixed seed
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> residuals(1000);
    for (auto& val : residuals) {
        val = dist(gen);
    }

    // Perform Ljung-Box test
    auto result = ag::stats::ljung_box_test(residuals, 10);

    // Check that result structure is populated correctly
    REQUIRE(result.statistic >= 0.0);
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);
    REQUIRE(result.lags == 10);
    REQUIRE(result.dof == 10);

    // For white noise, p-value should be high (not significant)
    // With probability ~0.95, p-value should be > 0.05
    // We use a more lenient threshold for robustness
    REQUIRE(result.p_value > 0.01);
}

// Test Ljung-Box test with autocorrelated data - should have low p-value
TEST(ljung_box_test_autocorrelated_low_pvalue) {
    // Generate AR(1) process with strong autocorrelation
    const double phi = 0.9;
    const int n = 500;

    std::mt19937 gen(123);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> residuals(n);
    residuals[0] = dist(gen);
    for (int i = 1; i < n; ++i) {
        residuals[i] = phi * residuals[i - 1] + dist(gen);
    }

    // Perform Ljung-Box test
    auto result = ag::stats::ljung_box_test(residuals, 10);

    // Check that result structure is populated
    REQUIRE(result.statistic >= 0.0);
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);
    REQUIRE(result.lags == 10);
    REQUIRE(result.dof == 10);

    // For strongly autocorrelated data, p-value should be very low (significant)
    REQUIRE(result.p_value < 0.05);
}

// Test Ljung-Box with custom degrees of freedom
TEST(ljung_box_test_custom_dof) {
    std::mt19937 gen(999);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> residuals(200);
    for (auto& val : residuals) {
        val = dist(gen);
    }

    // Test with custom DOF (e.g., adjusted for parameter estimation)
    std::size_t lags = 10;
    std::size_t dof = 8;  // Reduced DOF due to 2 estimated parameters

    auto result = ag::stats::ljung_box_test(residuals, lags, dof);

    REQUIRE(result.lags == lags);
    REQUIRE(result.dof == dof);
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);
}

// Test Ljung-Box statistic increases with more lags
TEST(ljung_box_statistic_increases_with_lags) {
    // Create data with slight autocorrelation
    const double phi = 0.5;
    const int n = 300;

    std::mt19937 gen(555);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> residuals(n);
    residuals[0] = dist(gen);
    for (int i = 1; i < n; ++i) {
        residuals[i] = phi * residuals[i - 1] + dist(gen);
    }

    // Compute statistic for different lags
    double q5 = ag::stats::ljung_box_statistic(residuals, 5);
    double q10 = ag::stats::ljung_box_statistic(residuals, 10);

    // With autocorrelated data, Q should generally increase with more lags
    // (though this isn't guaranteed for every random seed)
    REQUIRE(q10 > q5);
}

// Test Ljung-Box with constant series (zero variance)
TEST(ljung_box_constant_series) {
    std::vector<double> residuals(100, 5.0);  // All values are 5.0

    // ACF will be 0 for all non-zero lags, so Q should be 0
    double q = ag::stats::ljung_box_statistic(residuals, 10);
    REQUIRE_APPROX(q, 0.0, 1e-10);

    // P-value should be 1.0 (no evidence of autocorrelation)
    auto result = ag::stats::ljung_box_test(residuals, 10);
    REQUIRE_APPROX(result.p_value, 1.0, 1e-6);
}

// Test Ljung-Box with very small positive Q statistic
TEST(ljung_box_small_q_statistic) {
    // Generate very good white noise that should give small Q
    std::mt19937 gen(777);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> residuals(500);
    for (auto& val : residuals) {
        val = dist(gen);
    }

    auto result = ag::stats::ljung_box_test(residuals, 5);

    // Small Q should give high p-value
    if (result.statistic < 5.0) {  // If Q is less than the degrees of freedom
        REQUIRE(result.p_value > 0.2);
    }
}

// Test error handling - empty residuals
TEST(ljung_box_empty_residuals) {
    std::vector<double> residuals = {};
    bool caught = false;
    try {
        [[maybe_unused]] auto q = ag::stats::ljung_box_statistic(residuals, 1);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test error handling - zero lags
TEST(ljung_box_zero_lags) {
    std::vector<double> residuals = {1.0, 2.0, 3.0, 4.0, 5.0};
    bool caught = false;
    try {
        [[maybe_unused]] auto q = ag::stats::ljung_box_statistic(residuals, 0);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test error handling - lags >= sample size
TEST(ljung_box_lags_too_large) {
    std::vector<double> residuals = {1.0, 2.0, 3.0};
    bool caught = false;
    try {
        [[maybe_unused]] auto q = ag::stats::ljung_box_statistic(residuals, 3);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test error handling - zero degrees of freedom in test
TEST(ljung_box_test_zero_custom_dof) {
    std::vector<double> residuals = {1.0, 2.0, 3.0, 4.0, 5.0};
    bool caught = false;
    try {
        // Explicitly pass 0 as custom dof (not default)
        [[maybe_unused]] auto result = ag::stats::ljung_box_test(residuals, 1, 0);
        // Note: This should use default dof=lags=1, so it should NOT throw
        caught = false;
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    // With default behavior (dof=0 means use lags), this should not throw
    REQUIRE(!caught);
}

// Test with minimum valid input
TEST(ljung_box_minimum_valid_input) {
    std::vector<double> residuals = {1.0, 2.0, 3.0, 4.0};
    auto result = ag::stats::ljung_box_test(residuals, 1);

    REQUIRE(result.statistic >= 0.0);
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);
    REQUIRE(result.lags == 1);
    REQUIRE(result.dof == 1);
}

// Test with different random seeds for robustness
TEST(ljung_box_multiple_white_noise_samples) {
    const std::vector<unsigned int> seeds = {1, 10, 100, 1000, 10000};
    int high_pvalue_count = 0;

    for (unsigned int seed : seeds) {
        std::mt19937 gen(seed);
        std::normal_distribution<double> dist(0.0, 1.0);

        std::vector<double> residuals(500);
        for (auto& val : residuals) {
            val = dist(gen);
        }

        auto result = ag::stats::ljung_box_test(residuals, 10);

        // Count how many have p-value > 0.05
        if (result.p_value > 0.05) {
            high_pvalue_count++;
        }
    }

    // At least 3 out of 5 white noise samples should have high p-values
    // (statistically, we expect ~95% to be > 0.05, so 3/5 is very conservative)
    REQUIRE(high_pvalue_count >= 3);
}

// Test p-value bounds are reasonable
TEST(ljung_box_pvalue_bounds) {
    std::mt19937 gen(2024);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> residuals(300);
    for (auto& val : residuals) {
        val = dist(gen);
    }

    auto result = ag::stats::ljung_box_test(residuals, 5);

    // P-value must be in [0, 1]
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);

    // For reasonable data, p-value should not be exactly 0 or 1
    // (unless something is very wrong)
    REQUIRE(result.p_value > 0.0);
    REQUIRE(result.p_value < 1.0);
}

int main() {
    report_test_results("Ljung-Box Tests");
    return get_test_result();
}
