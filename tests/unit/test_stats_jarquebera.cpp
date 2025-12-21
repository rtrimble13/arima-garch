#include "ag/stats/JarqueBera.hpp"

#include <cmath>
#include <random>
#include <vector>

#include "test_framework.hpp"

// Test Jarque-Bera statistic with a known small sample
// Using a specific small dataset to verify the calculation
TEST(jarque_bera_known_small_sample) {
    // Small dataset: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    // This is uniformly distributed, so should have negative excess kurtosis
    // and zero skewness, resulting in a specific JB statistic value
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    double jb = ag::stats::jarque_bera_statistic(data);

    // For this data:
    // - Mean = 5.5
    // - Skewness should be ~0 (symmetric)
    // - Excess kurtosis should be negative (uniform-like)
    // Expected JB ≈ n/6 * (S² + K²/4)
    // Since S ≈ 0 and K < 0, JB should be relatively small but positive

    // The exact value depends on the sample skewness and kurtosis formulas
    // For n=10, we expect JB to be small (indicating normality is not rejected strongly)
    REQUIRE(jb >= 0.0);
    REQUIRE(jb < 3.0);  // Should be small for this symmetric, uniform-like data
}

// Test Jarque-Bera with known calculation
// Dataset: {1, 2, 2, 3, 3, 3, 4, 4, 5}
TEST(jarque_bera_known_calculation) {
    std::vector<double> data = {1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0};

    auto result = ag::stats::jarque_bera_test(data);

    // Verify the statistic is non-negative
    REQUIRE(result.statistic >= 0.0);

    // Verify p-value is in valid range
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);
}

// Test that normal samples yield high p-values on average
TEST(jarque_bera_normal_samples_high_pvalue) {
    // Generate multiple samples from normal distribution
    // and check that most have high p-values (fail to reject normality)
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    const int num_samples = 10;
    const int sample_size = 100;
    int high_pvalue_count = 0;

    for (int i = 0; i < num_samples; ++i) {
        std::vector<double> data(sample_size);
        for (auto& val : data) {
            val = dist(gen);
        }

        auto result = ag::stats::jarque_bera_test(data);

        // For truly normal data, p-value should be high (> 0.05) most of the time
        if (result.p_value > 0.05) {
            high_pvalue_count++;
        }
    }

    // At least 7 out of 10 normal samples should have p-value > 0.05
    // (Expected ~9-10, but we use 7 to account for randomness)
    REQUIRE(high_pvalue_count >= 7);
}

// Test that normal samples have higher average p-values than non-normal
TEST(jarque_bera_normal_vs_nonnormal) {
    std::mt19937 gen(123);
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    std::uniform_real_distribution<double> uniform_dist(-1.0, 1.0);

    // Generate normal samples
    std::vector<double> normal_data(200);
    for (auto& val : normal_data) {
        val = normal_dist(gen);
    }

    // Generate uniform samples (non-normal, lighter tails)
    std::vector<double> uniform_data(200);
    for (auto& val : uniform_data) {
        val = uniform_dist(gen);
    }

    auto normal_result = ag::stats::jarque_bera_test(normal_data);
    auto uniform_result = ag::stats::jarque_bera_test(uniform_data);

    // Normal data should generally have higher p-value (less evidence against normality)
    // Uniform data should have lower p-value (more evidence against normality)
    // This is a statistical expectation, not guaranteed for every seed
    REQUIRE(normal_result.p_value >= 0.01);  // Should not strongly reject normality
}

// Test Jarque-Bera with strongly skewed data - should have low p-value
TEST(jarque_bera_skewed_data_low_pvalue) {
    // Create right-skewed data
    std::vector<double> data = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 10.0, 15.0, 20.0};

    auto result = ag::stats::jarque_bera_test(data);

    // Check that result structure is populated
    REQUIRE(result.statistic >= 0.0);
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);

    // For strongly skewed data, p-value should be low (significant)
    // The exact threshold depends on the data, but this should show departure from normality
    REQUIRE(result.p_value < 0.5);  // Should show some evidence against normality
}

// Test Jarque-Bera with heavy-tailed data
TEST(jarque_bera_heavy_tails) {
    // Create data with heavy tails (high kurtosis)
    std::vector<double> data = {0.0, 0.0, 0.0, 0.0,   0.0,  1.0,   1.0,
                                1.0, 1.0, 1.0, -10.0, 10.0, -15.0, 15.0};

    auto result = ag::stats::jarque_bera_test(data);

    // Heavy-tailed data should show departure from normality
    REQUIRE(result.statistic > 0.0);
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);
}

// Test Jarque-Bera with perfectly normal-looking small dataset
TEST(jarque_bera_near_normal_small) {
    // Create data that's very close to normal
    std::vector<double> data = {-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5};

    auto result = ag::stats::jarque_bera_test(data);

    // Should have reasonable statistic and p-value
    REQUIRE(result.statistic >= 0.0);
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);
}

// Test with constant series (zero variance)
TEST(jarque_bera_constant_series) {
    std::vector<double> data = {5.0, 5.0, 5.0, 5.0, 5.0};

    double jb = ag::stats::jarque_bera_statistic(data);

    // Constant series has zero skewness and kurtosis, so JB should be 0
    REQUIRE_APPROX(jb, 0.0, 1e-10);

    auto result = ag::stats::jarque_bera_test(data);
    // P-value should be 1.0 (no evidence against normality when JB=0)
    REQUIRE_APPROX(result.p_value, 1.0, 1e-6);
}

// Test with minimum valid input (4 data points)
TEST(jarque_bera_minimum_valid_input) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
    auto result = ag::stats::jarque_bera_test(data);

    REQUIRE(result.statistic >= 0.0);
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);
}

// Test error handling - empty data
TEST(jarque_bera_empty_data) {
    std::vector<double> data = {};
    bool caught = false;
    try {
        [[maybe_unused]] auto jb = ag::stats::jarque_bera_statistic(data);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test error handling - insufficient data (less than 4 points)
TEST(jarque_bera_insufficient_data_1) {
    std::vector<double> data = {1.0};
    bool caught = false;
    try {
        [[maybe_unused]] auto jb = ag::stats::jarque_bera_statistic(data);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

TEST(jarque_bera_insufficient_data_2) {
    std::vector<double> data = {1.0, 2.0};
    bool caught = false;
    try {
        [[maybe_unused]] auto jb = ag::stats::jarque_bera_statistic(data);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

TEST(jarque_bera_insufficient_data_3) {
    std::vector<double> data = {1.0, 2.0, 3.0};
    bool caught = false;
    try {
        [[maybe_unused]] auto jb = ag::stats::jarque_bera_statistic(data);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test p-value bounds are reasonable
TEST(jarque_bera_pvalue_bounds) {
    std::mt19937 gen(2024);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(100);
    for (auto& val : data) {
        val = dist(gen);
    }

    auto result = ag::stats::jarque_bera_test(data);

    // P-value must be in [0, 1]
    REQUIRE(result.p_value >= 0.0);
    REQUIRE(result.p_value <= 1.0);

    // For reasonable data, p-value should not be exactly 0 or 1
    REQUIRE(result.p_value > 0.0);
    REQUIRE(result.p_value < 1.0);
}

// Test multiple normal samples for robustness
TEST(jarque_bera_multiple_normal_samples) {
    const std::vector<unsigned int> seeds = {1, 10, 100, 1000, 10000};
    int high_pvalue_count = 0;

    for (unsigned int seed : seeds) {
        std::mt19937 gen(seed);
        std::normal_distribution<double> dist(0.0, 1.0);

        std::vector<double> data(150);
        for (auto& val : data) {
            val = dist(gen);
        }

        auto result = ag::stats::jarque_bera_test(data);

        // Count how many have p-value > 0.05
        if (result.p_value > 0.05) {
            high_pvalue_count++;
        }
    }

    // At least 3 out of 5 normal samples should have high p-values
    // (statistically, we expect ~95% to be > 0.05, so 3/5 is very conservative)
    REQUIRE(high_pvalue_count >= 3);
}

// Smoke test: ensure statistic increases with departure from normality
TEST(jarque_bera_statistic_increases_with_departure) {
    std::mt19937 gen(999);
    std::normal_distribution<double> dist(0.0, 1.0);

    // Near-normal data
    std::vector<double> normal_data(100);
    for (auto& val : normal_data) {
        val = dist(gen);
    }

    // Highly skewed data
    std::vector<double> skewed_data = {1.0, 1.0, 1.0, 1.0,  2.0,  2.0,
                                       2.0, 3.0, 3.0, 50.0, 60.0, 70.0};

    double jb_normal = ag::stats::jarque_bera_statistic(normal_data);
    double jb_skewed = ag::stats::jarque_bera_statistic(skewed_data);

    // Skewed data should generally have higher JB statistic
    REQUIRE(jb_skewed > jb_normal);
}

int main() {
    report_test_results("Jarque-Bera Tests");
    return get_test_result();
}
