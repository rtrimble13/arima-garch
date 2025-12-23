#include "ag/selection/InformationCriteria.hpp"

#include <cmath>
#include <limits>

#include "test_framework.hpp"

using ag::selection::computeAIC;
using ag::selection::computeAICc;
using ag::selection::computeBIC;

// ============================================================================
// AIC Tests
// ============================================================================

// Test AIC with known values
TEST(aic_known_values) {
    // AIC = 2k - 2*loglik
    // Example 1: k=3, loglik=-100
    // AIC = 2*3 - 2*(-100) = 6 + 200 = 206
    double aic1 = computeAIC(-100.0, 3);
    REQUIRE_APPROX(aic1, 206.0, 1e-10);

    // Example 2: k=5, loglik=-250.5
    // AIC = 2*5 - 2*(-250.5) = 10 + 501 = 511
    double aic2 = computeAIC(-250.5, 5);
    REQUIRE_APPROX(aic2, 511.0, 1e-10);

    // Example 3: k=1, loglik=-50.25
    // AIC = 2*1 - 2*(-50.25) = 2 + 100.5 = 102.5
    double aic3 = computeAIC(-50.25, 1);
    REQUIRE_APPROX(aic3, 102.5, 1e-10);
}

// Test AIC with zero parameters
TEST(aic_zero_parameters) {
    // AIC = 2*0 - 2*loglik = -2*loglik
    double aic = computeAIC(-100.0, 0);
    REQUIRE_APPROX(aic, 200.0, 1e-10);
}

// Test AIC with positive log-likelihood
TEST(aic_positive_loglik) {
    // AIC = 2k - 2*loglik
    // k=2, loglik=10
    // AIC = 2*2 - 2*10 = 4 - 20 = -16
    double aic = computeAIC(10.0, 2);
    REQUIRE_APPROX(aic, -16.0, 1e-10);
}

// Test AIC formula consistency
TEST(aic_formula_consistency) {
    // Verify the formula: AIC = 2k - 2*loglik
    double loglik = -123.456;
    int k = 7;
    double aic = computeAIC(loglik, k);
    double expected = 2.0 * k - 2.0 * loglik;
    REQUIRE_APPROX(aic, expected, 1e-10);
}

// ============================================================================
// BIC Tests
// ============================================================================

// Test BIC with known values
TEST(bic_known_values) {
    // BIC = k*log(n) - 2*loglik
    // Example 1: k=3, n=100, loglik=-100
    // BIC = 3*log(100) - 2*(-100) = 3*4.605... + 200 ≈ 13.816 + 200 = 213.816
    double bic1 = computeBIC(-100.0, 3, 100);
    double expected1 = 3.0 * std::log(100.0) - 2.0 * (-100.0);
    REQUIRE_APPROX(bic1, expected1, 1e-10);

    // Example 2: k=5, n=200, loglik=-250.5
    // BIC = 5*log(200) - 2*(-250.5) = 5*5.298... + 501
    double bic2 = computeBIC(-250.5, 5, 200);
    double expected2 = 5.0 * std::log(200.0) - 2.0 * (-250.5);
    REQUIRE_APPROX(bic2, expected2, 1e-10);

    // Example 3: k=1, n=50, loglik=-50.25
    // BIC = 1*log(50) - 2*(-50.25) = log(50) + 100.5
    double bic3 = computeBIC(-50.25, 1, 50);
    double expected3 = std::log(50.0) - 2.0 * (-50.25);
    REQUIRE_APPROX(bic3, expected3, 1e-10);
}

// Test BIC with zero parameters
TEST(bic_zero_parameters) {
    // BIC = 0*log(n) - 2*loglik = -2*loglik
    double bic = computeBIC(-100.0, 0, 100);
    REQUIRE_APPROX(bic, 200.0, 1e-10);
}

// Test BIC with sample size 1
TEST(bic_sample_size_one) {
    // BIC = k*log(1) - 2*loglik = k*0 - 2*loglik = -2*loglik
    double bic = computeBIC(-100.0, 3, 1);
    REQUIRE_APPROX(bic, 200.0, 1e-10);
}

// Test BIC formula consistency
TEST(bic_formula_consistency) {
    // Verify the formula: BIC = k*log(n) - 2*loglik
    double loglik = -123.456;
    int k = 7;
    std::size_t n = 150;
    double bic = computeBIC(loglik, k, n);
    double expected = k * std::log(static_cast<double>(n)) - 2.0 * loglik;
    REQUIRE_APPROX(bic, expected, 1e-10);
}

// Test BIC increases with sample size (holding other factors constant)
TEST(bic_increases_with_sample_size) {
    // For fixed k and loglik, BIC should increase with n
    double loglik = -100.0;
    int k = 5;

    double bic_100 = computeBIC(loglik, k, 100);
    double bic_200 = computeBIC(loglik, k, 200);
    double bic_1000 = computeBIC(loglik, k, 1000);

    REQUIRE(bic_200 > bic_100);
    REQUIRE(bic_1000 > bic_200);
}

// ============================================================================
// AICc Tests
// ============================================================================

// Test AICc with known values
TEST(aicc_known_values) {
    // AICc = AIC + 2k(k+1)/(n-k-1)
    // Example 1: k=3, n=100, loglik=-100
    // AIC = 2*3 - 2*(-100) = 206
    // correction = 2*3*4/(100-3-1) = 24/96 = 0.25
    // AICc = 206 + 0.25 = 206.25
    double aicc1 = computeAICc(-100.0, 3, 100);
    double aic1 = computeAIC(-100.0, 3);
    double correction1 = (2.0 * 3.0 * 4.0) / (100.0 - 3.0 - 1.0);
    double expected1 = aic1 + correction1;
    REQUIRE_APPROX(aicc1, expected1, 1e-10);

    // Example 2: k=5, n=50, loglik=-250.5
    // AIC = 2*5 - 2*(-250.5) = 511
    // correction = 2*5*6/(50-5-1) = 60/44 = 1.363636...
    // AICc = 511 + 1.363636...
    double aicc2 = computeAICc(-250.5, 5, 50);
    double aic2 = computeAIC(-250.5, 5);
    double correction2 = (2.0 * 5.0 * 6.0) / (50.0 - 5.0 - 1.0);
    double expected2 = aic2 + correction2;
    REQUIRE_APPROX(aicc2, expected2, 1e-10);
}

// Test AICc converges to AIC for large n
TEST(aicc_converges_to_aic) {
    // As n → ∞, AICc → AIC
    double loglik = -100.0;
    int k = 3;

    double aic = computeAIC(loglik, k);

    // For very large n, correction term approaches 0
    double aicc_1000 = computeAICc(loglik, k, 1000);
    double aicc_10000 = computeAICc(loglik, k, 10000);
    double aicc_100000 = computeAICc(loglik, k, 100000);

    // Check that AICc gets closer to AIC as n increases
    double diff_1000 = std::abs(aicc_1000 - aic);
    double diff_10000 = std::abs(aicc_10000 - aic);
    double diff_100000 = std::abs(aicc_100000 - aic);

    REQUIRE(diff_10000 < diff_1000);
    REQUIRE(diff_100000 < diff_10000);

    // For very large n, they should be very close
    REQUIRE_APPROX(aicc_100000, aic, 0.01);
}

// Test AICc formula consistency
TEST(aicc_formula_consistency) {
    // Verify the formula: AICc = AIC + 2k(k+1)/(n-k-1)
    double loglik = -123.456;
    int k = 7;
    std::size_t n = 150;

    double aicc = computeAICc(loglik, k, n);
    double aic = computeAIC(loglik, k);
    double correction = (2.0 * k * (k + 1.0)) / (static_cast<double>(n) - k - 1.0);
    double expected = aic + correction;

    REQUIRE_APPROX(aicc, expected, 1e-10);
}

// Test AICc is always >= AIC (correction is always positive)
TEST(aicc_greater_than_or_equal_aic) {
    double loglik = -100.0;
    int k = 5;

    // For various sample sizes, AICc >= AIC
    for (std::size_t n : {50, 100, 200, 500, 1000}) {
        if (n > static_cast<std::size_t>(k + 1)) {
            double aic = computeAIC(loglik, k);
            double aicc = computeAICc(loglik, k, n);
            REQUIRE(aicc >= aic);
        }
    }
}

// Test AICc with small sample size
TEST(aicc_small_sample) {
    // k=3, n=20, loglik=-50
    // AIC = 2*3 - 2*(-50) = 106
    // correction = 2*3*4/(20-3-1) = 24/16 = 1.5
    // AICc = 106 + 1.5 = 107.5
    double aicc = computeAICc(-50.0, 3, 20);
    double expected = 106.0 + 1.5;
    REQUIRE_APPROX(aicc, expected, 1e-10);
}

// ============================================================================
// Comparison Tests
// ============================================================================

// Test relative ordering of AIC, BIC, AICc
TEST(criteria_relative_ordering) {
    // For typical scenarios with moderate sample sizes and parameters:
    // - AICc > AIC (always, due to correction)
    // - BIC can be greater or less than AIC depending on n and k

    double loglik = -100.0;
    int k = 5;
    std::size_t n = 100;

    double aic = computeAIC(loglik, k);
    double bic = computeBIC(loglik, k, n);
    double aicc = computeAICc(loglik, k, n);

    // AICc should always be >= AIC
    REQUIRE(aicc >= aic);

    // For n=100, k=5: k*log(n) = 5*4.605 ≈ 23.03 > 2*k = 10
    // So BIC should be larger than AIC
    REQUIRE(bic > aic);
}

// Test that all criteria prefer better log-likelihood
TEST(criteria_prefer_better_loglik) {
    // Higher log-likelihood (closer to 0, less negative) should give lower IC
    int k = 5;
    std::size_t n = 100;

    double loglik_worse = -200.0;
    double loglik_better = -100.0;

    double aic_worse = computeAIC(loglik_worse, k);
    double aic_better = computeAIC(loglik_better, k);
    REQUIRE(aic_better < aic_worse);

    double bic_worse = computeBIC(loglik_worse, k, n);
    double bic_better = computeBIC(loglik_better, k, n);
    REQUIRE(bic_better < bic_worse);

    double aicc_worse = computeAICc(loglik_worse, k, n);
    double aicc_better = computeAICc(loglik_better, k, n);
    REQUIRE(aicc_better < aicc_worse);
}

// Test that all criteria penalize more parameters
TEST(criteria_penalize_parameters) {
    // More parameters should give higher IC (holding loglik constant)
    double loglik = -100.0;
    std::size_t n = 100;

    int k_fewer = 3;
    int k_more = 10;

    double aic_fewer = computeAIC(loglik, k_fewer);
    double aic_more = computeAIC(loglik, k_more);
    REQUIRE(aic_more > aic_fewer);

    double bic_fewer = computeBIC(loglik, k_fewer, n);
    double bic_more = computeBIC(loglik, k_more, n);
    REQUIRE(bic_more > bic_fewer);

    double aicc_fewer = computeAICc(loglik, k_fewer, n);
    double aicc_more = computeAICc(loglik, k_more, n);
    REQUIRE(aicc_more > aicc_fewer);
}

// ============================================================================
// Edge Cases
// ============================================================================

// Test with very small log-likelihood (very negative)
TEST(criteria_very_negative_loglik) {
    double loglik = -1e6;
    int k = 5;
    std::size_t n = 1000;

    double aic = computeAIC(loglik, k);
    double bic = computeBIC(loglik, k, n);
    double aicc = computeAICc(loglik, k, n);

    // Should produce large positive values
    REQUIRE(aic > 0);
    REQUIRE(bic > 0);
    REQUIRE(aicc > 0);

    // Verify exact values
    REQUIRE_APPROX(aic, 2.0 * k - 2.0 * loglik, 1e-6);
    REQUIRE_APPROX(bic, k * std::log(static_cast<double>(n)) - 2.0 * loglik, 1e-6);
}

// Test with large sample size
TEST(criteria_large_sample_size) {
    double loglik = -100.0;
    int k = 5;
    std::size_t n = 1000000;

    double aic = computeAIC(loglik, k);
    double bic = computeBIC(loglik, k, n);
    double aicc = computeAICc(loglik, k, n);

    // All should be valid numbers
    REQUIRE(!std::isnan(aic));
    REQUIRE(!std::isnan(bic));
    REQUIRE(!std::isnan(aicc));
    REQUIRE(!std::isinf(aic));
    REQUIRE(!std::isinf(bic));
    REQUIRE(!std::isinf(aicc));

    // BIC should be much larger than AIC for large n
    REQUIRE(bic > aic);

    // AICc should be very close to AIC for large n
    REQUIRE_APPROX(aicc, aic, 0.001);
}

int main() {
    report_test_results("Information Criteria Selection");
    return get_test_result();
}
