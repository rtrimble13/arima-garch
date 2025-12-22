#include "ag/estimation/Constraints.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "test_framework.hpp"

using ag::estimation::ArimaGarchTransform;
using ag::estimation::ParameterVector;

// ============================================================================
// ArimaGarchTransform Tests
// ============================================================================

// Test basic transform to constrained parameters
TEST(transform_to_constrained_basic) {
    // Test GARCH(1,1) with simple unconstrained parameters
    ParameterVector theta(3, 0.0);
    theta[0] = 0.0;  // omega: exp(0) = 1
    theta[1] = 0.0;  // alpha
    theta[2] = 0.0;  // beta

    ParameterVector params = ArimaGarchTransform::toConstrained(theta, 1, 1);

    REQUIRE(params.size() == 3);
    // omega should be positive
    REQUIRE(params[0] > 0.0);
    // alpha and beta should be non-negative
    REQUIRE(params[1] >= 0.0);
    REQUIRE(params[2] >= 0.0);
    // sum should be less than 1
    REQUIRE(params[1] + params[2] < 1.0);
}

// Test omega constraint (omega > 0)
TEST(transform_omega_positive) {
    // Test with various theta[0] values
    std::vector<double> theta_omega_values = {-5.0, -1.0, 0.0, 1.0, 5.0};

    for (double theta_omega : theta_omega_values) {
        ParameterVector theta(3, 0.0);
        theta[0] = theta_omega;

        ParameterVector params = ArimaGarchTransform::toConstrained(theta, 1, 1);

        // omega = exp(theta[0]) should always be positive
        REQUIRE(params[0] > 0.0);
        // Verify it matches expected value
        REQUIRE_APPROX(params[0], std::exp(theta_omega), 1e-10);
    }
}

// Test non-negativity of ARCH and GARCH coefficients
TEST(transform_coefficients_non_negative) {
    // Test with various unconstrained values, including negative
    std::vector<double> test_values = {-10.0, -1.0, 0.0, 1.0, 10.0};

    for (double val1 : test_values) {
        for (double val2 : test_values) {
            ParameterVector theta(3, 0.0);
            theta[0] = 0.0;
            theta[1] = val1;
            theta[2] = val2;

            ParameterVector params = ArimaGarchTransform::toConstrained(theta, 1, 1);

            // All coefficients should be non-negative
            REQUIRE(params[1] >= 0.0);
            REQUIRE(params[2] >= 0.0);
        }
    }
}

// Test stationarity constraint (sum < 1)
TEST(transform_stationarity_constraint) {
    // Test with random unconstrained parameters
    std::mt19937 gen(12345);  // Fixed seed for reproducibility
    std::uniform_real_distribution<> dis(-5.0, 5.0);

    for (int trial = 0; trial < 100; ++trial) {
        ParameterVector theta(3, 0.0);
        theta[0] = dis(gen);
        theta[1] = dis(gen);
        theta[2] = dis(gen);

        ParameterVector params = ArimaGarchTransform::toConstrained(theta, 1, 1);

        // Sum of alpha and beta should always be less than 1
        double sum = params[1] + params[2];
        REQUIRE(sum < 1.0);
        // Should also be less than MAX_PERSISTENCE
        REQUIRE(sum < 0.999);
    }
}

// Test GARCH(2,2) model
TEST(transform_garch_22) {
    ParameterVector theta(5, 0.0);
    theta[0] = 1.0;  // omega
    theta[1] = 0.5;  // alpha1
    theta[2] = 0.3;  // alpha2
    theta[3] = 0.2;  // beta1
    theta[4] = 0.1;  // beta2

    ParameterVector params = ArimaGarchTransform::toConstrained(theta, 2, 2);

    REQUIRE(params.size() == 5);
    REQUIRE(params[0] > 0.0);   // omega
    REQUIRE(params[1] >= 0.0);  // alpha1
    REQUIRE(params[2] >= 0.0);  // alpha2
    REQUIRE(params[3] >= 0.0);  // beta1
    REQUIRE(params[4] >= 0.0);  // beta2

    // Check stationarity
    double sum = params[1] + params[2] + params[3] + params[4];
    REQUIRE(sum < 1.0);
}

// Test validate constraints function
TEST(validate_constraints_valid) {
    // Valid GARCH(1,1) parameters
    ParameterVector params(3, 0.0);
    params[0] = 0.01;  // omega > 0
    params[1] = 0.1;   // alpha >= 0
    params[2] = 0.8;   // beta >= 0
    // sum = 0.9 < 1

    REQUIRE(ArimaGarchTransform::validateConstraints(params, 1, 1));
}

// Test validate constraints - omega not positive
TEST(validate_constraints_omega_not_positive) {
    ParameterVector params(3, 0.0);
    params[0] = 0.0;  // omega = 0 (not valid, must be > 0)
    params[1] = 0.1;
    params[2] = 0.8;

    REQUIRE(!ArimaGarchTransform::validateConstraints(params, 1, 1));

    params[0] = -0.01;  // negative omega
    REQUIRE(!ArimaGarchTransform::validateConstraints(params, 1, 1));
}

// Test validate constraints - negative coefficients
TEST(validate_constraints_negative_coefficients) {
    ParameterVector params(3, 0.0);
    params[0] = 0.01;
    params[1] = -0.1;  // negative alpha
    params[2] = 0.8;

    REQUIRE(!ArimaGarchTransform::validateConstraints(params, 1, 1));

    params[1] = 0.1;
    params[2] = -0.1;  // negative beta
    REQUIRE(!ArimaGarchTransform::validateConstraints(params, 1, 1));
}

// Test validate constraints - non-stationary (sum >= 1)
TEST(validate_constraints_non_stationary) {
    ParameterVector params(3, 0.0);
    params[0] = 0.01;
    params[1] = 0.5;
    params[2] = 0.5;  // sum = 1.0 (boundary, not valid)

    REQUIRE(!ArimaGarchTransform::validateConstraints(params, 1, 1));

    params[2] = 0.6;  // sum > 1
    REQUIRE(!ArimaGarchTransform::validateConstraints(params, 1, 1));
}

// Test validate constraints - wrong size
TEST(validate_constraints_wrong_size) {
    ParameterVector params(2, 0.0);  // Too small for GARCH(1,1)
    REQUIRE(!ArimaGarchTransform::validateConstraints(params, 1, 1));

    ParameterVector params2(4, 0.0);  // Too large for GARCH(1,1)
    REQUIRE(!ArimaGarchTransform::validateConstraints(params2, 1, 1));
}

// Test to_unconstrained inverse
TEST(transform_to_unconstrained_basic) {
    // Start with valid constrained parameters
    ParameterVector params(3, 0.0);
    params[0] = 0.01;  // omega
    params[1] = 0.1;   // alpha
    params[2] = 0.8;   // beta

    // Convert to unconstrained
    ParameterVector theta = ArimaGarchTransform::toUnconstrained(params, 1, 1);

    REQUIRE(theta.size() == 3);
    // omega: theta[0] = log(omega)
    REQUIRE_APPROX(theta[0], std::log(0.01), 1e-6);
}

// Test round-trip transformation
TEST(transform_round_trip) {
    // Start with unconstrained parameters
    ParameterVector theta_original(3, 0.0);
    theta_original[0] = 0.5;
    theta_original[1] = -0.3;
    theta_original[2] = 1.2;

    // Transform to constrained
    ParameterVector params = ArimaGarchTransform::toConstrained(theta_original, 1, 1);

    // Validate constraints
    REQUIRE(ArimaGarchTransform::validateConstraints(params, 1, 1));

    // Transform back to unconstrained
    ParameterVector theta_recovered = ArimaGarchTransform::toUnconstrained(params, 1, 1);

    // Transform again to constrained
    ParameterVector params_recovered = ArimaGarchTransform::toConstrained(theta_recovered, 1, 1);

    // The constrained parameters should be very close
    REQUIRE_APPROX(params[0], params_recovered[0], 1e-6);
    REQUIRE_APPROX(params[1], params_recovered[1], 1e-6);
    REQUIRE_APPROX(params[2], params_recovered[2], 1e-6);
}

// Test with random theta inputs - broad range
TEST(transform_random_theta_broad_range) {
    std::mt19937 gen(42);  // Fixed seed
    std::uniform_real_distribution<> dis(-10.0, 10.0);

    for (int trial = 0; trial < 200; ++trial) {
        ParameterVector theta(3, 0.0);
        theta[0] = dis(gen);
        theta[1] = dis(gen);
        theta[2] = dis(gen);

        ParameterVector params = ArimaGarchTransform::toConstrained(theta, 1, 1);

        // All constraints should be satisfied
        REQUIRE(ArimaGarchTransform::validateConstraints(params, 1, 1));

        // Explicitly check each constraint
        REQUIRE(params[0] > 0.0);              // omega > 0
        REQUIRE(params[1] >= 0.0);             // alpha >= 0
        REQUIRE(params[2] >= 0.0);             // beta >= 0
        REQUIRE(params[1] + params[2] < 1.0);  // stationarity
    }
}

// Test with extreme theta values
TEST(transform_extreme_theta_values) {
    std::vector<double> extreme_values = {-100.0, -50.0, -20.0, 20.0, 50.0, 100.0};

    for (double val : extreme_values) {
        ParameterVector theta(3, val);

        ParameterVector params = ArimaGarchTransform::toConstrained(theta, 1, 1);

        // Should still satisfy all constraints
        REQUIRE(ArimaGarchTransform::validateConstraints(params, 1, 1));
    }
}

// Test GARCH(3,2) with random inputs
TEST(transform_garch_32_random) {
    std::mt19937 gen(999);
    std::uniform_real_distribution<> dis(-5.0, 5.0);

    for (int trial = 0; trial < 50; ++trial) {
        ParameterVector theta(6, 0.0);  // 1 + 3 + 2
        for (std::size_t i = 0; i < theta.size(); ++i) {
            theta[i] = dis(gen);
        }

        ParameterVector params = ArimaGarchTransform::toConstrained(theta, 3, 2);

        REQUIRE(params.size() == 6);
        REQUIRE(ArimaGarchTransform::validateConstraints(params, 3, 2));

        // Check omega
        REQUIRE(params[0] > 0.0);

        // Check all alphas
        for (int i = 0; i < 3; ++i) {
            REQUIRE(params[1 + i] >= 0.0);
        }

        // Check all betas
        for (int j = 0; j < 2; ++j) {
            REQUIRE(params[1 + 3 + j] >= 0.0);
        }

        // Check stationarity
        double sum = 0.0;
        for (std::size_t i = 1; i < params.size(); ++i) {
            sum += params[i];
        }
        REQUIRE(sum < 1.0);
    }
}

// Test error handling - wrong theta size
TEST(transform_error_wrong_theta_size) {
    ParameterVector theta(2, 0.0);  // Too small for GARCH(1,1)

    bool caught_exception = false;
    try {
        ParameterVector params = ArimaGarchTransform::toConstrained(theta, 1, 1);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
    }
    REQUIRE(caught_exception);
}

// Test error handling - invalid p or q
TEST(transform_error_invalid_pq) {
    ParameterVector theta(3, 0.0);

    bool caught_exception = false;
    try {
        ParameterVector params = ArimaGarchTransform::toConstrained(theta, 0, 1);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
    }
    REQUIRE(caught_exception);

    caught_exception = false;
    try {
        ParameterVector params = ArimaGarchTransform::toConstrained(theta, 1, 0);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
    }
    REQUIRE(caught_exception);
}

// Test error handling - to_unconstrained with invalid params
TEST(transform_error_invalid_params_to_unconstrained) {
    // Non-stationary parameters
    ParameterVector params(3, 0.0);
    params[0] = 0.01;
    params[1] = 0.6;
    params[2] = 0.6;  // sum > 1

    bool caught_exception = false;
    try {
        ParameterVector theta = ArimaGarchTransform::toUnconstrained(params, 1, 1);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
    }
    REQUIRE(caught_exception);
}

// Test that different theta values produce different params
TEST(transform_different_theta_different_params) {
    ParameterVector theta1(3, 0.0);
    theta1[0] = 0.5;
    theta1[1] = 0.2;
    theta1[2] = 0.8;

    ParameterVector theta2(3, 0.0);
    theta2[0] = 1.5;
    theta2[1] = -0.3;
    theta2[2] = 1.5;

    ParameterVector params1 = ArimaGarchTransform::toConstrained(theta1, 1, 1);
    ParameterVector params2 = ArimaGarchTransform::toConstrained(theta2, 1, 1);

    // omega should be different
    REQUIRE(std::abs(params1[0] - params2[0]) > 1e-6);
}

// Test numerical stability with very small values
TEST(transform_numerical_stability_small) {
    ParameterVector theta(3, -100.0);  // Very small exp values

    ParameterVector params = ArimaGarchTransform::toConstrained(theta, 1, 1);

    // Should still be valid
    REQUIRE(ArimaGarchTransform::validateConstraints(params, 1, 1));
    REQUIRE(params[0] > 0.0);
    REQUIRE(std::isfinite(params[0]));
    REQUIRE(std::isfinite(params[1]));
    REQUIRE(std::isfinite(params[2]));
}

// Test numerical stability with very large values
TEST(transform_numerical_stability_large) {
    ParameterVector theta(3, 50.0);  // Very large exp values

    ParameterVector params = ArimaGarchTransform::toConstrained(theta, 1, 1);

    // Should still be valid
    REQUIRE(ArimaGarchTransform::validateConstraints(params, 1, 1));
    REQUIRE(std::isfinite(params[0]));
    REQUIRE(std::isfinite(params[1]));
    REQUIRE(std::isfinite(params[2]));
}

int main() {
    report_test_results("Estimation Constraints");
    return get_test_result();
}
