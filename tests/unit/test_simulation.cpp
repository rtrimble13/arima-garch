#include "ag/simulation/ArimaGarchSimulator.hpp"
#include "ag/simulation/Innovations.hpp"

#include <cmath>

#include "test_framework.hpp"

using ag::models::ArimaGarchSpec;
using ag::models::composite::ArimaGarchParameters;
using ag::simulation::ArimaGarchSimulator;
using ag::simulation::Innovations;

// ============================================================================
// Innovations Tests
// ============================================================================

// Test that Innovations can be constructed and generates values
TEST(innovations_construction) {
    Innovations innov(42);
    double val = innov.drawNormal();
    // Just verify it returns a finite value
    REQUIRE(std::isfinite(val));
}

// Test reproducibility: same seed produces same sequence
TEST(innovations_reproducibility) {
    Innovations innov1(12345);
    Innovations innov2(12345);

    for (int i = 0; i < 100; ++i) {
        double val1 = innov1.drawNormal();
        double val2 = innov2.drawNormal();
        REQUIRE_APPROX(val1, val2, 1e-15);
    }
}

// Test that different seeds produce different sequences
TEST(innovations_different_seeds) {
    Innovations innov1(12345);
    Innovations innov2(54321);

    double val1 = innov1.drawNormal();
    double val2 = innov2.drawNormal();

    // With high probability, these should be different
    // (technically could be equal, but extremely unlikely)
    REQUIRE(std::abs(val1 - val2) > 1e-10);
}

// Test reseed functionality
TEST(innovations_reseed) {
    Innovations innov(12345);
    double val1 = innov.drawNormal();

    innov.reseed(12345);
    double val2 = innov.drawNormal();

    REQUIRE_APPROX(val1, val2, 1e-15);
}

// Test Student-t stub throws error
TEST(innovations_student_t_stub) {
    Innovations innov(42);
    bool caught_exception = false;
    try {
        innov.drawStudentT(5.0);
    } catch (const std::runtime_error&) {
        caught_exception = true;
    }
    REQUIRE(caught_exception);
}

// ============================================================================
// ArimaGarchSimulator Construction Tests
// ============================================================================

// Test basic construction
TEST(simulator_construction) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    // Set valid parameters
    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.5;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchSimulator simulator(spec, params);
    // Construction should succeed without throwing
    REQUIRE(true);
}

// Test construction with invalid parameters
TEST(simulator_invalid_garch_params) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.5;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = -0.1;  // Invalid: omega must be > 0
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    bool caught_exception = false;
    try {
        ArimaGarchSimulator simulator(spec, params);
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }

    REQUIRE(caught_exception);
}

// ============================================================================
// Simulation Tests
// ============================================================================

// Test basic simulation runs without error
TEST(simulation_basic) {
    ArimaGarchSpec spec(0, 0, 0, 1, 1);
    ArimaGarchParameters params(spec);

    // White noise mean
    params.arima_params.intercept = 0.0;

    // Simple GARCH(1,1)
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchSimulator simulator(spec, params);
    auto result = simulator.simulate(100, 12345);

    REQUIRE(result.returns.size() == 100);
    REQUIRE(result.volatilities.size() == 100);
}

// Test reproducibility: same seed produces identical output
TEST(simulation_reproducibility) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.5;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchSimulator simulator(spec, params);

    auto result1 = simulator.simulate(200, 42);
    auto result2 = simulator.simulate(200, 42);

    // Check all returns are identical
    for (size_t i = 0; i < result1.returns.size(); ++i) {
        REQUIRE_APPROX(result1.returns[i], result2.returns[i], 1e-15);
    }

    // Check all volatilities are identical
    for (size_t i = 0; i < result1.volatilities.size(); ++i) {
        REQUIRE_APPROX(result1.volatilities[i], result2.volatilities[i], 1e-15);
    }
}

// Test different seeds produce different results
TEST(simulation_different_seeds) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.5;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchSimulator simulator(spec, params);

    auto result1 = simulator.simulate(100, 12345);
    auto result2 = simulator.simulate(100, 54321);

    // At least one return should be different (with high probability, most will be)
    bool found_difference = false;
    for (size_t i = 0; i < result1.returns.size(); ++i) {
        if (std::abs(result1.returns[i] - result2.returns[i]) > 1e-10) {
            found_difference = true;
            break;
        }
    }
    REQUIRE(found_difference);
}

// Test shape correctness: output has correct dimensions
TEST(simulation_shape) {
    ArimaGarchSpec spec(2, 0, 1, 1, 2);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.05;
    params.arima_params.ar_coef[0] = 0.3;
    params.arima_params.ar_coef[1] = 0.2;
    params.arima_params.ma_coef[0] = 0.4;
    params.garch_params.omega = 0.05;
    params.garch_params.alpha_coef[0] = 0.05;
    params.garch_params.alpha_coef[1] = 0.05;
    params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchSimulator simulator(spec, params);

    for (int length : {1, 10, 100, 1000}) {
        auto result = simulator.simulate(length, 42);
        REQUIRE(static_cast<int>(result.returns.size()) == length);
        REQUIRE(static_cast<int>(result.volatilities.size()) == length);
    }
}

// Test all volatilities are positive
TEST(simulation_positive_volatilities) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.0;
    params.arima_params.ar_coef[0] = 0.5;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.15;
    params.garch_params.beta_coef[0] = 0.75;

    ArimaGarchSimulator simulator(spec, params);
    auto result = simulator.simulate(500, 42);

    for (double vol : result.volatilities) {
        REQUIRE(vol > 0.0);
    }
}

// Test invalid length throws exception
TEST(simulation_invalid_length) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.5;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchSimulator simulator(spec, params);

    bool caught_exception = false;
    try {
        simulator.simulate(0, 42);
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }
    REQUIRE(caught_exception);

    caught_exception = false;
    try {
        simulator.simulate(-10, 42);
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }
    REQUIRE(caught_exception);
}

// ============================================================================
// Main test runner
// ============================================================================

int main() {
    report_test_results("Simulation");
    return get_test_result();
}
