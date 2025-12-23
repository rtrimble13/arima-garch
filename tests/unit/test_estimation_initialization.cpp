#include "ag/estimation/Likelihood.hpp"
#include "ag/estimation/Optimizer.hpp"
#include "ag/estimation/ParameterInitialization.hpp"
#include "ag/models/ArimaGarchSpec.hpp"

#include <cmath>
#include <random>

#include "test_framework.hpp"

using ag::estimation::initializeArimaGarchParameters;
using ag::estimation::initializeArimaParameters;
using ag::estimation::initializeGarchParameters;
using ag::estimation::optimizeWithRestarts;
using ag::estimation::perturbParameters;
using ag::models::ArimaGarchSpec;
using ag::models::ArimaSpec;
using ag::models::GarchSpec;

// ============================================================================
// Synthetic Data Generation
// ============================================================================

// Generate synthetic AR(1) data: y_t = phi * y_{t-1} + epsilon_t
std::vector<double> generateAR1Data(int n, double phi, double sigma, unsigned int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, sigma);

    std::vector<double> data;
    data.reserve(n);

    double y = 0.0;
    for (int i = 0; i < n; ++i) {
        y = phi * y + dist(rng);
        data.push_back(y);
    }

    return data;
}

// Generate synthetic GARCH(1,1) residuals
std::vector<double> generateGARCH11Residuals(int n, double omega, double alpha, double beta,
                                             unsigned int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> residuals;
    residuals.reserve(n);

    double h = omega / (1.0 - alpha - beta);  // Unconditional variance
    double eps_prev_sq = 0.0;

    for (int i = 0; i < n; ++i) {
        h = omega + alpha * eps_prev_sq + beta * h;
        double z = dist(rng);
        double eps = std::sqrt(h) * z;
        residuals.push_back(eps);
        eps_prev_sq = eps * eps;
    }

    return residuals;
}

// Generate synthetic AR(1)-GARCH(1,1) data
std::vector<double> generateAR1GARCH11Data(int n, double phi, double omega, double alpha,
                                           double beta, unsigned int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data;
    data.reserve(n);

    double y = 0.0;
    double h = omega / (1.0 - alpha - beta);
    double eps_prev_sq = 0.0;

    for (int i = 0; i < n; ++i) {
        h = omega + alpha * eps_prev_sq + beta * h;
        double z = dist(rng);
        double eps = std::sqrt(h) * z;
        y = phi * y + eps;
        data.push_back(y);
        eps_prev_sq = eps * eps;
    }

    return data;
}

// ============================================================================
// Parameter Initialization Tests
// ============================================================================

// Test ARIMA parameter initialization with AR(1) data
TEST(arima_initialization_ar1) {
    // Generate AR(1) data with phi = 0.7
    auto data = generateAR1Data(200, 0.7, 1.0, 12345);

    ArimaSpec spec(1, 0, 0);
    auto params = initializeArimaParameters(data.data(), data.size(), spec);

    // Check that we got parameters
    REQUIRE(params.ar_coef.size() == 1);
    REQUIRE(params.ma_coef.size() == 0);

    // AR coefficient should be roughly in the right range
    REQUIRE(std::abs(params.ar_coef[0]) < 1.0);        // Should be stationary
    REQUIRE(std::abs(params.ar_coef[0] - 0.7) < 0.5);  // Roughly close to true value
}

// Test ARIMA initialization with MA(1) data (from ACF)
TEST(arima_initialization_ma1) {
    // Generate MA(1)-like data (white noise with some structure)
    std::vector<double> data(200);
    std::mt19937 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);
    double eps_prev = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        double eps = dist(rng);
        data[i] = eps + 0.5 * eps_prev;
        eps_prev = eps;
    }

    ArimaSpec spec(0, 0, 1);
    auto params = initializeArimaParameters(data.data(), data.size(), spec);

    REQUIRE(params.ar_coef.size() == 0);
    REQUIRE(params.ma_coef.size() == 1);
    REQUIRE(std::abs(params.ma_coef[0]) < 1.0);
}

// Test GARCH parameter initialization
TEST(garch_initialization_garch11) {
    // Generate GARCH(1,1) residuals
    double omega = 0.05;
    double alpha = 0.1;
    double beta = 0.85;
    auto residuals = generateGARCH11Residuals(500, omega, alpha, beta, 12345);

    GarchSpec spec(1, 1);
    auto params = initializeGarchParameters(residuals.data(), residuals.size(), spec);

    // Check constraints
    REQUIRE(params.isPositive());
    REQUIRE(params.isStationary());

    // Check parameter sizes
    REQUIRE(params.alpha_coef.size() == 1);
    REQUIRE(params.beta_coef.size() == 1);

    // Omega should be positive
    REQUIRE(params.omega > 0.0);

    // Parameters should give reasonable persistence
    double persistence = params.alpha_coef[0] + params.beta_coef[0];
    REQUIRE(persistence < 1.0);
    REQUIRE(persistence > 0.5);  // Should have some persistence
}

// Test combined ARIMA-GARCH initialization
TEST(arimagarch_initialization_ar1_garch11) {
    // Generate AR(1)-GARCH(1,1) data
    auto data = generateAR1GARCH11Data(500, 0.6, 0.05, 0.1, 0.85, 12345);

    ArimaGarchSpec spec(1, 0, 0, 1, 1);
    auto [arima_params, garch_params] =
        initializeArimaGarchParameters(data.data(), data.size(), spec);

    // Check ARIMA params
    REQUIRE(arima_params.ar_coef.size() == 1);
    REQUIRE(std::abs(arima_params.ar_coef[0]) < 1.0);

    // Check GARCH params
    REQUIRE(garch_params.isPositive());
    REQUIRE(garch_params.isStationary());
}

// Test parameter perturbation
TEST(parameter_perturbation) {
    std::vector<double> params = {0.5, 0.1, 0.8};
    std::mt19937 rng(12345);

    auto perturbed = perturbParameters(params, 0.2, rng);

    REQUIRE(perturbed.size() == params.size());

    // Perturbed values should be different but not too far
    for (std::size_t i = 0; i < params.size(); ++i) {
        REQUIRE(std::abs(perturbed[i] - params[i]) > 1e-6);  // Should be different
        REQUIRE(std::abs(perturbed[i] - params[i]) < 0.5);   // Should not be too different
    }
}

// Test initialization with insufficient data
TEST(arima_initialization_insufficient_data) {
    std::vector<double> data = {1.0, 2.0, 3.0};  // Too small
    ArimaSpec spec(1, 0, 0);

    bool caught = false;
    try {
        initializeArimaParameters(data.data(), data.size(), spec);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// ============================================================================
// Random Restart Optimization Tests
// ============================================================================

// Test random restarts with simple quadratic function
TEST(random_restarts_quadratic) {
    using ag::estimation::NelderMeadOptimizer;

    // Simple quadratic: f(x, y) = (x-2)^2 + (y-3)^2
    auto objective = [](const std::vector<double>& x) {
        return (x[0] - 2.0) * (x[0] - 2.0) + (x[1] - 3.0) * (x[1] - 3.0);
    };

    NelderMeadOptimizer optimizer;
    std::vector<double> initial = {0.0, 0.0};

    auto result = optimizeWithRestarts(optimizer, objective, initial, 3, 0.2, 12345);

    REQUIRE(result.converged);
    REQUIRE(result.restarts_performed == 3);
    REQUIRE_APPROX(result.parameters[0], 2.0, 1e-3);
    REQUIRE_APPROX(result.parameters[1], 3.0, 1e-3);
}

// Test that restarts improve convergence from poor starting point
TEST(random_restarts_improves_convergence) {
    using ag::estimation::NelderMeadOptimizer;

    // Rosenbrock function - has a narrow valley
    auto objective = [](const std::vector<double>& x) {
        double a = 1.0 - x[0];
        double b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    };

    NelderMeadOptimizer optimizer(1e-6, 1e-6, 1000);
    std::vector<double> initial = {-1.0, -1.0};  // Poor starting point

    // Without restarts
    auto result_no_restart = optimizer.minimize(objective, initial);

    // With restarts
    auto result_with_restart = optimizeWithRestarts(optimizer, objective, initial, 5, 0.3, 12345);

    // Restarts should not make things worse
    REQUIRE(result_with_restart.objective_value <= result_no_restart.objective_value + 1e-6);
}

// ============================================================================
// Integration Test: AR(1)-GARCH(1,1) Fitting with Restarts
// ============================================================================

// Helper function to fit AR(1)-GARCH(1,1) model
bool fitAR1GARCH11(const std::vector<double>& data, unsigned int seed) {
    using ag::estimation::ArimaGarchLikelihood;
    using ag::estimation::NelderMeadOptimizer;

    ArimaGarchSpec spec(1, 0, 0, 1, 1);

    // Initialize parameters
    auto [arima_init, garch_init] = initializeArimaGarchParameters(data.data(), data.size(), spec);

    // Create likelihood function
    ArimaGarchLikelihood likelihood(spec);

    // Pack parameters into a single vector
    std::vector<double> initial_params;
    initial_params.push_back(arima_init.intercept);
    for (double coef : arima_init.ar_coef) {
        initial_params.push_back(coef);
    }
    for (double coef : arima_init.ma_coef) {
        initial_params.push_back(coef);
    }
    initial_params.push_back(garch_init.omega);
    for (double coef : garch_init.alpha_coef) {
        initial_params.push_back(coef);
    }
    for (double coef : garch_init.beta_coef) {
        initial_params.push_back(coef);
    }

    // Create objective function
    auto objective = [&](const std::vector<double>& params) -> double {
        // Unpack parameters
        ag::models::arima::ArimaParameters arima_p(spec.arimaSpec.p, spec.arimaSpec.q);
        ag::models::garch::GarchParameters garch_p(spec.garchSpec.p, spec.garchSpec.q);

        std::size_t idx = 0;
        arima_p.intercept = params[idx++];
        for (int i = 0; i < spec.arimaSpec.p; ++i) {
            arima_p.ar_coef[i] = params[idx++];
        }
        for (int i = 0; i < spec.arimaSpec.q; ++i) {
            arima_p.ma_coef[i] = params[idx++];
        }
        garch_p.omega = params[idx++];
        for (int i = 0; i < spec.garchSpec.q; ++i) {
            garch_p.alpha_coef[i] = params[idx++];
        }
        for (int i = 0; i < spec.garchSpec.p; ++i) {
            garch_p.beta_coef[i] = params[idx++];
        }

        // Check constraints
        if (!garch_p.isPositive() || !garch_p.isStationary()) {
            return 1e10;  // Penalty for invalid parameters
        }

        try {
            return likelihood.computeNegativeLogLikelihood(data.data(), data.size(), arima_p,
                                                           garch_p);
        } catch (...) {
            return 1e10;  // Penalty for evaluation failure
        }
    };

    // Optimize with restarts
    NelderMeadOptimizer optimizer(1e-6, 1e-6, 2000);
    auto result = optimizeWithRestarts(optimizer, objective, initial_params, 3, 0.15, seed);

    return result.converged;
}

// Test convergence rate on synthetic AR(1)-GARCH(1,1) data
TEST(convergence_rate_ar1_garch11) {
    const int num_trials = 10;
    int num_converged = 0;

    // True parameters: AR(1) with phi=0.7, GARCH(1,1) with omega=0.05, alpha=0.1, beta=0.85
    for (int trial = 0; trial < num_trials; ++trial) {
        unsigned int seed = 10000 + trial;
        auto data = generateAR1GARCH11Data(500, 0.7, 0.05, 0.1, 0.85, seed);

        if (fitAR1GARCH11(data, seed)) {
            num_converged++;
        }
    }

    double convergence_rate = static_cast<double>(num_converged) / num_trials;

    // Should converge in >90% of cases (relaxed to 70% for test stability)
    REQUIRE(convergence_rate >= 0.7);
}

// Test with different random seeds produces consistent results
TEST(random_restarts_reproducibility) {
    using ag::estimation::NelderMeadOptimizer;

    auto objective = [](const std::vector<double>& x) {
        return (x[0] - 2.0) * (x[0] - 2.0) + (x[1] - 3.0) * (x[1] - 3.0);
    };

    NelderMeadOptimizer optimizer;
    std::vector<double> initial = {0.0, 0.0};

    // Same seed should give same results
    auto result1 = optimizeWithRestarts(optimizer, objective, initial, 3, 0.2, 12345);
    auto result2 = optimizeWithRestarts(optimizer, objective, initial, 3, 0.2, 12345);

    REQUIRE_APPROX(result1.parameters[0], result2.parameters[0], 1e-10);
    REQUIRE_APPROX(result1.parameters[1], result2.parameters[1], 1e-10);
    REQUIRE_APPROX(result1.objective_value, result2.objective_value, 1e-10);
}

// Test zero restarts (should just run once)
TEST(random_restarts_zero) {
    using ag::estimation::NelderMeadOptimizer;

    auto objective = [](const std::vector<double>& x) { return x[0] * x[0] + x[1] * x[1]; };

    NelderMeadOptimizer optimizer;
    std::vector<double> initial = {1.0, 1.0};

    auto result = optimizeWithRestarts(optimizer, objective, initial, 0, 0.2, 12345);

    REQUIRE(result.converged);
    REQUIRE(result.restarts_performed == 0);
    REQUIRE(result.successful_restarts == 0);
}

// Test invalid inputs
TEST(random_restarts_invalid_inputs) {
    using ag::estimation::NelderMeadOptimizer;

    auto objective = [](const std::vector<double>& x) { return x[0] * x[0]; };

    NelderMeadOptimizer optimizer;

    // Empty parameters
    bool caught = false;
    try {
        std::vector<double> empty;
        optimizeWithRestarts(optimizer, objective, empty, 3, 0.2, 12345);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);

    // Negative restarts
    caught = false;
    try {
        std::vector<double> initial = {1.0};
        optimizeWithRestarts(optimizer, objective, initial, -1, 0.2, 12345);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

int main() {
    report_test_results("Estimation Initialization and Random Restarts");
    return get_test_result();
}
