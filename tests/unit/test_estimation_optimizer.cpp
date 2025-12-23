#include "ag/estimation/Optimizer.hpp"

#include <cmath>
#include <vector>

#include "test_framework.hpp"

using ag::estimation::NelderMeadOptimizer;
using ag::estimation::OptimizationResult;

// ============================================================================
// Test Functions with Known Optima
// ============================================================================

// Simple quadratic function: f(x) = (x-2)^2 + (y-3)^2
// Minimum: (2, 3), f_min = 0
double quadratic_function(const std::vector<double>& x) {
    double dx = x[0] - 2.0;
    double dy = x[1] - 3.0;
    return dx * dx + dy * dy;
}

// Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
// Minimum: (1, 1), f_min = 0
// This is a challenging test case with a narrow valley
double rosenbrock_function(const std::vector<double>& x) {
    double a = 1.0 - x[0];
    double b = x[1] - x[0] * x[0];
    return a * a + 100.0 * b * b;
}

// Sphere function: f(x) = sum(x_i^2)
// Minimum: (0, ..., 0), f_min = 0
double sphere_function(const std::vector<double>& x) {
    double sum = 0.0;
    for (double val : x) {
        sum += val * val;
    }
    return sum;
}

// Simple 1D quadratic: f(x) = (x-5)^2
// Minimum: x = 5, f_min = 0
double simple_1d_quadratic(const std::vector<double>& x) {
    double dx = x[0] - 5.0;
    return dx * dx;
}

// Beale function: f(x,y) = (1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2
// Minimum: (3, 0.5), f_min = 0
double beale_function(const std::vector<double>& x) {
    double term1 = 1.5 - x[0] + x[0] * x[1];
    double term2 = 2.25 - x[0] + x[0] * x[1] * x[1];
    double term3 = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];
    return term1 * term1 + term2 * term2 + term3 * term3;
}

// Simple likelihood-like function: negative log of normal distribution
// f(x, mu, sigma) = 0.5 * ((x - mu) / sigma)^2 + log(sigma)
// For fixed data, minimizing over mu and sigma
double simple_likelihood(const std::vector<double>& params) {
    double mu = params[0];
    double log_sigma = params[1];
    double sigma = std::exp(log_sigma);  // Keep sigma positive

    // Simulated data: 5 observations from N(2.0, 1.5)
    std::vector<double> data = {2.1, 3.5, 1.8, 2.3, 0.9};

    double nll = 0.0;
    for (double x : data) {
        double z = (x - mu) / sigma;
        nll += 0.5 * z * z + log_sigma;
    }

    return nll;
}

// ============================================================================
// NelderMeadOptimizer Tests
// ============================================================================

// Test default constructor
TEST(optimizer_default_constructor) {
    NelderMeadOptimizer optimizer;
    REQUIRE(optimizer.getFunctionTolerance() > 0.0);
    REQUIRE(optimizer.getParameterTolerance() > 0.0);
    REQUIRE(optimizer.getMaxIterations() > 0);
}

// Test custom constructor
TEST(optimizer_custom_constructor) {
    NelderMeadOptimizer optimizer(1e-6, 1e-5, 500);
    REQUIRE_APPROX(optimizer.getFunctionTolerance(), 1e-6, 1e-10);
    REQUIRE_APPROX(optimizer.getParameterTolerance(), 1e-5, 1e-10);
    REQUIRE(optimizer.getMaxIterations() == 500);
}

// Test invalid tolerance throws exception
TEST(optimizer_invalid_ftol) {
    bool caught = false;
    try {
        NelderMeadOptimizer optimizer(-1e-6, 1e-5, 100);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

TEST(optimizer_invalid_xtol) {
    bool caught = false;
    try {
        NelderMeadOptimizer optimizer(1e-6, -1e-5, 100);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

TEST(optimizer_invalid_max_iterations) {
    bool caught = false;
    try {
        NelderMeadOptimizer optimizer(1e-6, 1e-5, 0);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test setters
TEST(optimizer_set_function_tolerance) {
    NelderMeadOptimizer optimizer;
    optimizer.setFunctionTolerance(1e-10);
    REQUIRE_APPROX(optimizer.getFunctionTolerance(), 1e-10, 1e-15);
}

TEST(optimizer_set_parameter_tolerance) {
    NelderMeadOptimizer optimizer;
    optimizer.setParameterTolerance(1e-10);
    REQUIRE_APPROX(optimizer.getParameterTolerance(), 1e-10, 1e-15);
}

TEST(optimizer_set_max_iterations) {
    NelderMeadOptimizer optimizer;
    optimizer.setMaxIterations(2000);
    REQUIRE(optimizer.getMaxIterations() == 2000);
}

// Test invalid setter values
TEST(optimizer_set_invalid_ftol) {
    NelderMeadOptimizer optimizer;
    bool caught = false;
    try {
        optimizer.setFunctionTolerance(-1.0);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test empty parameters throws
TEST(optimizer_empty_parameters) {
    NelderMeadOptimizer optimizer;
    std::vector<double> empty_params;

    bool caught = false;
    try {
        optimizer.minimize(quadratic_function, empty_params);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test 1D quadratic optimization
TEST(optimizer_simple_1d_quadratic) {
    NelderMeadOptimizer optimizer;
    std::vector<double> initial = {0.0};

    auto result = optimizer.minimize(simple_1d_quadratic, initial);

    REQUIRE(result.converged);
    REQUIRE(result.parameters.size() == 1);
    REQUIRE_APPROX(result.parameters[0], 5.0, 5e-4);
    REQUIRE_APPROX(result.objective_value, 0.0, 1e-6);
}

// Test 2D quadratic optimization
TEST(optimizer_quadratic_function) {
    NelderMeadOptimizer optimizer;
    std::vector<double> initial = {0.0, 0.0};

    auto result = optimizer.minimize(quadratic_function, initial);

    REQUIRE(result.converged);
    REQUIRE(result.parameters.size() == 2);
    REQUIRE_APPROX(result.parameters[0], 2.0, 1e-4);
    REQUIRE_APPROX(result.parameters[1], 3.0, 1e-4);
    REQUIRE_APPROX(result.objective_value, 0.0, 1e-6);
}

// Test quadratic from different starting point
TEST(optimizer_quadratic_different_start) {
    NelderMeadOptimizer optimizer;
    std::vector<double> initial = {10.0, -5.0};

    auto result = optimizer.minimize(quadratic_function, initial);

    REQUIRE(result.converged);
    REQUIRE_APPROX(result.parameters[0], 2.0, 1e-4);
    REQUIRE_APPROX(result.parameters[1], 3.0, 1e-4);
    REQUIRE_APPROX(result.objective_value, 0.0, 1e-6);
}

// Test Rosenbrock function (challenging)
TEST(optimizer_rosenbrock_function) {
    NelderMeadOptimizer optimizer(1e-6, 1e-6, 2000);  // More iterations for Rosenbrock
    std::vector<double> initial = {0.0, 0.0};

    auto result = optimizer.minimize(rosenbrock_function, initial);

    REQUIRE(result.converged);
    REQUIRE(result.parameters.size() == 2);
    // Rosenbrock is harder, use larger tolerance
    REQUIRE_APPROX(result.parameters[0], 1.0, 0.01);
    REQUIRE_APPROX(result.parameters[1], 1.0, 0.01);
    REQUIRE(result.objective_value < 0.01);
}

// Test sphere function (multi-dimensional)
TEST(optimizer_sphere_function_3d) {
    NelderMeadOptimizer optimizer;
    std::vector<double> initial = {1.0, 2.0, 3.0};

    auto result = optimizer.minimize(sphere_function, initial);

    REQUIRE(result.converged);
    REQUIRE(result.parameters.size() == 3);
    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE_APPROX(result.parameters[i], 0.0, 1e-4);
    }
    REQUIRE_APPROX(result.objective_value, 0.0, 1e-6);
}

// Test sphere function (higher dimensional)
TEST(optimizer_sphere_function_5d) {
    NelderMeadOptimizer optimizer(1e-6, 1e-6, 2000);
    std::vector<double> initial = {1.0, 2.0, -1.0, 3.0, -2.0};

    auto result = optimizer.minimize(sphere_function, initial);

    REQUIRE(result.converged);
    REQUIRE(result.parameters.size() == 5);
    for (std::size_t i = 0; i < 5; ++i) {
        REQUIRE_APPROX(result.parameters[i], 0.0, 1e-3);
    }
    REQUIRE(result.objective_value < 1e-4);
}

// Test Beale function
TEST(optimizer_beale_function) {
    NelderMeadOptimizer optimizer(1e-6, 1e-6, 2000);
    std::vector<double> initial = {1.0, 1.0};

    auto result = optimizer.minimize(beale_function, initial);

    REQUIRE(result.converged);
    REQUIRE(result.parameters.size() == 2);
    REQUIRE_APPROX(result.parameters[0], 3.0, 0.01);
    REQUIRE_APPROX(result.parameters[1], 0.5, 0.01);
    REQUIRE(result.objective_value < 0.01);
}

// Test simple likelihood function
TEST(optimizer_simple_likelihood) {
    NelderMeadOptimizer optimizer;
    std::vector<double> initial = {0.0, 0.0};  // Initial guess: mu=0, log_sigma=0 (sigma=1)

    auto result = optimizer.minimize(simple_likelihood, initial);

    REQUIRE(result.converged);
    REQUIRE(result.parameters.size() == 2);

    // True values from data: mean ≈ 2.12, std ≈ 0.93
    double estimated_mu = result.parameters[0];
    double estimated_sigma = std::exp(result.parameters[1]);

    REQUIRE_APPROX(estimated_mu, 2.12, 0.1);     // Within 0.1 of true mean
    REQUIRE_APPROX(estimated_sigma, 0.93, 0.2);  // Within 0.2 of true std
    REQUIRE(result.objective_value > 0.0);       // Likelihood should be positive
}

// Test convergence with tight tolerance
TEST(optimizer_tight_tolerance) {
    NelderMeadOptimizer optimizer(1e-10, 1e-10, 5000);
    std::vector<double> initial = {0.0, 0.0};

    auto result = optimizer.minimize(quadratic_function, initial);

    REQUIRE(result.converged);
    REQUIRE_APPROX(result.parameters[0], 2.0, 1e-5);
    REQUIRE_APPROX(result.parameters[1], 3.0, 1e-5);
    REQUIRE_APPROX(result.objective_value, 0.0, 1e-8);
}

// Test max iterations limit
TEST(optimizer_max_iterations_limit) {
    NelderMeadOptimizer optimizer(1e-10, 1e-10, 10);  // Very few iterations
    std::vector<double> initial = {100.0, 100.0};     // Far from optimum

    auto result = optimizer.minimize(quadratic_function, initial);

    // Should not converge with so few iterations from far away
    REQUIRE(!result.converged);
    REQUIRE(result.iterations == 10);
    REQUIRE(result.message == "Maximum iterations reached");
}

// Test starting at optimum
TEST(optimizer_start_at_optimum) {
    NelderMeadOptimizer optimizer;
    std::vector<double> initial = {2.0, 3.0};  // Already at optimum

    auto result = optimizer.minimize(quadratic_function, initial);

    REQUIRE(result.converged);
    REQUIRE(result.iterations < 50);  // Should converge quickly
    REQUIRE_APPROX(result.parameters[0], 2.0, 1e-4);
    REQUIRE_APPROX(result.parameters[1], 3.0, 1e-4);
    REQUIRE_APPROX(result.objective_value, 0.0, 1e-6);
}

// Test optimization near optimum
TEST(optimizer_start_near_optimum) {
    NelderMeadOptimizer optimizer;
    std::vector<double> initial = {2.1, 2.9};  // Very close to optimum

    auto result = optimizer.minimize(quadratic_function, initial);

    REQUIRE(result.converged);
    REQUIRE(result.iterations < 100);  // Should converge quickly
    REQUIRE_APPROX(result.parameters[0], 2.0, 1e-4);
    REQUIRE_APPROX(result.parameters[1], 3.0, 1e-4);
}

// Test consistency across multiple runs
TEST(optimizer_consistency) {
    NelderMeadOptimizer optimizer;
    std::vector<double> initial = {0.5, 0.5};

    // Run optimization multiple times
    auto result1 = optimizer.minimize(quadratic_function, initial);
    auto result2 = optimizer.minimize(quadratic_function, initial);

    REQUIRE(result1.converged && result2.converged);
    REQUIRE_APPROX(result1.parameters[0], result2.parameters[0], 1e-6);
    REQUIRE_APPROX(result1.parameters[1], result2.parameters[1], 1e-6);
    REQUIRE_APPROX(result1.objective_value, result2.objective_value, 1e-8);
}

// Test iteration count is reasonable
TEST(optimizer_iteration_count) {
    NelderMeadOptimizer optimizer;
    std::vector<double> initial = {0.0, 0.0};

    auto result = optimizer.minimize(quadratic_function, initial);

    REQUIRE(result.converged);
    REQUIRE(result.iterations > 0);
    REQUIRE(result.iterations < 500);  // Should not need many iterations for simple function
}

// Test result message on convergence
TEST(optimizer_convergence_message) {
    NelderMeadOptimizer optimizer;
    std::vector<double> initial = {0.0, 0.0};

    auto result = optimizer.minimize(quadratic_function, initial);

    REQUIRE(result.converged);
    REQUIRE(result.message == "Converged");
}

int main() {
    report_test_results("Estimation Optimizer");
    return get_test_result();
}
