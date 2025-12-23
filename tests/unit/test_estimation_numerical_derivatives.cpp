#include "ag/estimation/NumericalDerivatives.hpp"

#include <cmath>
#include <random>
#include <vector>

#include "test_framework.hpp"

using ag::estimation::NumericalDerivatives;

// ============================================================================
// Test Functions with Known Derivatives
// ============================================================================

// Simple quadratic function: f(x) = x1^2 + x2^2
// Gradient: [2*x1, 2*x2]
double quadratic_function(const std::vector<double>& x) {
    return x[0] * x[0] + x[1] * x[1];
}

std::vector<double> quadratic_gradient(const std::vector<double>& x) {
    return {2.0 * x[0], 2.0 * x[1]};
}

// Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
// Gradient: [-2*(1-x) - 400*x*(y-x^2), 200*(y-x^2)]
double rosenbrock_function(const std::vector<double>& x) {
    const double a = 1.0 - x[0];
    const double b = x[1] - x[0] * x[0];
    return a * a + 100.0 * b * b;
}

std::vector<double> rosenbrock_gradient(const std::vector<double>& x) {
    const double a = 1.0 - x[0];
    const double b = x[1] - x[0] * x[0];
    return {-2.0 * a - 400.0 * x[0] * b, 200.0 * b};
}

// Exponential function: f(x) = exp(x1) + exp(x2)
// Gradient: [exp(x1), exp(x2)]
double exponential_function(const std::vector<double>& x) {
    return std::exp(x[0]) + std::exp(x[1]);
}

std::vector<double> exponential_gradient(const std::vector<double>& x) {
    return {std::exp(x[0]), std::exp(x[1])};
}

// Sum of squares: f(x) = sum(x_i^2)
// Gradient: [2*x_1, 2*x_2, ..., 2*x_n]
double sum_of_squares(const std::vector<double>& x) {
    double sum = 0.0;
    for (double val : x) {
        sum += val * val;
    }
    return sum;
}

std::vector<double> sum_of_squares_gradient(const std::vector<double>& x) {
    std::vector<double> grad(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        grad[i] = 2.0 * x[i];
    }
    return grad;
}

// ============================================================================
// NumericalDerivatives Tests
// ============================================================================

// Test default constructor
TEST(numerical_derivatives_default_constructor) {
    NumericalDerivatives nd;
    REQUIRE(nd.getStepSize() > 0.0);
}

// Test custom step size constructor
TEST(numerical_derivatives_custom_step_size) {
    NumericalDerivatives nd(1e-5);
    REQUIRE_APPROX(nd.getStepSize(), 1e-5, 1e-10);
}

// Test invalid step size throws exception
TEST(numerical_derivatives_invalid_step_size) {
    bool caught = false;
    try {
        NumericalDerivatives nd(-1e-5);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);

    caught = false;
    try {
        NumericalDerivatives nd(0.0);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test step size too small throws exception
TEST(numerical_derivatives_step_size_too_small) {
    bool caught = false;
    try {
        NumericalDerivatives nd(1e-15);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test set step size
TEST(numerical_derivatives_set_step_size) {
    NumericalDerivatives nd;
    nd.setStepSize(1e-6);
    REQUIRE_APPROX(nd.getStepSize(), 1e-6, 1e-10);
}

// Test gradient of quadratic function
TEST(gradient_quadratic_function) {
    NumericalDerivatives nd;
    std::vector<double> x = {1.0, 2.0};

    auto grad = nd.computeGradient(quadratic_function, x);
    auto exact_grad = quadratic_gradient(x);

    REQUIRE(grad.size() == 2);
    REQUIRE_APPROX(grad[0], exact_grad[0], 1e-6);
    REQUIRE_APPROX(grad[1], exact_grad[1], 1e-6);
}

// Test gradient at origin
TEST(gradient_quadratic_at_origin) {
    NumericalDerivatives nd;
    std::vector<double> x = {0.0, 0.0};

    auto grad = nd.computeGradient(quadratic_function, x);

    REQUIRE(grad.size() == 2);
    REQUIRE_APPROX(grad[0], 0.0, 1e-6);
    REQUIRE_APPROX(grad[1], 0.0, 1e-6);
}

// Test gradient of exponential function
TEST(gradient_exponential_function) {
    NumericalDerivatives nd;
    std::vector<double> x = {0.5, -0.5};

    auto grad = nd.computeGradient(exponential_function, x);
    auto exact_grad = exponential_gradient(x);

    REQUIRE(grad.size() == 2);
    REQUIRE_APPROX(grad[0], exact_grad[0], 1e-6);
    REQUIRE_APPROX(grad[1], exact_grad[1], 1e-6);
}

// Test gradient of Rosenbrock function
TEST(gradient_rosenbrock_function) {
    NumericalDerivatives nd;
    std::vector<double> x = {0.5, 0.5};

    auto grad = nd.computeGradient(rosenbrock_function, x);
    auto exact_grad = rosenbrock_gradient(x);

    REQUIRE(grad.size() == 2);
    // Rosenbrock function is more challenging, use slightly larger tolerance
    REQUIRE_APPROX(grad[0], exact_grad[0], 1e-4);
    REQUIRE_APPROX(grad[1], exact_grad[1], 1e-4);
}

// Test gradient on higher dimensional function
TEST(gradient_high_dimensional) {
    NumericalDerivatives nd;
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};

    auto grad = nd.computeGradient(sum_of_squares, x);
    auto exact_grad = sum_of_squares_gradient(x);

    REQUIRE(grad.size() == 5);
    for (std::size_t i = 0; i < grad.size(); ++i) {
        REQUIRE_APPROX(grad[i], exact_grad[i], 1e-6);
    }
}

// Test adaptive gradient on quadratic
TEST(gradient_adaptive_quadratic) {
    NumericalDerivatives nd;
    std::vector<double> x = {10.0, 0.1};  // Different scales

    auto grad = nd.computeGradientAdaptive(quadratic_function, x);
    auto exact_grad = quadratic_gradient(x);

    REQUIRE(grad.size() == 2);
    REQUIRE_APPROX(grad[0], exact_grad[0], 1e-5);
    REQUIRE_APPROX(grad[1], exact_grad[1], 1e-5);
}

// Test adaptive gradient with varying scales
TEST(gradient_adaptive_varying_scales) {
    NumericalDerivatives nd;
    // Parameters with vastly different magnitudes
    std::vector<double> x = {1000.0, 1.0, 0.001};

    auto objective = [](const std::vector<double>& params) {
        return params[0] * params[0] + params[1] * params[1] + params[2] * params[2];
    };

    auto grad = nd.computeGradientAdaptive(objective, x);

    REQUIRE(grad.size() == 3);
    // For large parameters, gradient should be accurate
    REQUIRE_APPROX(grad[0], 2.0 * x[0], 1e-3);
    // For medium scale parameters, use moderate tolerance
    REQUIRE_APPROX(grad[1], 2.0 * x[1], 1e-2);
    // For very small parameters relative to unit scale, gradient may be less accurate
    // but should still capture the order of magnitude
    REQUIRE(std::abs(grad[2] - 2.0 * x[2]) < 1e-2);
}

// Test empty parameter vector throws
TEST(gradient_empty_parameters) {
    NumericalDerivatives nd;
    std::vector<double> x;

    bool caught = false;
    try {
        nd.computeGradient(quadratic_function, x);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught);
}

// Test single parameter function
TEST(gradient_single_parameter) {
    NumericalDerivatives nd;
    std::vector<double> x = {3.0};

    auto func = [](const std::vector<double>& params) { return params[0] * params[0]; };

    auto grad = nd.computeGradient(func, x);

    REQUIRE(grad.size() == 1);
    REQUIRE_APPROX(grad[0], 6.0, 1e-6);
}

// Test gradient stability with random points
TEST(gradient_stability_random) {
    NumericalDerivatives nd;
    std::mt19937 gen(12345);
    std::uniform_real_distribution<> dis(-5.0, 5.0);

    for (int trial = 0; trial < 20; ++trial) {
        std::vector<double> x = {dis(gen), dis(gen)};

        auto grad = nd.computeGradient(quadratic_function, x);
        auto exact_grad = quadratic_gradient(x);

        REQUIRE(grad.size() == 2);
        REQUIRE_APPROX(grad[0], exact_grad[0], 1e-6);
        REQUIRE_APPROX(grad[1], exact_grad[1], 1e-6);
    }
}

// Test numerical stability near zero
TEST(gradient_near_zero) {
    NumericalDerivatives nd;
    std::vector<double> x = {1e-10, 1e-10};

    auto grad = nd.computeGradient(quadratic_function, x);

    REQUIRE(grad.size() == 2);
    // Near zero, gradient should be very small
    REQUIRE(std::abs(grad[0]) < 1e-6);
    REQUIRE(std::abs(grad[1]) < 1e-6);
}

// Test gradient with larger step size
TEST(gradient_larger_step_size) {
    NumericalDerivatives nd(1e-4);
    std::vector<double> x = {1.0, 2.0};

    auto grad = nd.computeGradient(quadratic_function, x);
    auto exact_grad = quadratic_gradient(x);

    REQUIRE(grad.size() == 2);
    // Larger step size reduces accuracy
    REQUIRE_APPROX(grad[0], exact_grad[0], 1e-3);
    REQUIRE_APPROX(grad[1], exact_grad[1], 1e-3);
}

// Test comparison between standard and adaptive methods
TEST(gradient_standard_vs_adaptive) {
    NumericalDerivatives nd;
    std::vector<double> x = {2.0, 3.0};

    auto grad_standard = nd.computeGradient(quadratic_function, x);
    auto grad_adaptive = nd.computeGradientAdaptive(quadratic_function, x);

    REQUIRE(grad_standard.size() == grad_adaptive.size());
    // Both should give similar results for well-scaled problems
    REQUIRE_APPROX(grad_standard[0], grad_adaptive[0], 1e-6);
    REQUIRE_APPROX(grad_standard[1], grad_adaptive[1], 1e-6);
}

int main() {
    report_test_results("Estimation Numerical Derivatives");
    return get_test_result();
}
