#pragma once

#include <cstddef>
#include <functional>
#include <vector>

namespace ag::estimation {

/**
 * @brief Numerical gradient computation using finite differences.
 *
 * NumericalDerivatives provides methods for computing gradients of objective
 * functions using central finite differences. This is useful when analytical
 * gradients are not available or difficult to compute.
 *
 * The central difference approximation for the i-th partial derivative is:
 *   ∂f/∂x_i ≈ [f(x + h*e_i) - f(x - h*e_i)] / (2*h)
 *
 * where e_i is the i-th unit vector and h is a small step size.
 *
 * Central differences provide O(h²) accuracy, which is superior to forward
 * or backward differences (O(h) accuracy), making them more suitable for
 * gradient-based optimization.
 */
class NumericalDerivatives {
public:
    /**
     * @brief Objective function type: takes a parameter vector and returns a scalar value.
     *
     * The function signature is: double f(const std::vector<double>& params)
     * where params is the point at which to evaluate the objective function.
     */
    using ObjectiveFunction = std::function<double(const std::vector<double>&)>;

    /**
     * @brief Default constructor with default step size.
     *
     * Uses a default step size of sqrt(machine epsilon) ≈ 1.5e-8 for double precision,
     * which is theoretically optimal for central differences.
     */
    NumericalDerivatives();

    /**
     * @brief Construct with custom step size.
     *
     * @param step_size The step size (h) to use for finite differences
     * @throws std::invalid_argument if step_size <= 0
     */
    explicit NumericalDerivatives(double step_size);

    /**
     * @brief Compute gradient using central finite differences.
     *
     * Computes the gradient ∇f at the given point x using central differences:
     *   ∂f/∂x_i ≈ [f(x + h*e_i) - f(x - h*e_i)] / (2*h)
     *
     * This method requires 2*n function evaluations where n is the dimension
     * of the parameter vector.
     *
     * @param objective The objective function to differentiate
     * @param params The point at which to compute the gradient
     * @return Gradient vector of the same size as params
     * @throws std::invalid_argument if params is empty
     */
    [[nodiscard]] std::vector<double> computeGradient(const ObjectiveFunction& objective,
                                                      const std::vector<double>& params) const;

    /**
     * @brief Compute gradient with adaptive step size for each parameter.
     *
     * Uses adaptive step sizing based on the magnitude of each parameter:
     *   h_i = base_step * max(|x_i|, 1.0)
     *
     * This provides better numerical stability when parameters have vastly
     * different scales.
     *
     * @param objective The objective function to differentiate
     * @param params The point at which to compute the gradient
     * @return Gradient vector of the same size as params
     * @throws std::invalid_argument if params is empty
     */
    [[nodiscard]] std::vector<double>
    computeGradientAdaptive(const ObjectiveFunction& objective,
                            const std::vector<double>& params) const;

    /**
     * @brief Get the current step size.
     * @return The step size used for finite differences
     */
    [[nodiscard]] double getStepSize() const noexcept { return step_size_; }

    /**
     * @brief Set a new step size.
     * @param step_size New step size (must be positive)
     * @throws std::invalid_argument if step_size <= 0
     */
    void setStepSize(double step_size);

private:
    double step_size_;  // Step size for finite differences

    // Default step size: sqrt(machine epsilon) for double precision
    static constexpr double DEFAULT_STEP_SIZE = 1.5e-8;

    // Minimum step size to prevent numerical instability
    static constexpr double MIN_STEP_SIZE = 1.0e-12;
};

}  // namespace ag::estimation
