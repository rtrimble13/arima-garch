#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace ag::estimation {

/**
 * @brief Result of an optimization run.
 *
 * OptimizationResult contains the optimal parameters found by the optimizer,
 * the final objective function value, and diagnostic information about
 * convergence and iteration count.
 */
struct OptimizationResult {
    std::vector<double> parameters;  // Optimal parameters found
    double objective_value;          // Final objective function value
    bool converged;                  // Whether the optimizer converged
    int iterations;                  // Number of iterations performed
    std::string message;             // Status message (e.g., "Converged", "Max iterations")
};

/**
 * @brief Abstract interface for optimization algorithms.
 *
 * IOptimizer defines the interface that all optimization algorithms must implement.
 * It provides a common API for minimizing objective functions, allowing different
 * optimization strategies to be used interchangeably.
 *
 * Implementations should support:
 * - Minimization of scalar-valued functions
 * - Convergence criteria (tolerance, max iterations)
 * - Diagnostic output (iterations, convergence status)
 */
class IOptimizer {
public:
    /**
     * @brief Objective function type: takes a parameter vector and returns a scalar value.
     *
     * The function signature is: double f(const std::vector<double>& params)
     * where params is the point at which to evaluate the objective function.
     */
    using ObjectiveFunction = std::function<double(const std::vector<double>&)>;

    /**
     * @brief Virtual destructor for proper cleanup of derived classes.
     */
    virtual ~IOptimizer() = default;

    /**
     * @brief Minimize the objective function starting from initial parameters.
     *
     * This method performs iterative optimization to find parameters that minimize
     * the objective function. The optimization continues until convergence criteria
     * are met or the maximum number of iterations is reached.
     *
     * @param objective The objective function to minimize
     * @param initial_params Starting point for optimization
     * @return OptimizationResult containing optimal parameters and diagnostic info
     * @throws std::invalid_argument if initial_params is empty or invalid
     */
    [[nodiscard]] virtual OptimizationResult
    minimize(const ObjectiveFunction& objective, const std::vector<double>& initial_params) = 0;
};

/**
 * @brief Nelder-Mead simplex optimizer (derivative-free).
 *
 * NelderMeadOptimizer implements the Nelder-Mead downhill simplex method,
 * a derivative-free optimization algorithm. This makes it particularly suitable
 * for ARIMA-GARCH likelihood optimization where gradients may be expensive or
 * unavailable.
 *
 * The algorithm maintains a simplex of n+1 points in n-dimensional space and
 * iteratively transforms the simplex through reflection, expansion, contraction,
 * and shrinkage operations to move toward the optimum.
 *
 * Algorithm characteristics:
 * - No gradient computation required
 * - Robust to noisy functions
 * - Moderate convergence speed (slower than gradient-based methods)
 * - Well-suited for low to moderate dimensional problems (n < 20)
 *
 * Convergence criteria:
 * - Function tolerance: |f_best - f_worst| < ftol
 * - Parameter tolerance: max(|x_i - x_best_i|) < xtol for all simplex points
 * - Maximum iterations: iterations >= max_iterations
 *
 * References:
 * - Nelder, J. A., & Mead, R. (1965). A simplex method for function minimization.
 *   The Computer Journal, 7(4), 308-313.
 * - Press, W. H., et al. (2007). Numerical Recipes: The Art of Scientific Computing
 *   (3rd ed.). Cambridge University Press.
 */
class NelderMeadOptimizer : public IOptimizer {
public:
    /**
     * @brief Default constructor with standard convergence criteria.
     *
     * Uses default tolerances suitable for most optimization problems:
     * - ftol = 1e-8 (function value tolerance)
     * - xtol = 1e-8 (parameter tolerance)
     * - max_iterations = 1000
     */
    NelderMeadOptimizer();

    /**
     * @brief Construct with custom convergence criteria.
     *
     * @param ftol Function value tolerance for convergence
     * @param xtol Parameter tolerance for convergence
     * @param max_iterations Maximum number of iterations
     * @throws std::invalid_argument if tolerances are negative or max_iterations < 1
     */
    NelderMeadOptimizer(double ftol, double xtol, int max_iterations);

    /**
     * @brief Minimize the objective function using Nelder-Mead algorithm.
     *
     * Implements the Nelder-Mead downhill simplex method:
     * 1. Initialize simplex of n+1 points around initial_params
     * 2. Sort simplex vertices by objective function value
     * 3. Attempt reflection, expansion, contraction, or shrinkage
     * 4. Check convergence criteria
     * 5. Repeat until convergence or max iterations
     *
     * @param objective The objective function to minimize
     * @param initial_params Starting point for optimization
     * @return OptimizationResult with optimal parameters and diagnostics
     * @throws std::invalid_argument if initial_params is empty
     */
    [[nodiscard]] OptimizationResult minimize(const ObjectiveFunction& objective,
                                              const std::vector<double>& initial_params) override;

    /**
     * @brief Get function value tolerance.
     * @return Current function tolerance (ftol)
     */
    [[nodiscard]] double getFunctionTolerance() const noexcept { return ftol_; }

    /**
     * @brief Get parameter tolerance.
     * @return Current parameter tolerance (xtol)
     */
    [[nodiscard]] double getParameterTolerance() const noexcept { return xtol_; }

    /**
     * @brief Get maximum iterations.
     * @return Maximum number of iterations allowed
     */
    [[nodiscard]] int getMaxIterations() const noexcept { return max_iterations_; }

    /**
     * @brief Set function value tolerance.
     * @param ftol New function tolerance (must be non-negative)
     * @throws std::invalid_argument if ftol < 0
     */
    void setFunctionTolerance(double ftol);

    /**
     * @brief Set parameter tolerance.
     * @param xtol New parameter tolerance (must be non-negative)
     * @throws std::invalid_argument if xtol < 0
     */
    void setParameterTolerance(double xtol);

    /**
     * @brief Set maximum iterations.
     * @param max_iterations New maximum iterations (must be positive)
     * @throws std::invalid_argument if max_iterations < 1
     */
    void setMaxIterations(int max_iterations);

private:
    double ftol_;         // Function value tolerance for convergence
    double xtol_;         // Parameter tolerance for convergence
    int max_iterations_;  // Maximum number of iterations

    // Default values for convergence criteria
    static constexpr double DEFAULT_FTOL = 1e-8;
    static constexpr double DEFAULT_XTOL = 1e-8;
    static constexpr int DEFAULT_MAX_ITERATIONS = 1000;

    // Nelder-Mead algorithm coefficients
    static constexpr double ALPHA = 1.0;  // Reflection coefficient
    static constexpr double GAMMA = 2.0;  // Expansion coefficient
    static constexpr double RHO = 0.5;    // Contraction coefficient
    static constexpr double SIGMA = 0.5;  // Shrinkage coefficient

    /**
     * @brief Initialize simplex around starting point.
     * @param initial_params Starting parameters
     * @return Simplex as vector of n+1 parameter vectors
     */
    std::vector<std::vector<double>>
    initializeSimplex(const std::vector<double>& initial_params) const;

    /**
     * @brief Compute centroid of all simplex points except the worst.
     * @param simplex Current simplex
     * @return Centroid coordinates
     */
    std::vector<double> computeCentroid(const std::vector<std::vector<double>>& simplex) const;

    /**
     * @brief Check if simplex has converged.
     * @param simplex_values Objective values at simplex vertices (sorted)
     * @param simplex Simplex vertices (sorted by objective value)
     * @return true if convergence criteria are met
     */
    bool hasConverged(const std::vector<double>& simplex_values,
                      const std::vector<std::vector<double>>& simplex) const;

    /**
     * @brief Compute reflected point: x_r = centroid + alpha * (centroid - x_worst)
     * @param centroid Centroid of best n points
     * @param worst_point Worst point in simplex
     * @return Reflected point
     */
    std::vector<double> reflect(const std::vector<double>& centroid,
                                const std::vector<double>& worst_point) const;

    /**
     * @brief Compute expanded point: x_e = centroid + gamma * (x_r - centroid)
     * @param centroid Centroid of best n points
     * @param reflected_point Reflected point
     * @return Expanded point
     */
    std::vector<double> expand(const std::vector<double>& centroid,
                               const std::vector<double>& reflected_point) const;

    /**
     * @brief Compute contracted point: x_c = centroid + rho * (worst_point - centroid)
     * @param centroid Centroid of best n points
     * @param worst_point Worst point in simplex
     * @return Contracted point
     */
    std::vector<double> contract(const std::vector<double>& centroid,
                                 const std::vector<double>& worst_point) const;

    /**
     * @brief Shrink simplex toward best point.
     * @param simplex Current simplex (modified in place)
     * @param best_point Best point in simplex
     */
    void shrink(std::vector<std::vector<double>>& simplex,
                const std::vector<double>& best_point) const;
};

}  // namespace ag::estimation
