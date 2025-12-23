#include "ag/estimation/Optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace ag::estimation {

// ============================================================================
// NelderMeadOptimizer Implementation
// ============================================================================

NelderMeadOptimizer::NelderMeadOptimizer()
    : ftol_(DEFAULT_FTOL), xtol_(DEFAULT_XTOL), max_iterations_(DEFAULT_MAX_ITERATIONS) {}

NelderMeadOptimizer::NelderMeadOptimizer(double ftol, double xtol, int max_iterations)
    : ftol_(ftol), xtol_(xtol), max_iterations_(max_iterations) {
    if (ftol < 0.0) {
        throw std::invalid_argument("Function tolerance must be non-negative");
    }
    if (xtol < 0.0) {
        throw std::invalid_argument("Parameter tolerance must be non-negative");
    }
    if (max_iterations < 1) {
        throw std::invalid_argument("Maximum iterations must be at least 1");
    }
}

void NelderMeadOptimizer::setFunctionTolerance(double ftol) {
    if (ftol < 0.0) {
        throw std::invalid_argument("Function tolerance must be non-negative");
    }
    ftol_ = ftol;
}

void NelderMeadOptimizer::setParameterTolerance(double xtol) {
    if (xtol < 0.0) {
        throw std::invalid_argument("Parameter tolerance must be non-negative");
    }
    xtol_ = xtol;
}

void NelderMeadOptimizer::setMaxIterations(int max_iterations) {
    if (max_iterations < 1) {
        throw std::invalid_argument("Maximum iterations must be at least 1");
    }
    max_iterations_ = max_iterations;
}

std::vector<std::vector<double>>
NelderMeadOptimizer::initializeSimplex(const std::vector<double>& initial_params) const {
    const std::size_t n = initial_params.size();
    std::vector<std::vector<double>> simplex(n + 1, initial_params);

    // Create simplex by perturbing each coordinate
    // Use adaptive step based on parameter magnitude
    static constexpr double SIMPLEX_PERTURBATION_RATIO = 0.05;  // 5% of parameter value
    static constexpr double SIMPLEX_MIN_STEP_SIZE = 0.00025;    // Minimum step for small params

    for (std::size_t i = 0; i < n; ++i) {
        double step = std::abs(initial_params[i]) * SIMPLEX_PERTURBATION_RATIO;
        if (step < SIMPLEX_MIN_STEP_SIZE) {
            step = SIMPLEX_MIN_STEP_SIZE;
        }
        simplex[i + 1][i] += step;
    }

    return simplex;
}

std::vector<double>
NelderMeadOptimizer::computeCentroid(const std::vector<std::vector<double>>& simplex) const {
    const std::size_t n = simplex[0].size();
    const std::size_t n_vertices = simplex.size() - 1;  // Exclude worst point

    std::vector<double> centroid(n, 0.0);
    for (std::size_t i = 0; i < n_vertices; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            centroid[j] += simplex[i][j];
        }
    }

    for (std::size_t j = 0; j < n; ++j) {
        centroid[j] /= static_cast<double>(n_vertices);
    }

    return centroid;
}

bool NelderMeadOptimizer::hasConverged(const std::vector<double>& simplex_values,
                                       const std::vector<std::vector<double>>& simplex) const {
    // Check function value convergence
    const double f_range = simplex_values.back() - simplex_values.front();
    if (f_range < ftol_) {
        return true;
    }

    // Check parameter convergence
    const std::size_t n = simplex[0].size();
    const std::vector<double>& best = simplex[0];

    double max_param_diff = 0.0;
    for (const auto& vertex : simplex) {
        for (std::size_t i = 0; i < n; ++i) {
            max_param_diff = std::max(max_param_diff, std::abs(vertex[i] - best[i]));
        }
    }

    return max_param_diff < xtol_;
}

std::vector<double> NelderMeadOptimizer::reflect(const std::vector<double>& centroid,
                                                 const std::vector<double>& worst_point) const {
    const std::size_t n = centroid.size();
    std::vector<double> reflected(n);

    for (std::size_t i = 0; i < n; ++i) {
        reflected[i] = centroid[i] + ALPHA * (centroid[i] - worst_point[i]);
    }

    return reflected;
}

std::vector<double> NelderMeadOptimizer::expand(const std::vector<double>& centroid,
                                                const std::vector<double>& reflected_point) const {
    const std::size_t n = centroid.size();
    std::vector<double> expanded(n);

    for (std::size_t i = 0; i < n; ++i) {
        expanded[i] = centroid[i] + GAMMA * (reflected_point[i] - centroid[i]);
    }

    return expanded;
}

std::vector<double> NelderMeadOptimizer::contract(const std::vector<double>& centroid,
                                                  const std::vector<double>& worst_point) const {
    const std::size_t n = centroid.size();
    std::vector<double> contracted(n);

    for (std::size_t i = 0; i < n; ++i) {
        contracted[i] = centroid[i] + RHO * (worst_point[i] - centroid[i]);
    }

    return contracted;
}

void NelderMeadOptimizer::shrink(std::vector<std::vector<double>>& simplex,
                                 const std::vector<double>& best_point) const {
    const std::size_t n = best_point.size();

    // Shrink all vertices toward best point
    for (std::size_t i = 1; i < simplex.size(); ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            simplex[i][j] = best_point[j] + SIGMA * (simplex[i][j] - best_point[j]);
        }
    }
}

OptimizationResult NelderMeadOptimizer::minimize(const ObjectiveFunction& objective,
                                                 const std::vector<double>& initial_params) {
    if (initial_params.empty()) {
        throw std::invalid_argument("Initial parameter vector cannot be empty");
    }

    // Initialize simplex
    std::vector<std::vector<double>> simplex = initializeSimplex(initial_params);
    const std::size_t n_vertices = simplex.size();

    // Evaluate objective at all simplex vertices
    std::vector<double> simplex_values(n_vertices);
    for (std::size_t i = 0; i < n_vertices; ++i) {
        simplex_values[i] = objective(simplex[i]);
    }

    // Main optimization loop
    int iteration = 0;
    bool converged = false;

    while (iteration < max_iterations_) {
        // Sort simplex by objective value (best to worst)
        std::vector<std::size_t> indices(n_vertices);
        for (std::size_t i = 0; i < n_vertices; ++i) {
            indices[i] = i;
        }
        std::sort(indices.begin(), indices.end(), [&simplex_values](std::size_t a, std::size_t b) {
            return simplex_values[a] < simplex_values[b];
        });

        // Reorder simplex and values
        std::vector<std::vector<double>> sorted_simplex(n_vertices);
        std::vector<double> sorted_values(n_vertices);
        for (std::size_t i = 0; i < n_vertices; ++i) {
            sorted_simplex[i] = simplex[indices[i]];
            sorted_values[i] = simplex_values[indices[i]];
        }
        simplex = sorted_simplex;
        simplex_values = sorted_values;

        // Check convergence
        if (hasConverged(simplex_values, simplex)) {
            converged = true;
            break;
        }

        // Compute centroid of all points except worst
        std::vector<double> centroid = computeCentroid(simplex);

        // Try reflection
        std::vector<double> reflected = reflect(centroid, simplex.back());
        double f_reflected = objective(reflected);

        if (f_reflected < simplex_values[n_vertices - 2] && f_reflected >= simplex_values[0]) {
            // Reflected point is better than second worst but not better than best
            simplex.back() = reflected;
            simplex_values.back() = f_reflected;
        } else if (f_reflected < simplex_values[0]) {
            // Reflected point is best so far, try expansion
            std::vector<double> expanded = expand(centroid, reflected);
            double f_expanded = objective(expanded);

            if (f_expanded < f_reflected) {
                simplex.back() = expanded;
                simplex_values.back() = f_expanded;
            } else {
                simplex.back() = reflected;
                simplex_values.back() = f_reflected;
            }
        } else {
            // Reflected point is worse than second worst, try contraction
            std::vector<double> contracted = contract(centroid, simplex.back());
            double f_contracted = objective(contracted);

            if (f_contracted < simplex_values.back()) {
                simplex.back() = contracted;
                simplex_values.back() = f_contracted;
            } else {
                // Contraction failed, shrink simplex
                shrink(simplex, simplex[0]);

                // Re-evaluate all vertices except best
                for (std::size_t i = 1; i < n_vertices; ++i) {
                    simplex_values[i] = objective(simplex[i]);
                }
            }
        }

        ++iteration;
    }

    // Prepare result
    OptimizationResult result;
    result.parameters = simplex[0];
    result.objective_value = simplex_values[0];
    result.converged = converged;
    result.iterations = iteration;

    if (converged) {
        result.message = "Converged";
    } else {
        result.message = "Maximum iterations reached";
    }

    return result;
}

// ============================================================================
// Random Restart Optimization
// ============================================================================

OptimizationResultWithRestarts optimizeWithRestarts(IOptimizer& optimizer,
                                                    const IOptimizer::ObjectiveFunction& objective,
                                                    const std::vector<double>& initial_params,
                                                    int num_restarts, double perturbation_scale,
                                                    unsigned int seed) {
    if (initial_params.empty()) {
        throw std::invalid_argument("Initial parameters cannot be empty");
    }
    if (num_restarts < 0) {
        throw std::invalid_argument("Number of restarts must be non-negative");
    }
    if (perturbation_scale < 0.0) {
        throw std::invalid_argument("Perturbation scale must be non-negative");
    }

    // Initialize random number generator
    std::mt19937 rng;
    if (seed == 0) {
        std::random_device rd;
        rng.seed(rd());
    } else {
        rng.seed(seed);
    }

    // Run initial optimization
    OptimizationResult best_result = optimizer.minimize(objective, initial_params);
    OptimizationResultWithRestarts final_result(best_result);

    // Perform random restarts
    for (int i = 0; i < num_restarts; ++i) {
        // Generate perturbed starting point
        std::vector<double> perturbed_params = initial_params;
        std::normal_distribution<double> dist(0.0, 1.0);

        for (std::size_t j = 0; j < perturbed_params.size(); ++j) {
            double param_scale = std::max(std::abs(initial_params[j]), 0.01);
            double noise = perturbation_scale * param_scale * dist(rng);
            perturbed_params[j] += noise;
        }

        // Run optimization from perturbed starting point
        OptimizationResult restart_result = optimizer.minimize(objective, perturbed_params);

        final_result.restarts_performed++;

        // Keep the best result
        if (restart_result.converged &&
            restart_result.objective_value < best_result.objective_value) {
            best_result = restart_result;
            final_result.successful_restarts++;

            // Update final result with new best
            final_result.parameters = best_result.parameters;
            final_result.objective_value = best_result.objective_value;
            final_result.converged = best_result.converged;
            final_result.iterations = best_result.iterations;
            final_result.message = best_result.message;
        }
    }

    return final_result;
}

}  // namespace ag::estimation
