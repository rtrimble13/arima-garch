#include "ag/estimation/NumericalDerivatives.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ag::estimation {

NumericalDerivatives::NumericalDerivatives() : step_size_(DEFAULT_STEP_SIZE) {}

NumericalDerivatives::NumericalDerivatives(double step_size) : step_size_(step_size) {
    if (step_size <= 0.0) {
        throw std::invalid_argument("Step size must be positive");
    }
    if (step_size < MIN_STEP_SIZE) {
        throw std::invalid_argument("Step size too small, may cause numerical instability");
    }
}

void NumericalDerivatives::setStepSize(double step_size) {
    if (step_size <= 0.0) {
        throw std::invalid_argument("Step size must be positive");
    }
    if (step_size < MIN_STEP_SIZE) {
        throw std::invalid_argument("Step size too small, may cause numerical instability");
    }
    step_size_ = step_size;
}

std::vector<double> NumericalDerivatives::computeGradient(const ObjectiveFunction& objective,
                                                          const std::vector<double>& params) const {
    if (params.empty()) {
        throw std::invalid_argument("Parameter vector cannot be empty");
    }

    const std::size_t n = params.size();
    std::vector<double> gradient(n);

    // Create working copy of parameters
    std::vector<double> params_plus = params;
    std::vector<double> params_minus = params;

    // Compute gradient using central differences
    for (std::size_t i = 0; i < n; ++i) {
        // Perturb the i-th parameter
        const double h = step_size_;
        params_plus[i] = params[i] + h;
        params_minus[i] = params[i] - h;

        // Evaluate function at perturbed points
        const double f_plus = objective(params_plus);
        const double f_minus = objective(params_minus);

        // Central difference approximation
        gradient[i] = (f_plus - f_minus) / (2.0 * h);

        // Restore original value
        params_plus[i] = params[i];
        params_minus[i] = params[i];
    }

    return gradient;
}

std::vector<double>
NumericalDerivatives::computeGradientAdaptive(const ObjectiveFunction& objective,
                                              const std::vector<double>& params) const {
    if (params.empty()) {
        throw std::invalid_argument("Parameter vector cannot be empty");
    }

    const std::size_t n = params.size();
    std::vector<double> gradient(n);

    // Create working copy of parameters
    std::vector<double> params_plus = params;
    std::vector<double> params_minus = params;

    // Compute gradient using central differences with adaptive step size
    for (std::size_t i = 0; i < n; ++i) {
        // Adaptive step size based on parameter magnitude
        // h_i = base_step * max(|x_i|, 1.0)
        const double param_scale = std::max(std::abs(params[i]), 1.0);
        const double h = step_size_ * param_scale;

        // Perturb the i-th parameter
        params_plus[i] = params[i] + h;
        params_minus[i] = params[i] - h;

        // Evaluate function at perturbed points
        const double f_plus = objective(params_plus);
        const double f_minus = objective(params_minus);

        // Central difference approximation
        gradient[i] = (f_plus - f_minus) / (2.0 * h);

        // Restore original value
        params_plus[i] = params[i];
        params_minus[i] = params[i];
    }

    return gradient;
}

}  // namespace ag::estimation
