#include "ag/models/garch/GarchModel.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ag::models::garch {

// ============================================================================
// GarchParameters Implementation
// ============================================================================

bool GarchParameters::isPositive() const noexcept {
    // Check omega > 0
    if (omega <= 0.0) {
        return false;
    }

    // Check all alpha >= 0
    for (double alpha : alpha_coef) {
        if (alpha < 0.0) {
            return false;
        }
    }

    // Check all beta >= 0
    for (double beta : beta_coef) {
        if (beta < 0.0) {
            return false;
        }
    }

    return true;
}

bool GarchParameters::isStationary() const noexcept {
    // Sum of all coefficients must be < 1 for stationarity
    double sum_alpha = std::accumulate(alpha_coef.begin(), alpha_coef.end(), 0.0);
    double sum_beta = std::accumulate(beta_coef.begin(), beta_coef.end(), 0.0);
    return (sum_alpha + sum_beta) < 1.0;
}

double GarchParameters::unconditionalVariance() const noexcept {
    if (!isStationary()) {
        return 0.0;
    }

    double sum_alpha = std::accumulate(alpha_coef.begin(), alpha_coef.end(), 0.0);
    double sum_beta = std::accumulate(beta_coef.begin(), beta_coef.end(), 0.0);
    double denominator = 1.0 - sum_alpha - sum_beta;

    // Guard against division by zero (though stationarity check should prevent this)
    if (denominator <= 0.0) {
        return 0.0;
    }

    return omega / denominator;
}

// ============================================================================
// GarchModel Implementation
// ============================================================================

GarchModel::GarchModel(const ag::models::GarchSpec& spec) : spec_(spec) {
    spec.validate();
}

std::vector<double> GarchModel::computeConditionalVariances(const double* residuals,
                                                            std::size_t size,
                                                            const GarchParameters& params) const {
    if (residuals == nullptr) {
        throw std::invalid_argument("Residuals pointer cannot be null");
    }
    if (size == 0) {
        throw std::invalid_argument("Residuals size must be positive");
    }

    // Validate parameter dimensions
    if (static_cast<int>(params.alpha_coef.size()) != spec_.q) {
        throw std::invalid_argument("ARCH coefficient count must match q");
    }
    if (static_cast<int>(params.beta_coef.size()) != spec_.p) {
        throw std::invalid_argument("GARCH coefficient count must match p");
    }

    // Validate parameter constraints
    if (!params.isPositive()) {
        throw std::invalid_argument(
            "GARCH parameters must satisfy positivity constraints: omega > 0, alpha >= 0, beta >= "
            "0");
    }

    // Initialize state
    GarchState state(spec_.p, spec_.q);

    // Use unconditional variance if stationary, otherwise use sample variance
    double init_variance = params.unconditionalVariance();
    state.initialize(residuals, size, init_variance);

    // Reserve space for conditional variances
    std::vector<double> conditional_variances;
    conditional_variances.reserve(size);

    // Compute conditional variances using recursion
    for (std::size_t t = 0; t < size; ++t) {
        // Compute conditional variance for this observation
        double h_t = computeConditionalVariance(state, params);

        // Ensure h_t > 0 (should be guaranteed by positive parameters, but guard against
        // numerical issues)
        h_t = std::max(h_t, 1e-10);

        conditional_variances.push_back(h_t);

        // Compute squared residual
        double eps_squared = residuals[t] * residuals[t];

        // Update state for next iteration
        state.update(h_t, eps_squared);
    }

    return conditional_variances;
}

double GarchModel::computeConditionalVariance(const GarchState& state,
                                              const GarchParameters& params) const {
    double h_t = params.omega;

    // Add ARCH component: α₁*ε²_{t-1} + α₂*ε²_{t-2} + ... + α_q*ε²_{t-q}
    const auto& sq_res_history = state.getSquaredResidualHistory();
    for (int i = 0; i < spec_.q; ++i) {
        // History is stored with oldest first, so sq_res_history[0] is ε²_{t-q}
        // and sq_res_history[q-1] is ε²_{t-1}
        h_t += params.alpha_coef[i] * sq_res_history[spec_.q - 1 - i];
    }

    // Add GARCH component: β₁*h_{t-1} + β₂*h_{t-2} + ... + βₚ*h_{t-p}
    const auto& var_history = state.getVarianceHistory();
    for (int i = 0; i < spec_.p; ++i) {
        // History is stored with oldest first, so var_history[0] is h_{t-p}
        // and var_history[p-1] is h_{t-1}
        h_t += params.beta_coef[i] * var_history[spec_.p - 1 - i];
    }

    return h_t;
}

}  // namespace ag::models::garch
