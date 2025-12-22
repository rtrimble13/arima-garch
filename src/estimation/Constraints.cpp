#include "ag/estimation/Constraints.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ag::estimation {

ParameterVector ArimaGarchTransform::toConstrained(const ParameterVector& theta, int p, int q) {
    // Validate input size
    const std::size_t expected_size = static_cast<std::size_t>(1 + p + q);
    if (theta.size() != expected_size) {
        throw std::invalid_argument("theta size mismatch: expected " +
                                    std::to_string(expected_size) + ", got " +
                                    std::to_string(theta.size()));
    }

    if (p < 1 || q < 1) {
        throw std::invalid_argument("p and q must be at least 1");
    }

    // Create output vector
    ParameterVector params(expected_size, 0.0);

    // Transform omega: must be positive
    // omega = exp(theta[0])
    params[0] = std::exp(theta[0]);

    // Transform ARCH and GARCH coefficients to ensure:
    // 1. All coefficients >= 0
    // 2. sum(alpha) + sum(beta) < MAX_PERSISTENCE
    //
    // Strategy: Apply exp to get positive values, then scale with logistic function
    // to ensure sum constraint

    // First, transform to positive values
    std::vector<double> exp_values(p + q);
    for (int i = 0; i < p + q; ++i) {
        exp_values[i] = std::exp(theta[1 + i]);
    }

    // Calculate sum of exponentials
    double sum_exp = std::accumulate(exp_values.begin(), exp_values.end(), 0.0);

    // Scale to ensure sum < MAX_PERSISTENCE using a logistic-like transform
    // scaled_value_i = MAX_PERSISTENCE * exp(theta_i) / (1 + sum(exp(theta_j)))
    // This ensures: sum(scaled_values) = MAX_PERSISTENCE * sum_exp / (1 + sum_exp) <
    // MAX_PERSISTENCE
    double scale_factor = MAX_PERSISTENCE / (1.0 + sum_exp);

    // Apply scaling to ARCH coefficients (alpha)
    for (int i = 0; i < p; ++i) {
        params[1 + i] = exp_values[i] * scale_factor;
    }

    // Apply scaling to GARCH coefficients (beta)
    for (int j = 0; j < q; ++j) {
        params[1 + p + j] = exp_values[p + j] * scale_factor;
    }

    return params;
}

ParameterVector ArimaGarchTransform::toUnconstrained(const ParameterVector& params, int p, int q) {
    // Validate input size
    const std::size_t expected_size = static_cast<std::size_t>(1 + p + q);
    if (params.size() != expected_size) {
        throw std::invalid_argument("params size mismatch: expected " +
                                    std::to_string(expected_size) + ", got " +
                                    std::to_string(params.size()));
    }

    if (p < 1 || q < 1) {
        throw std::invalid_argument("p and q must be at least 1");
    }

    // Validate constraints
    if (!validateConstraints(params, p, q)) {
        throw std::invalid_argument("Parameters violate GARCH constraints");
    }

    // Create output vector
    ParameterVector theta(expected_size, 0.0);

    // Inverse transform for omega: theta[0] = log(omega)
    theta[0] = std::log(params[0]);

    // For the coefficients, we need to invert the transformation:
    // params[i] = exp(theta[i]) * scale_factor
    // where scale_factor = MAX_PERSISTENCE / (1 + sum_exp)
    // and sum_exp = sum(exp(theta[j]))
    //
    // This is not directly invertible, so we use an approximation:
    // Given that sum(params[1:]) < MAX_PERSISTENCE, we can approximate:
    // sum_exp = MAX_PERSISTENCE / scale_factor - 1
    // But scale_factor = sum(params[1:]) * (1 + sum_exp) / MAX_PERSISTENCE
    //
    // More directly: if params[i] = exp(theta[i]) * k, then theta[i] = log(params[i] / k)
    // We can estimate k from the constraint

    // Calculate sum of ARCH and GARCH coefficients
    double sum_coeffs = 0.0;
    for (int i = 0; i < p + q; ++i) {
        sum_coeffs += params[1 + i];
    }

    // Estimate the scale factor used
    // From the forward transform: sum_coeffs ≈ MAX_PERSISTENCE * sum_exp / (1 + sum_exp)
    // Solving for sum_exp: sum_exp ≈ sum_coeffs / (MAX_PERSISTENCE - sum_coeffs)
    // Note: EPSILON is added to the denominator to prevent division by zero when
    // sum_coeffs is very close to MAX_PERSISTENCE (which should not happen for valid
    // parameters, but provides numerical stability for edge cases)
    double sum_exp_estimate = sum_coeffs / (MAX_PERSISTENCE - sum_coeffs + EPSILON);
    double scale_factor_estimate = MAX_PERSISTENCE / (1.0 + sum_exp_estimate);

    // Invert the transform for each coefficient
    for (int i = 0; i < p + q; ++i) {
        // params[1+i] = exp(theta[1+i]) * scale_factor
        // theta[1+i] = log(params[1+i] / scale_factor)
        double value = params[1 + i] / (scale_factor_estimate + EPSILON);
        theta[1 + i] = std::log(std::max(value, EPSILON));  // Clamp to avoid log(0)
    }

    return theta;
}

bool ArimaGarchTransform::validateConstraints(const ParameterVector& params, int p,
                                              int q) noexcept {
    // Check size
    const std::size_t expected_size = static_cast<std::size_t>(1 + p + q);
    if (params.size() != expected_size) {
        return false;
    }

    if (p < 1 || q < 1) {
        return false;
    }

    // Check omega > 0
    if (params[0] <= 0.0) {
        return false;
    }

    // Check all ARCH coefficients >= 0
    for (int i = 0; i < p; ++i) {
        if (params[1 + i] < 0.0) {
            return false;
        }
    }

    // Check all GARCH coefficients >= 0
    for (int j = 0; j < q; ++j) {
        if (params[1 + p + j] < 0.0) {
            return false;
        }
    }

    // Check stationarity: sum(alpha) + sum(beta) < 1
    double sum_alpha = 0.0;
    for (int i = 0; i < p; ++i) {
        sum_alpha += params[1 + i];
    }

    double sum_beta = 0.0;
    for (int j = 0; j < q; ++j) {
        sum_beta += params[1 + p + j];
    }

    if (sum_alpha + sum_beta >= 1.0) {
        return false;
    }

    return true;
}

}  // namespace ag::estimation
