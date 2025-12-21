#include "ag/stats/PACF.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "ag/stats/ACF.hpp"

namespace ag::stats {

double pacf_at_lag(std::span<const double> data, std::size_t lag) {
    if (data.size() == 0) {
        throw std::invalid_argument("Cannot compute PACF of empty data");
    }

    if (lag == 0) {
        throw std::invalid_argument("PACF lag must be >= 1 (lag 0 is always 1)");
    }

    if (lag >= data.size()) {
        throw std::invalid_argument("Lag must be less than data size");
    }

    // Need to compute PACF up to the requested lag using Durbin-Levinson algorithm
    // This requires having all ACF values up to that lag
    std::vector<double> acf_values = acf(data, lag);

    // Durbin-Levinson recursion
    std::vector<double> phi(lag + 1, 0.0);  // phi coefficients
    std::vector<double> phi_new(lag + 1, 0.0);

    // Initialize for lag 1
    phi[1] = acf_values[1];

    // If we only need PACF at lag 1, return early
    if (lag == 1) {
        return phi[1];
    }

    // Recursively compute PACF for each lag
    for (std::size_t k = 2; k <= lag; ++k) {
        // Compute numerator for phi[k][k]
        double numerator = acf_values[k];
        for (std::size_t j = 1; j < k; ++j) {
            numerator -= phi[j] * acf_values[k - j];
        }

        // Compute denominator
        double denominator = 1.0;
        for (std::size_t j = 1; j < k; ++j) {
            denominator -= phi[j] * acf_values[j];
        }

        // Handle numerical issues
        if (std::abs(denominator) < 1e-10) {
            phi_new[k] = 0.0;
        } else {
            phi_new[k] = numerator / denominator;
        }

        // Update phi coefficients for this lag
        for (std::size_t j = 1; j < k; ++j) {
            phi_new[j] = phi[j] - phi_new[k] * phi[k - j];
        }

        // Copy new coefficients to phi
        for (std::size_t j = 1; j <= k; ++j) {
            phi[j] = phi_new[j];
        }
    }

    return phi[lag];
}

std::vector<double> pacf(std::span<const double> data, std::size_t max_lag) {
    if (data.size() == 0) {
        throw std::invalid_argument("Cannot compute PACF of empty data");
    }

    if (max_lag >= data.size()) {
        throw std::invalid_argument("max_lag must be less than data size");
    }

    if (max_lag == 0) {
        return std::vector<double>();  // Empty result for max_lag = 0
    }

    std::vector<double> result;
    result.reserve(max_lag);

    // Get all ACF values we need
    std::vector<double> acf_values = acf(data, max_lag);

    // Durbin-Levinson recursion
    std::vector<double> phi(max_lag + 1, 0.0);  // phi coefficients
    std::vector<double> phi_new(max_lag + 1, 0.0);

    // PACF at lag 1 equals ACF at lag 1
    phi[1] = acf_values[1];
    result.push_back(phi[1]);

    // Recursively compute PACF for each lag
    for (std::size_t k = 2; k <= max_lag; ++k) {
        // Compute numerator for phi[k][k]
        double numerator = acf_values[k];
        for (std::size_t j = 1; j < k; ++j) {
            numerator -= phi[j] * acf_values[k - j];
        }

        // Compute denominator
        double denominator = 1.0;
        for (std::size_t j = 1; j < k; ++j) {
            denominator -= phi[j] * acf_values[j];
        }

        // Handle numerical issues
        if (std::abs(denominator) < 1e-10) {
            phi_new[k] = 0.0;
        } else {
            phi_new[k] = numerator / denominator;
        }

        // Update phi coefficients for this lag
        for (std::size_t j = 1; j < k; ++j) {
            phi_new[j] = phi[j] - phi_new[k] * phi[k - j];
        }

        // Copy new coefficients to phi
        for (std::size_t j = 1; j <= k; ++j) {
            phi[j] = phi_new[j];
        }

        // Store the PACF value for this lag
        result.push_back(phi[k]);
    }

    return result;
}

}  // namespace ag::stats
