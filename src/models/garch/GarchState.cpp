#include "ag/models/garch/GarchState.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ag::models::garch {

GarchState::GarchState(int p, int q) : p_(p), q_(q), initialized_(false), initial_variance_(0.0) {
    if (p < 1 || q < 1) {
        throw std::invalid_argument("GARCH orders p and q must be >= 1");
    }

    // Pre-allocate buffers for historical values
    variance_history_.reserve(p);
    squared_residual_history_.reserve(q);
}

void GarchState::initialize(const double* residuals, std::size_t size,
                            double unconditional_variance) {
    if (residuals == nullptr) {
        throw std::invalid_argument("Residuals pointer cannot be null");
    }
    if (size == 0) {
        throw std::invalid_argument("Residuals size must be positive");
    }

    // Determine initial variance (h_0)
    if (unconditional_variance > 0.0) {
        // Use provided unconditional variance (when stationary)
        initial_variance_ = unconditional_variance;
    } else {
        // Use sample variance of residuals as fallback
        initial_variance_ = computeSampleVariance(residuals, size);
    }

    // Initialize variance history with h_0
    variance_history_.clear();
    variance_history_.resize(p_, initial_variance_);

    // Initialize squared residual history with zeros
    // (will be filled during recursion)
    squared_residual_history_.clear();
    squared_residual_history_.resize(q_, 0.0);

    initialized_ = true;
}

void GarchState::update(double conditional_variance, double squared_residual) {
    // Update variance history (shift left and add new)
    std::shift_left(variance_history_.begin(), variance_history_.end(), 1);
    variance_history_.back() = conditional_variance;

    // Update squared residual history (shift left and add new)
    std::shift_left(squared_residual_history_.begin(), squared_residual_history_.end(), 1);
    squared_residual_history_.back() = squared_residual;
}

double GarchState::computeSampleVariance(const double* residuals, std::size_t size) const {
    if (size < 2) {
        // Return a small positive value if insufficient data
        return 1.0;
    }

    // Compute mean
    double mean = std::accumulate(residuals, residuals + size, 0.0) / static_cast<double>(size);

    // Compute variance: Σ(x_i - mean)² / (n - 1)
    double sum_sq_diff = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        double diff = residuals[i] - mean;
        sum_sq_diff += diff * diff;
    }

    double variance = sum_sq_diff / static_cast<double>(size - 1);

    // Ensure variance is positive (handle numerical issues)
    return std::max(variance, 1e-10);
}

}  // namespace ag::models::garch
