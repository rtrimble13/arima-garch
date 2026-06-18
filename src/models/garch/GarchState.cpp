#include "ag/models/garch/GarchState.hpp"

#include "ag/util/NumericConstants.hpp"
#include "ag/util/SlidingWindow.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ag::models::garch {

GarchState::GarchState(int p, int q) : p_(p), q_(q), initialized_(false), initial_variance_(0.0) {
    if (p < 0 || q < 0) {
        throw std::invalid_argument("GARCH orders p and q must be >= 0");
    }

    // Both must be 0 (ARIMA-only) or both must be >= 1 (GARCH model)
    if ((p == 0) != (q == 0)) {
        throw std::invalid_argument(
            "GARCH orders must both be 0 (ARIMA-only) or both be >= 1, got p=" + std::to_string(p) +
            ", q=" + std::to_string(q));
    }

    // Pre-allocate buffers for historical values (skip if both are 0)
    if (p > 0 && q > 0) {
        variance_history_.reserve(p);
        squared_residual_history_.reserve(q);
    }
}

void GarchState::initialize(const double* residuals, std::size_t size,
                            double unconditional_variance) {
    if (residuals == nullptr) {
        throw std::invalid_argument("Residuals pointer cannot be null");
    }
    if (size == 0) {
        throw std::invalid_argument("Residuals size must be positive");
    }

    // Determine initial variance (h_0). Use the provided unconditional variance
    // when it is a usable finite positive value; otherwise (non-stationary —
    // signalled by NaN — or unspecified) fall back to the sample variance. The
    // isfinite check, rather than a bare > 0.0, avoids relying on a 0.0 magic
    // value and cleanly handles the NaN returned at the stationarity boundary.
    if (std::isfinite(unconditional_variance) && unconditional_variance > 0.0) {
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
    ag::util::shiftAndAppend(variance_history_, conditional_variance);
    ag::util::shiftAndAppend(squared_residual_history_, squared_residual);
}

void GarchState::restore(const std::vector<double>& variance_history,
                         const std::vector<double>& squared_residual_history,
                         double initial_variance) {
    if (variance_history.size() != static_cast<std::size_t>(p_)) {
        throw std::invalid_argument(
            "GarchState::restore: variance history size must equal GARCH order p");
    }
    if (squared_residual_history.size() != static_cast<std::size_t>(q_)) {
        throw std::invalid_argument(
            "GarchState::restore: squared residual history size must equal ARCH order q");
    }

    initial_variance_ = initial_variance;
    variance_history_ = variance_history;
    squared_residual_history_ = squared_residual_history;
    initialized_ = true;
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
    return std::max(variance, ag::util::MIN_VARIANCE);
}

}  // namespace ag::models::garch
