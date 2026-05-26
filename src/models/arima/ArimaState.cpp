#include "ag/models/arima/ArimaState.hpp"

#include "ag/util/Differencing.hpp"
#include "ag/util/SlidingWindow.hpp"

#include <stdexcept>

namespace ag::models::arima {

ArimaState::ArimaState(int p, int d, int q) : p_(p), d_(d), q_(q), initialized_(false) {
    if (p < 0 || d < 0 || q < 0) {
        throw std::invalid_argument("ARIMA orders must be non-negative");
    }

    // Pre-allocate buffers for historical values
    if (p > 0) {
        obs_history_.reserve(p);
    }
    if (q > 0) {
        residual_history_.reserve(q);
    }
}

void ArimaState::initialize(const double* data, std::size_t size) {
    if (data == nullptr) {
        throw std::invalid_argument("Data pointer cannot be null");
    }
    if (size == 0) {
        throw std::invalid_argument("Data size must be positive");
    }

    // Apply differencing if d > 0
    if (d_ > 0) {
        differenced_series_ = applyDifferencing(data, size);
        if (differenced_series_.size() < static_cast<std::size_t>(p_)) {
            throw std::runtime_error(
                "Insufficient data after differencing: need at least p observations");
        }
    } else {
        differenced_series_.clear();
    }

    // Initialize observation history with zeros (will be filled during recursion)
    obs_history_.clear();
    obs_history_.resize(p_, 0.0);

    // Initialize residual history with zeros
    residual_history_.clear();
    residual_history_.resize(q_, 0.0);

    initialized_ = true;
}

void ArimaState::update(double observation, double residual) {
    ag::util::shiftAndAppend(obs_history_, observation);
    ag::util::shiftAndAppend(residual_history_, residual);
}

std::vector<double> ArimaState::applyDifferencing(const double* data, std::size_t size) const {
    return ag::util::differenceSeries(data, size, d_);
}

}  // namespace ag::models::arima
