#include "ag/stats/ACF.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ag::stats {

double acf_at_lag(std::span<const double> data, std::size_t lag) {
    const std::size_t n = data.size();

    if (n == 0) {
        throw std::invalid_argument("Cannot compute ACF of empty data");
    }

    if (lag >= n) {
        throw std::invalid_argument("Lag must be less than data size");
    }

    // Special case: ACF at lag 0 is always 1
    if (lag == 0) {
        return 1.0;
    }

    // Compute mean
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / static_cast<double>(n);

    // Compute variance (denominator) using all data points
    double variance = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double diff = data[i] - mean;
        variance += diff * diff;
    }

    // Handle constant series
    if (variance == 0.0) {
        return 0.0;
    }

    // Compute autocovariance at lag k (numerator)
    double autocovariance = 0.0;
    for (std::size_t i = 0; i < n - lag; ++i) {
        autocovariance += (data[i] - mean) * (data[i + lag] - mean);
    }

    // ACF = autocovariance / variance
    return autocovariance / variance;
}

std::vector<double> acf(std::span<const double> data, std::size_t max_lag) {
    const std::size_t n = data.size();

    if (n == 0) {
        throw std::invalid_argument("Cannot compute ACF of empty data");
    }

    if (max_lag >= n) {
        throw std::invalid_argument("max_lag must be less than data size");
    }

    std::vector<double> result;
    result.reserve(max_lag + 1);

    // Compute mean once
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / static_cast<double>(n);

    // Compute variance once (denominator for all lags)
    double variance = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double diff = data[i] - mean;
        variance += diff * diff;
    }

    // Handle constant series - all ACF values are 0 except lag 0
    if (variance == 0.0) {
        result.push_back(1.0);  // ACF at lag 0
        for (std::size_t lag = 1; lag <= max_lag; ++lag) {
            result.push_back(0.0);
        }
        return result;
    }

    // Lag 0: ACF is always 1.0
    result.push_back(1.0);

    // Compute ACF for each lag
    for (std::size_t lag = 1; lag <= max_lag; ++lag) {
        double autocovariance = 0.0;
        for (std::size_t i = 0; i < n - lag; ++i) {
            autocovariance += (data[i] - mean) * (data[i + lag] - mean);
        }
        result.push_back(autocovariance / variance);
    }

    return result;
}

}  // namespace ag::stats
