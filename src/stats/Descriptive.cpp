#include "ag/stats/Descriptive.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ag::stats {

double mean(std::span<const double> data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot compute mean of empty data");
    }

    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / static_cast<double>(data.size());
}

double variance(std::span<const double> data) {
    if (data.size() < 2) {
        throw std::invalid_argument("Cannot compute variance with fewer than 2 data points");
    }

    double m = mean(data);
    double sum_squared_diff = 0.0;

    for (double value : data) {
        double diff = value - m;
        sum_squared_diff += diff * diff;
    }

    // Use Bessel's correction (n-1) for unbiased sample variance
    return sum_squared_diff / static_cast<double>(data.size() - 1);
}

double skewness(std::span<const double> data) {
    if (data.size() < 3) {
        throw std::invalid_argument("Cannot compute skewness with fewer than 3 data points");
    }

    double m = mean(data);
    double n = static_cast<double>(data.size());

    double m2 = 0.0;  // Second central moment
    double m3 = 0.0;  // Third central moment

    for (double value : data) {
        double diff = value - m;
        double diff2 = diff * diff;
        m2 += diff2;
        m3 += diff2 * diff;
    }

    m2 /= n;
    m3 /= n;

    // Sample skewness with adjustment factor
    double std_dev = std::sqrt(m2);
    if (std_dev == 0.0) {
        return 0.0;  // All values are the same
    }

    double g1 = m3 / (std_dev * std_dev * std_dev);

    // Apply adjustment for sample skewness (Fisher's correction)
    double adjustment = std::sqrt(n * (n - 1.0)) / (n - 2.0);
    return g1 * adjustment;
}

double kurtosis(std::span<const double> data) {
    if (data.size() < 4) {
        throw std::invalid_argument("Cannot compute kurtosis with fewer than 4 data points");
    }

    double m = mean(data);
    double n = static_cast<double>(data.size());

    double m2 = 0.0;  // Second central moment
    double m4 = 0.0;  // Fourth central moment

    for (double value : data) {
        double diff = value - m;
        double diff2 = diff * diff;
        m2 += diff2;
        m4 += diff2 * diff2;
    }

    m2 /= n;
    m4 /= n;

    if (m2 == 0.0) {
        return 0.0;  // All values are the same
    }

    // Sample kurtosis
    double kurt = m4 / (m2 * m2);

    // Apply adjustment for sample kurtosis and subtract 3 for excess kurtosis
    double adjustment =
        ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * kurt - 3.0 * (n - 1.0));

    return adjustment;
}

}  // namespace ag::stats
