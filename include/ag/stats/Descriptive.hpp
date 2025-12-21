#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace ag::stats {

/**
 * @brief Calculate the arithmetic mean of a data sequence.
 * @param data Span of data values
 * @return The arithmetic mean
 * @throws std::invalid_argument if data is empty
 */
[[nodiscard]] double mean(std::span<const double> data);

/**
 * @brief Calculate the sample variance of a data sequence.
 * 
 * Uses Bessel's correction (dividing by n-1) for unbiased estimation.
 * 
 * @param data Span of data values
 * @return The sample variance
 * @throws std::invalid_argument if data has fewer than 2 elements
 */
[[nodiscard]] double variance(std::span<const double> data);

/**
 * @brief Calculate the skewness of a data sequence.
 * 
 * Skewness measures the asymmetry of the distribution.
 * Uses the adjusted Fisher-Pearson standardized moment coefficient (G1).
 * 
 * @param data Span of data values
 * @return The skewness coefficient
 * @throws std::invalid_argument if data has fewer than 3 elements
 */
[[nodiscard]] double skewness(std::span<const double> data);

/**
 * @brief Calculate the excess kurtosis of a data sequence.
 * 
 * Kurtosis measures the "tailedness" of the distribution.
 * Returns excess kurtosis (kurtosis - 3), where normal distribution has excess kurtosis of 0.
 * Uses the adjusted estimator for sample kurtosis.
 * 
 * @param data Span of data values
 * @return The excess kurtosis coefficient
 * @throws std::invalid_argument if data has fewer than 4 elements
 */
[[nodiscard]] double kurtosis(std::span<const double> data);

// Convenience overloads for std::vector
[[nodiscard]] inline double mean(const std::vector<double>& data) {
    return mean(std::span<const double>(data));
}

[[nodiscard]] inline double variance(const std::vector<double>& data) {
    return variance(std::span<const double>(data));
}

[[nodiscard]] inline double skewness(const std::vector<double>& data) {
    return skewness(std::span<const double>(data));
}

[[nodiscard]] inline double kurtosis(const std::vector<double>& data) {
    return kurtosis(std::span<const double>(data));
}

}  // namespace ag::stats
