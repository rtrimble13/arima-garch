#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace ag::stats {

/**
 * @brief Result of the Jarque-Bera test for normality.
 *
 * The Jarque-Bera test is used to test whether sample data has the skewness
 * and kurtosis matching a normal distribution.
 */
struct JarqueBeraResult {
    double statistic;  ///< The JB test statistic
    double p_value;    ///< P-value from chi-square distribution with df=2
};

/**
 * @brief Calculate the Jarque-Bera test statistic for testing normality.
 *
 * The Jarque-Bera test statistic is computed as:
 * JB = n/6 * (S² + K²/4)
 *
 * where:
 * - n is the sample size
 * - S is the sample skewness
 * - K is the sample excess kurtosis
 *
 * Under the null hypothesis (data is normally distributed),
 * JB follows a chi-square distribution with 2 degrees of freedom.
 *
 * @param data Span of data values to test
 * @return The Jarque-Bera test statistic
 * @throws std::invalid_argument if data is too short (needs at least 4 elements)
 */
[[nodiscard]] double jarque_bera_statistic(std::span<const double> data);

/**
 * @brief Perform the Jarque-Bera test for normality.
 *
 * Tests the null hypothesis that data comes from a normal distribution
 * versus the alternative that it does not.
 *
 * The test computes:
 * 1. The Jarque-Bera test statistic based on sample skewness and kurtosis
 * 2. The p-value from the chi-square distribution with 2 degrees of freedom
 *
 * Interpretation:
 * - High p-value (e.g., > 0.05): Fail to reject null hypothesis - data appears
 *   to be normally distributed
 * - Low p-value (e.g., < 0.05): Reject null hypothesis - data shows significant
 *   departure from normality
 *
 * @param data Span of data values to test
 * @return JarqueBeraResult containing the statistic and p-value
 * @throws std::invalid_argument if data is too short (needs at least 4 elements)
 */
[[nodiscard]] JarqueBeraResult jarque_bera_test(std::span<const double> data);

// Convenience overloads for std::vector
[[nodiscard]] inline double jarque_bera_statistic(const std::vector<double>& data) {
    return jarque_bera_statistic(std::span<const double>(data));
}

[[nodiscard]] inline JarqueBeraResult jarque_bera_test(const std::vector<double>& data) {
    return jarque_bera_test(std::span<const double>(data));
}

}  // namespace ag::stats
