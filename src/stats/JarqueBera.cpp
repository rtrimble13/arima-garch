#include "ag/stats/JarqueBera.hpp"

#include "ag/stats/Descriptive.hpp"
#include "ag/stats/SpecialFunctions.hpp"

#include <stdexcept>

namespace ag::stats {

double jarque_bera_statistic(std::span<const double> data) {
    const std::size_t n = data.size();

    if (n < 4) {
        throw std::invalid_argument(
            "Cannot compute Jarque-Bera statistic with fewer than 4 data points");
    }

    // Compute sample skewness and kurtosis
    double S = skewness(data);
    double K = kurtosis(data);

    // Calculate the Jarque-Bera statistic
    // JB = n/6 * (S² + K²/4)
    double jb = (static_cast<double>(n) / 6.0) * (S * S + (K * K) / 4.0);

    return jb;
}

JarqueBeraResult jarque_bera_test(std::span<const double> data) {
    // Compute the test statistic
    double jb = jarque_bera_statistic(data);

    // Compute p-value using chi-square distribution with 2 degrees of freedom
    double p_value = chi_square_ccdf(jb, 2.0);

    return JarqueBeraResult{
        .statistic = jb,
        .p_value = p_value,
    };
}

}  // namespace ag::stats
