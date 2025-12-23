#include "ag/selection/InformationCriteria.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ag::selection {

double computeAIC(double log_likelihood, int k) {
    return 2.0 * k - 2.0 * log_likelihood;
}

double computeBIC(double log_likelihood, int k, std::size_t n) {
    return k * std::log(static_cast<double>(n)) - 2.0 * log_likelihood;
}

double computeAICc(double log_likelihood, int k, std::size_t n) {
    // Validate that n > k+1 to avoid division by zero or negative denominator
    if (n <= static_cast<std::size_t>(k + 1)) {
        throw std::invalid_argument("AICc requires n > k+1 (sample size must exceed number of "
                                    "parameters plus one), got n=" +
                                    std::to_string(n) + ", k=" + std::to_string(k));
    }

    double aic = computeAIC(log_likelihood, k);
    double n_double = static_cast<double>(n);
    double k_double = static_cast<double>(k);
    double correction = (2.0 * k_double * (k_double + 1.0)) / (n_double - k_double - 1.0);
    return aic + correction;
}

}  // namespace ag::selection
