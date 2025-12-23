#include "ag/selection/InformationCriteria.hpp"

#include <cmath>

namespace ag::selection {

double computeAIC(double log_likelihood, int k) {
    return 2.0 * k - 2.0 * log_likelihood;
}

double computeBIC(double log_likelihood, int k, std::size_t n) {
    return k * std::log(static_cast<double>(n)) - 2.0 * log_likelihood;
}

double computeAICc(double log_likelihood, int k, std::size_t n) {
    double aic = computeAIC(log_likelihood, k);
    double n_double = static_cast<double>(n);
    double k_double = static_cast<double>(k);
    double correction = (2.0 * k_double * (k_double + 1.0)) / (n_double - k_double - 1.0);
    return aic + correction;
}

}  // namespace ag::selection
