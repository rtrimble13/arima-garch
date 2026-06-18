#include "ag/stats/LjungBox.hpp"

#include "ag/stats/ACF.hpp"
#include "ag/stats/SpecialFunctions.hpp"

#include <stdexcept>

namespace ag::stats {

double ljung_box_statistic(std::span<const double> residuals, std::size_t lags) {
    const std::size_t n = residuals.size();

    if (n == 0) {
        throw std::invalid_argument("Cannot compute Ljung-Box statistic for empty residuals");
    }

    if (lags == 0) {
        throw std::invalid_argument("Number of lags must be positive");
    }

    if (lags >= n) {
        throw std::invalid_argument("Number of lags must be less than sample size");
    }

    // Compute ACF for the residuals
    std::vector<double> acf_values = acf(residuals, lags);

    // Calculate the Ljung-Box Q statistic
    // Q = n(n+2) * Σ(ρ²ₖ/(n-k)) for k=1 to h
    double q = 0.0;
    for (std::size_t k = 1; k <= lags; ++k) {
        double rho_k = acf_values[k];
        q += (rho_k * rho_k) / static_cast<double>(n - k);
    }

    q *= static_cast<double>(n * (n + 2));

    return q;
}

LjungBoxResult ljung_box_test(std::span<const double> residuals, std::size_t lags,
                              std::size_t dof) {
    // Compute the test statistic
    double q = ljung_box_statistic(residuals, lags);

    // Degrees of freedom defaults to lags if not specified
    std::size_t degrees_of_freedom = (dof == 0) ? lags : dof;

    if (degrees_of_freedom == 0) {
        throw std::invalid_argument("Degrees of freedom must be positive");
    }

    // Compute p-value using chi-square distribution
    double p_value = chi_square_ccdf(q, static_cast<double>(degrees_of_freedom));

    return LjungBoxResult{
        .statistic = q,
        .p_value = p_value,
        .lags = lags,
        .dof = degrees_of_freedom,
    };
}

}  // namespace ag::stats
