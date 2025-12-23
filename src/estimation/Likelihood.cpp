#include "ag/estimation/Likelihood.hpp"

#include <cmath>
#include <stdexcept>

namespace ag::estimation {

ArimaGarchLikelihood::ArimaGarchLikelihood(const ag::models::ArimaGarchSpec& spec)
    : spec_(spec), arima_(spec.arimaSpec), garch_(spec.garchSpec) {
    // Specs are validated in model constructors
}

double ArimaGarchLikelihood::computeNegativeLogLikelihood(
    const double* data, std::size_t size, const ag::models::arima::ArimaParameters& arima_params,
    const ag::models::garch::GarchParameters& garch_params) const {
    if (data == nullptr) {
        throw std::invalid_argument("Data pointer cannot be null");
    }
    if (size == 0) {
        throw std::invalid_argument("Data size must be positive");
    }

    // Step 1: Compute ARIMA residuals (innovations)
    std::vector<double> residuals = arima_.computeResiduals(data, size, arima_params);

    // Step 2: Compute GARCH conditional variances
    std::vector<double> conditional_variances =
        garch_.computeConditionalVariances(residuals.data(), residuals.size(), garch_params);

    // Step 3: Compute negative log-likelihood for Normal innovations
    // NLL = Σ 0.5 * (log(h_t) + ε_t² / h_t)
    double nll = 0.0;
    for (std::size_t t = 0; t < residuals.size(); ++t) {
        double eps_t = residuals[t];
        double h_t = conditional_variances[t];

        // Guard against non-positive variance (should not happen with valid parameters)
        if (h_t <= 0.0) {
            throw std::runtime_error("Conditional variance must be positive");
        }

        // Accumulate: 0.5 * (log(h_t) + ε_t² / h_t)
        nll += 0.5 * (std::log(h_t) + (eps_t * eps_t) / h_t);
    }

    return nll;
}

}  // namespace ag::estimation
