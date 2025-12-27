#include "ag/estimation/Likelihood.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace ag::estimation {

ArimaGarchLikelihood::ArimaGarchLikelihood(const ag::models::ArimaGarchSpec& spec,
                                           InnovationDistribution dist)
    : spec_(spec), dist_(dist), arima_(spec.arimaSpec), garch_(spec.garchSpec) {
    // Specs are validated in model constructors
}

double ArimaGarchLikelihood::computeNegativeLogLikelihood(
    const double* data, std::size_t size, const ag::models::arima::ArimaParameters& arima_params,
    const ag::models::garch::GarchParameters& garch_params, double df) const {
    if (data == nullptr) {
        throw std::invalid_argument("Data pointer cannot be null");
    }
    if (size == 0) {
        throw std::invalid_argument("Data size must be positive");
    }

    // Validate df for Student-t distribution
    if (dist_ == InnovationDistribution::StudentT) {
        if (df <= 2.0) {
            throw std::invalid_argument(
                "Degrees of freedom must be > 2 for Student-t distribution");
        }
    }

    // Step 1: Compute ARIMA residuals (innovations)
    std::vector<double> residuals = arima_.computeResiduals(data, size, arima_params);

    // Step 2: Compute GARCH conditional variances
    std::vector<double> conditional_variances =
        garch_.computeConditionalVariances(residuals.data(), residuals.size(), garch_params);

    // Step 3: Compute negative log-likelihood based on distribution type
    double nll = 0.0;

    if (dist_ == InnovationDistribution::Normal) {
        // Normal distribution: NLL = Σ 0.5 * (log(h_t) + ε_t² / h_t)
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
    } else {  // StudentT
        // Student-t distribution: Use specialized log-likelihood
        for (std::size_t t = 0; t < residuals.size(); ++t) {
            double eps_t = residuals[t];
            double h_t = conditional_variances[t];

            // Guard against non-positive variance
            if (h_t <= 0.0) {
                throw std::runtime_error("Conditional variance must be positive");
            }

            nll += studentTLogLikelihood(eps_t, h_t, df);
        }
    }

    return nll;
}

double ArimaGarchLikelihood::studentTLogLikelihood(double residual, double variance,
                                                   double df) const {
    // Student-t log-likelihood contribution (negative)
    // -log L = -log(Γ((df+1)/2)) + log(Γ(df/2)) + 0.5*log(π*(df-2)*h_t)
    //          + 0.5*(df+1)*log(1 + ε_t²/((df-2)*h_t))
    //
    // Simplifying by factoring out constant terms that depend only on df:
    // C(df) = -log(Γ((df+1)/2)) + log(Γ(df/2)) + 0.5*log(π*(df-2))
    //
    // Per-observation contribution:
    // -log L_t = C(df) + 0.5*log(h_t) + 0.5*(df+1)*log(1 + ε_t²/((df-2)*h_t))

    const double df_minus_2 = df - 2.0;
    const double scaled_variance = df_minus_2 * variance;
    const double standardized_sq = (residual * residual) / scaled_variance;

    // Compute constant term (depends only on df, could be cached for efficiency)
    // C(df) = -lgamma((df+1)/2) + lgamma(df/2) + 0.5*log(π*(df-2))
    const double constant = -std::lgamma((df + 1.0) / 2.0) + std::lgamma(df / 2.0) +
                            0.5 * std::log(std::numbers::pi * df_minus_2);

    // Per-observation contribution
    const double nll_t =
        constant + 0.5 * std::log(variance) + 0.5 * (df + 1.0) * std::log1p(standardized_sq);

    return nll_t;
}

}  // namespace ag::estimation
