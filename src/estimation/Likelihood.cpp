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

            // Guard against non-positive or non-finite variance. The negated
            // comparison also rejects NaN (every comparison with NaN is false),
            // which a bare `h_t <= 0.0` would silently let through.
            if (!(h_t > 0.0)) {
                throw std::runtime_error("Conditional variance must be positive and finite");
            }

            // Accumulate: 0.5 * (log(h_t) + ε_t² / h_t)
            nll += 0.5 * (std::log(h_t) + (eps_t * eps_t) / h_t);
        }
    } else {  // StudentT
        // Student-t distribution. The df-only constant terms factor out of
        // the inner loop and are far from free (two lgamma calls); compute
        // them once.
        const double df_minus_2 = df - 2.0;
        const double half_df_plus_1 = (df + 1.0) / 2.0;
        const double df_constant = -std::lgamma(half_df_plus_1) + std::lgamma(df / 2.0) +
                                   0.5 * std::log(std::numbers::pi * df_minus_2);

        for (std::size_t t = 0; t < residuals.size(); ++t) {
            double eps_t = residuals[t];
            double h_t = conditional_variances[t];

            if (!(h_t > 0.0)) {
                throw std::runtime_error("Conditional variance must be positive and finite");
            }

            const double standardized_sq = (eps_t * eps_t) / (df_minus_2 * h_t);
            nll += df_constant + 0.5 * std::log(h_t) + half_df_plus_1 * std::log1p(standardized_sq);
        }
    }

    // A diverging/explosive recursion (or a non-finite residual that slipped
    // through) can yield a NaN/Inf NLL. Returning it would let Nelder-Mead
    // treat the vertex as "not worse" (every comparison with NaN is false) and
    // potentially report convergence on garbage parameters. Throwing here lets
    // the FitDriver objective map it to CONSTRAINT_PENALTY so the optimizer
    // steers away consistently.
    if (!std::isfinite(nll)) {
        throw std::runtime_error("Negative log-likelihood is not finite");
    }

    return nll;
}

}  // namespace ag::estimation
