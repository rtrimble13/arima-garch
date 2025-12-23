#pragma once

#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/arima/ArimaModel.hpp"
#include "ag/models/garch/GarchModel.hpp"

#include <cstddef>
#include <vector>

namespace ag::estimation {

/**
 * @brief Likelihood computation for ARIMA-GARCH models with Normal innovations.
 *
 * ArimaGarchLikelihood computes the negative log-likelihood (NLL) for a combined
 * ARIMA-GARCH model assuming normally distributed innovations. This is the
 * objective function typically minimized during maximum likelihood estimation.
 *
 * The likelihood combines two components:
 * 1. ARIMA model: Computes residuals (innovations) ε_t from the conditional mean
 * 2. GARCH model: Computes conditional variances h_t from the residuals
 *
 * For Normal innovations, the negative log-likelihood is:
 *   NLL = Σ 0.5 * (log(h_t) + ε_t² / h_t)
 *
 * This ignores the constant term 0.5*log(2π) which doesn't affect optimization.
 */
class ArimaGarchLikelihood {
public:
    /**
     * @brief Construct a likelihood evaluator for a given ARIMA-GARCH specification.
     * @param spec The ARIMA-GARCH model specification
     */
    explicit ArimaGarchLikelihood(const ag::models::ArimaGarchSpec& spec);

    /**
     * @brief Compute negative log-likelihood for Normal innovations.
     *
     * This method performs the complete likelihood computation:
     * 1. Computes ARIMA residuals from the time series data
     * 2. Computes GARCH conditional variances from the residuals
     * 3. Evaluates the Normal NLL: Σ 0.5 * (log(h_t) + ε_t² / h_t)
     *
     * The computation is deterministic and efficient, suitable for use in
     * iterative optimization algorithms.
     *
     * @param data Pointer to time series data
     * @param size Number of observations in the time series
     * @param arima_params ARIMA model parameters (intercept, AR, MA coefficients)
     * @param garch_params GARCH model parameters (omega, ARCH, GARCH coefficients)
     * @return Negative log-likelihood value (smaller is better fit)
     * @throws std::invalid_argument if inputs are invalid or parameters violate constraints
     */
    [[nodiscard]] double
    computeNegativeLogLikelihood(const double* data, std::size_t size,
                                 const ag::models::arima::ArimaParameters& arima_params,
                                 const ag::models::garch::GarchParameters& garch_params) const;

    /**
     * @brief Get the ARIMA-GARCH specification for this likelihood evaluator.
     * @return The ARIMA-GARCH specification
     */
    [[nodiscard]] const ag::models::ArimaGarchSpec& getSpec() const noexcept { return spec_; }

private:
    ag::models::ArimaGarchSpec spec_;      // Model specification
    ag::models::arima::ArimaModel arima_;  // ARIMA model for residuals
    ag::models::garch::GarchModel garch_;  // GARCH model for variances
};

}  // namespace ag::estimation
