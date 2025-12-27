#pragma once

#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/arima/ArimaModel.hpp"
#include "ag/models/garch/GarchModel.hpp"

#include <cstddef>
#include <vector>

namespace ag::estimation {

/**
 * @brief Innovation distribution type for likelihood estimation.
 */
enum class InnovationDistribution {
    Normal,   // Standard normal N(0,1)
    StudentT  // Standardized Student-t with specified degrees of freedom
};

/**
 * @brief Likelihood computation for ARIMA-GARCH models with Normal or Student-t innovations.
 *
 * ArimaGarchLikelihood computes the negative log-likelihood (NLL) for a combined
 * ARIMA-GARCH model assuming normally distributed or Student-t distributed innovations.
 * This is the objective function typically minimized during maximum likelihood estimation.
 *
 * The likelihood combines two components:
 * 1. ARIMA model: Computes residuals (innovations) ε_t from the conditional mean
 * 2. GARCH model: Computes conditional variances h_t from the residuals
 *
 * For Normal innovations, the negative log-likelihood is:
 *   NLL = Σ 0.5 * (log(h_t) + ε_t² / h_t)
 *
 * For Student-t innovations with df degrees of freedom, the negative log-likelihood is:
 *   NLL = Σ -log(Γ((df+1)/2)) + log(Γ(df/2)) + 0.5*log(π*(df-2)*h_t)
 *            + 0.5*(df+1)*log(1 + ε_t²/((df-2)*h_t))
 *
 * Constants that don't affect optimization may be omitted.
 */
class ArimaGarchLikelihood {
public:
    /**
     * @brief Construct a likelihood evaluator for a given ARIMA-GARCH specification.
     * @param spec The ARIMA-GARCH model specification
     * @param dist Innovation distribution type (default: Normal)
     */
    explicit ArimaGarchLikelihood(const ag::models::ArimaGarchSpec& spec,
                                  InnovationDistribution dist = InnovationDistribution::Normal);

    /**
     * @brief Compute negative log-likelihood for Normal or Student-t innovations.
     *
     * This method performs the complete likelihood computation:
     * 1. Computes ARIMA residuals from the time series data
     * 2. Computes GARCH conditional variances from the residuals
     * 3. Evaluates the NLL based on the distribution type:
     *    - Normal: Σ 0.5 * (log(h_t) + ε_t² / h_t)
     *    - Student-t: Uses Student-t log-likelihood with specified df
     *
     * The computation is deterministic and efficient, suitable for use in
     * iterative optimization algorithms.
     *
     * @param data Pointer to time series data
     * @param size Number of observations in the time series
     * @param arima_params ARIMA model parameters (intercept, AR, MA coefficients)
     * @param garch_params GARCH model parameters (omega, ARCH, GARCH coefficients)
     * @param df Degrees of freedom for Student-t distribution (required if dist=StudentT, must be >
     * 2)
     * @return Negative log-likelihood value (smaller is better fit)
     * @throws std::invalid_argument if inputs are invalid, parameters violate constraints,
     *         or df is invalid for Student-t distribution
     */
    [[nodiscard]] double
    computeNegativeLogLikelihood(const double* data, std::size_t size,
                                 const ag::models::arima::ArimaParameters& arima_params,
                                 const ag::models::garch::GarchParameters& garch_params,
                                 double df = 0.0) const;

    /**
     * @brief Get the ARIMA-GARCH specification for this likelihood evaluator.
     * @return The ARIMA-GARCH specification
     */
    [[nodiscard]] const ag::models::ArimaGarchSpec& getSpec() const noexcept { return spec_; }

    /**
     * @brief Get the innovation distribution type for this likelihood evaluator.
     * @return The innovation distribution type
     */
    [[nodiscard]] InnovationDistribution getDistribution() const noexcept { return dist_; }

private:
    /**
     * @brief Compute log-likelihood contribution for a single Student-t innovation.
     *
     * Computes the log-likelihood for a standardized residual under Student-t distribution:
     *   log L = log(Γ((df+1)/2)) - log(Γ(df/2)) - 0.5*log(π*(df-2)*h_t)
     *           - 0.5*(df+1)*log(1 + z_t²)
     * where z_t = ε_t / sqrt((df-2)*h_t) is the standardized residual
     *
     * The negative log-likelihood contribution is:
     *   -log L = -log(Γ((df+1)/2)) + log(Γ(df/2)) + 0.5*log(π*(df-2)*h_t)
     *            + 0.5*(df+1)*log(1 + ε_t²/((df-2)*h_t))
     *
     * @param residual ARIMA residual ε_t
     * @param variance Conditional variance h_t
     * @param df Degrees of freedom (must be > 2)
     * @return Negative log-likelihood contribution for this observation
     */
    [[nodiscard]] double studentTLogLikelihood(double residual, double variance, double df) const;

    ag::models::ArimaGarchSpec spec_;      // Model specification
    InnovationDistribution dist_;          // Innovation distribution type
    ag::models::arima::ArimaModel arima_;  // ARIMA model for residuals
    ag::models::garch::GarchModel garch_;  // GARCH model for variances
};

}  // namespace ag::estimation
