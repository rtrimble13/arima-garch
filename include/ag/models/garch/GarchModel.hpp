#pragma once

#include "ag/models/GarchSpec.hpp"
#include "ag/models/garch/GarchState.hpp"

#include <cstddef>
#include <vector>

namespace ag::models::garch {

/**
 * @brief Parameters for a GARCH model.
 *
 * Contains the coefficient values for the ARCH and GARCH components,
 * plus an omega term (constant) for the conditional variance equation.
 */
struct GarchParameters {
    double omega;                    // Constant term (ω > 0)
    std::vector<double> alpha_coef;  // ARCH coefficients: α₁, α₂, ..., α_q
    std::vector<double> beta_coef;   // GARCH coefficients: β₁, β₂, ..., βₚ

    /**
     * @brief Construct GARCH parameters with given sizes.
     * @param p Number of GARCH coefficients
     * @param q Number of ARCH coefficients
     */
    GarchParameters(int p, int q) : omega(0.0), alpha_coef(q, 0.0), beta_coef(p, 0.0) {}

    /**
     * @brief Check if parameters satisfy positivity constraints.
     * @return true if omega > 0, all alpha >= 0, all beta >= 0
     */
    [[nodiscard]] bool isPositive() const noexcept;

    /**
     * @brief Check if parameters satisfy stationarity constraint.
     *
     * For stationarity, the sum of all alpha and beta coefficients must be < 1:
     * Σα_i + Σβ_j < 1
     *
     * @return true if stationarity constraint is satisfied
     */
    [[nodiscard]] bool isStationary() const noexcept;

    /**
     * @brief Compute unconditional variance from parameters.
     *
     * The unconditional variance is given by:
     * σ² = ω / (1 - Σα_i - Σβ_j)
     *
     * This is only valid when the process is stationary.
     *
     * @return Unconditional variance if stationary, 0.0 otherwise
     */
    [[nodiscard]] double unconditionalVariance() const noexcept;
};

/**
 * @brief GARCH model for computing conditional variance (variance filter).
 *
 * GarchModel implements the core recursion for computing the conditional variance h_t
 * from a GARCH(p,q) model given parameters and residuals from an ARIMA model.
 *
 * The GARCH model equation for the conditional variance is:
 *   h_t = ω + α₁*ε²_{t-1} + ... + α_q*ε²_{t-q} + β₁*h_{t-1} + ... + βₚ*h_{t-p}
 *
 * where:
 * - h_t is the conditional variance at time t
 * - ε_t are the residuals from the ARIMA model
 * - ω > 0 is the constant term
 * - α_i >= 0 are the ARCH coefficients
 * - β_j >= 0 are the GARCH coefficients
 *
 * For valid parameters, h_t > 0 for all t.
 */
class GarchModel {
public:
    /**
     * @brief Construct a GARCH model with given specification.
     * @param spec GARCH(p,q) specification
     */
    explicit GarchModel(const ag::models::GarchSpec& spec);

    /**
     * @brief Compute conditional variances for a given residual series.
     *
     * This method performs the core GARCH recursion to compute the conditional
     * variance h_t for each residual in the series, given the model parameters.
     *
     * The conditional variances represent the time-varying variance of the
     * residuals (innovations) from the ARIMA model.
     *
     * @param residuals Pointer to residual series from ARIMA model
     * @param size Number of residuals
     * @param params GARCH model parameters (omega, ARCH, and GARCH coefficients)
     * @return Vector of conditional variances, one for each residual
     * @throws std::invalid_argument if parameters violate constraints
     */
    [[nodiscard]] std::vector<double>
    computeConditionalVariances(const double* residuals, std::size_t size,
                                const GarchParameters& params) const;

    /**
     * @brief Get the GARCH specification for this model.
     * @return The GARCH(p,q) specification
     */
    [[nodiscard]] const ag::models::GarchSpec& getSpec() const noexcept { return spec_; }

private:
    ag::models::GarchSpec spec_;  // GARCH(p,q) specification

    /**
     * @brief Compute the conditional variance for a single observation.
     *
     * Calculates the conditional variance based on the GARCH model equation:
     * h_t = ω + α₁*ε²_{t-1} + ... + α_q*ε²_{t-q} + β₁*h_{t-1} + ... + βₚ*h_{t-p}
     *
     * @param state Current state containing historical variances and squared residuals
     * @param params Model parameters
     * @return The conditional variance for the current time step
     */
    [[nodiscard]] double computeConditionalVariance(const GarchState& state,
                                                    const GarchParameters& params) const;
};

}  // namespace ag::models::garch
