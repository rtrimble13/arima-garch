#pragma once

#include "ag/models/ArimaSpec.hpp"
#include "ag/models/arima/ArimaState.hpp"

#include <cstddef>
#include <vector>

namespace ag::models::arima {

/**
 * @brief Parameters for an ARIMA model.
 *
 * Contains the coefficient values for the AR and MA components,
 * plus an intercept term for the conditional mean.
 */
struct ArimaParameters {
    double intercept;             // Constant term (c or μ)
    std::vector<double> ar_coef;  // AR coefficients: φ₁, φ₂, ..., φₚ
    std::vector<double> ma_coef;  // MA coefficients: θ₁, θ₂, ..., θ_q

    /**
     * @brief Construct ARIMA parameters with given sizes.
     * @param p Number of AR coefficients
     * @param q Number of MA coefficients
     */
    ArimaParameters(int p, int q) : intercept(0.0), ar_coef(p, 0.0), ma_coef(q, 0.0) {}
};

/**
 * @brief ARIMA model for computing conditional mean and residuals.
 *
 * ArimaModel implements the core recursion for computing residuals (innovations)
 * from an ARIMA(p,d,q) model given parameters and time series data.
 *
 * The ARIMA model equation for the conditional mean is:
 * - After differencing d times, the model is ARMA(p,q):
 *   y_t = c + φ₁*y_{t-1} + ... + φₚ*y_{t-p} + ε_t + θ₁*ε_{t-1} + ... + θ_q*ε_{t-q}
 *
 * The residual (innovation) at time t is:
 *   ε_t = y_t - (c + φ₁*y_{t-1} + ... + φₚ*y_{t-p} + θ₁*ε_{t-1} + ... + θ_q*ε_{t-q})
 */
class ArimaModel {
public:
    /**
     * @brief Construct an ARIMA model with given specification.
     * @param spec ARIMA(p,d,q) specification
     */
    explicit ArimaModel(const ag::models::ArimaSpec& spec);

    /**
     * @brief Compute residuals for a given time series.
     *
     * This method performs the core ARIMA recursion to compute the residuals
     * (innovations) for each observation in the time series, given the model
     * parameters.
     *
     * The residuals represent the difference between observed values and the
     * conditional mean predicted by the ARIMA model.
     *
     * @param data Pointer to time series data
     * @param size Number of observations in the time series
     * @param params ARIMA model parameters (intercept, AR, and MA coefficients)
     * @return Vector of residuals, one for each observation (after differencing loss)
     */
    [[nodiscard]] std::vector<double> computeResiduals(const double* data, std::size_t size,
                                                       const ArimaParameters& params) const;

    /**
     * @brief Get the ARIMA specification for this model.
     * @return The ARIMA(p,d,q) specification
     */
    [[nodiscard]] const ag::models::ArimaSpec& getSpec() const noexcept { return spec_; }

private:
    ag::models::ArimaSpec spec_;  // ARIMA(p,d,q) specification

    /**
     * @brief Compute the conditional mean for a single observation.
     *
     * Calculates the expected value based on the ARIMA model equation:
     * E[y_t] = c + φ₁*y_{t-1} + ... + φₚ*y_{t-p} + θ₁*ε_{t-1} + ... + θ_q*ε_{t-q}
     *
     * @param state Current state containing historical observations and residuals
     * @param params Model parameters
     * @return The conditional mean for the current time step
     */
    [[nodiscard]] double computeConditionalMean(const ArimaState& state,
                                                const ArimaParameters& params) const;
};

}  // namespace ag::models::arima
