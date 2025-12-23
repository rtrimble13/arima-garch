#pragma once

#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/arima/ArimaModel.hpp"
#include "ag/models/garch/GarchModel.hpp"

#include <cstddef>
#include <random>
#include <vector>

namespace ag::estimation {

/**
 * @brief Initialize ARIMA parameters using ACF/PACF heuristics.
 *
 * This function generates initial ARIMA parameter estimates using simple
 * heuristics based on the autocorrelation function (ACF) and partial
 * autocorrelation function (PACF) of the time series data.
 *
 * Heuristics:
 * - Intercept: Set to the sample mean of the (differenced) data
 * - AR coefficients: Set to PACF values at corresponding lags, scaled by 0.9
 *   to ensure stability
 * - MA coefficients: Set to negative ACF values at corresponding lags,
 *   scaled by 0.9
 *
 * These simple heuristics provide a reasonable starting point for optimization,
 * though they are not guaranteed to be in the stationary/invertible region.
 *
 * @param data Pointer to time series data
 * @param size Number of observations in the time series
 * @param spec ARIMA specification (p, d, q)
 * @return Initialized ARIMA parameters
 * @throws std::invalid_argument if data is null, size is too small, or spec is invalid
 */
[[nodiscard]] ag::models::arima::ArimaParameters
initializeArimaParameters(const double* data, std::size_t size, const ag::models::ArimaSpec& spec);

/**
 * @brief Initialize GARCH parameters using method-of-moments.
 *
 * This function generates initial GARCH parameter estimates using the
 * method-of-moments approach based on the sample variance and autocorrelation
 * of squared residuals.
 *
 * Heuristics:
 * - omega: Set based on unconditional variance formula
 * - alpha coefficients: Set to small positive values (0.05 each) scaled by q
 * - beta coefficients: Set based on persistence, targeting sum(alpha + beta) ≈ 0.9
 *
 * The parameters are chosen to satisfy positivity and stationarity constraints.
 *
 * @param residuals Pointer to residual series from ARIMA model
 * @param size Number of residuals
 * @param spec GARCH specification (p, q)
 * @return Initialized GARCH parameters
 * @throws std::invalid_argument if residuals is null, size is too small, or spec is invalid
 */
[[nodiscard]] ag::models::garch::GarchParameters
initializeGarchParameters(const double* residuals, std::size_t size,
                          const ag::models::GarchSpec& spec);

/**
 * @brief Initialize combined ARIMA-GARCH parameters.
 *
 * This is a convenience function that initializes both ARIMA and GARCH
 * parameters in sequence:
 * 1. Initialize ARIMA parameters from data
 * 2. Compute ARIMA residuals
 * 3. Initialize GARCH parameters from residuals
 *
 * @param data Pointer to time series data
 * @param size Number of observations
 * @param spec ARIMA-GARCH specification
 * @return Pair of (ARIMA parameters, GARCH parameters)
 * @throws std::invalid_argument if inputs are invalid
 */
[[nodiscard]] std::pair<ag::models::arima::ArimaParameters, ag::models::garch::GarchParameters>
initializeArimaGarchParameters(const double* data, std::size_t size,
                               const ag::models::ArimaGarchSpec& spec);

/**
 * @brief Generate random perturbation around initial parameters.
 *
 * Creates a perturbed version of the input parameters by adding random noise.
 * This is used for random restarts in optimization.
 *
 * Each parameter is perturbed by adding a random value from a normal
 * distribution with mean 0 and standard deviation = scale * |param|,
 * where scale is typically 0.1 to 0.3.
 *
 * @param params Initial parameters
 * @param scale Perturbation scale (fraction of parameter magnitude)
 * @param rng Random number generator
 * @return Perturbed parameters
 */
[[nodiscard]] std::vector<double> perturbParameters(const std::vector<double>& params, double scale,
                                                    std::mt19937& rng);

}  // namespace ag::estimation
