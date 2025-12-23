#pragma once

#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"

#include <cstddef>
#include <vector>

namespace ag::diagnostics {

/**
 * @brief Container for residual series from ARIMA-GARCH model filtering.
 *
 * Contains the three key series produced by filtering observations through
 * a fitted ARIMA-GARCH model:
 * - eps_t: Raw residuals (innovations) from the conditional mean
 * - h_t: Conditional variances from the GARCH component
 * - std_eps_t: Standardized residuals (eps_t / sqrt(h_t))
 *
 * These series are essential for model diagnostics, including:
 * - Testing for remaining autocorrelation (Ljung-Box test)
 * - Testing for normality (Jarque-Bera test)
 * - Verifying that standardized residuals are approximately N(0,1)
 */
struct ResidualSeries {
    std::vector<double> eps_t;      // Raw residuals (innovations)
    std::vector<double> h_t;        // Conditional variances
    std::vector<double> std_eps_t;  // Standardized residuals
};

/**
 * @brief Compute residual series by filtering data through a fitted ARIMA-GARCH model.
 *
 * This function runs the ARIMA-GARCH filter over the provided time series data,
 * producing the residual series (eps_t), conditional variances (h_t), and
 * standardized residuals (std_eps_t = eps_t / sqrt(h_t)).
 *
 * The filtering process:
 * 1. For each observation y_t in the series:
 *    a. Compute conditional mean μ_t from ARIMA component
 *    b. Compute residual eps_t = y_t - μ_t
 *    c. Compute conditional variance h_t from GARCH component
 *    d. Compute standardized residual std_eps_t = eps_t / sqrt(h_t)
 *    e. Update model state for next observation
 *
 * For a correctly specified model, the standardized residuals should be
 * approximately i.i.d. with mean 0 and variance 1.
 *
 * @param spec ARIMA-GARCH model specification
 * @param params Fitted model parameters
 * @param data Pointer to time series data
 * @param size Number of observations in the time series
 * @return ResidualSeries containing eps_t, h_t, and std_eps_t vectors
 * @throws std::invalid_argument if data is nullptr or size is 0
 * @throws std::invalid_argument if model parameters are invalid
 *
 * @note The returned series will have the same length as the input data.
 * @note All returned values are guaranteed to be finite (no NaNs or Infs).
 */
ResidualSeries computeResiduals(const ag::models::ArimaGarchSpec& spec,
                                const ag::models::composite::ArimaGarchParameters& params,
                                const double* data, std::size_t size);

/**
 * @brief Compute residual series by filtering data through a fitted ARIMA-GARCH model.
 *
 * Convenience overload that accepts a vector instead of a pointer.
 *
 * @param spec ARIMA-GARCH model specification
 * @param params Fitted model parameters
 * @param data Time series data as a vector
 * @return ResidualSeries containing eps_t, h_t, and std_eps_t vectors
 * @throws std::invalid_argument if data is empty
 * @throws std::invalid_argument if model parameters are invalid
 */
inline ResidualSeries computeResiduals(const ag::models::ArimaGarchSpec& spec,
                                       const ag::models::composite::ArimaGarchParameters& params,
                                       const std::vector<double>& data) {
    return computeResiduals(spec, params, data.data(), data.size());
}

}  // namespace ag::diagnostics
