#pragma once

#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/arima/ArimaModel.hpp"
#include "ag/models/garch/GarchModel.hpp"

#include <cstddef>
#include <vector>

namespace ag::estimation {

/**
 * @brief Pack/unpack ARIMA-GARCH parameters into the flat std::vector<double>
 *        layout consumed by the optimizer.
 *
 * Single source of truth for the convention used between Engine,
 * ModelSelector, CrossValidation, and FitDriver. Previously these sites
 * each had inline copies of the same loop, and a subtle convention
 * mismatch existed: Engine packed the ARIMA intercept unconditionally
 * while ModelSelector and CrossValidation skipped the ARIMA block
 * entirely for zero-order specs. The Selector convention is adopted here
 * (skip the ARIMA block when spec.arimaSpec.isZeroOrder()): for a
 * zero-order ARIMA spec the intercept is intentionally fixed at zero
 * rather than estimated. With no AR or MA structure to anchor it, adding
 * the intercept as a free optimization parameter would provide no
 * useful information. Callers that need a non-zero constant mean should
 * difference the data or use a non-zero-order ARIMA spec.
 *
 * Layout (when present, in order):
 *   - intercept (if !arimaSpec.isZeroOrder())
 *   - p AR coefficients
 *   - q MA coefficients
 *   - omega (if !garchSpec.isNull())
 *   - q ARCH (alpha) coefficients
 *   - p GARCH (beta) coefficients
 */
namespace param_vector {

/**
 * @brief Compute the optimization vector length for a given spec.
 */
[[nodiscard]] std::size_t size(const ag::models::ArimaGarchSpec& spec) noexcept;

/**
 * @brief Pack initial ARIMA + GARCH parameters into the flat vector.
 */
[[nodiscard]] std::vector<double> pack(const ag::models::ArimaGarchSpec& spec,
                                       const ag::models::arima::ArimaParameters& arima_params,
                                       const ag::models::garch::GarchParameters& garch_params);

/**
 * @brief Unpack a flat parameter vector back into typed structs.
 *
 * out_arima and out_garch must already be sized for the spec (use the
 * ArimaParameters/GarchParameters constructors that take p/q). Throws
 * std::invalid_argument if params.size() != size(spec).
 */
void unpack(const std::vector<double>& params, const ag::models::ArimaGarchSpec& spec,
            ag::models::arima::ArimaParameters& out_arima,
            ag::models::garch::GarchParameters& out_garch);

}  // namespace param_vector

}  // namespace ag::estimation
