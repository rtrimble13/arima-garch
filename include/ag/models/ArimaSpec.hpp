#pragma once

#include <stdexcept>
#include <string>

namespace ag::models {

/**
 * @brief Immutable specification for ARIMA (Autoregressive Integrated Moving Average) model.
 *
 * ArimaSpec defines the order parameters (p, d, q) for an ARIMA model:
 * - p: Order of the autoregressive (AR) component
 * - d: Degree of differencing
 * - q: Order of the moving average (MA) component
 *
 * All parameters must be non-negative integers.
 */
struct ArimaSpec {
    const int p;  // AR order (non-negative)
    const int d;  // Differencing degree (non-negative)
    const int q;  // MA order (non-negative)

    /**
     * @brief Construct an ARIMA specification with validation.
     * @param p_val Order of autoregressive component (must be >= 0)
     * @param d_val Degree of differencing (must be >= 0)
     * @param q_val Order of moving average component (must be >= 0)
     * @throws std::invalid_argument if any parameter is negative
     */
    ArimaSpec(int p_val, int d_val, int q_val) : p(p_val), d(d_val), q(q_val) { validate(); }

    /**
     * @brief Validate ARIMA specification parameters.
     * @throws std::invalid_argument if any parameter is negative
     */
    void validate() const {
        if (p < 0) {
            throw std::invalid_argument("ARIMA parameter p must be non-negative, got: " +
                                        std::to_string(p));
        }
        if (d < 0) {
            throw std::invalid_argument("ARIMA parameter d must be non-negative, got: " +
                                        std::to_string(d));
        }
        if (q < 0) {
            throw std::invalid_argument("ARIMA parameter q must be non-negative, got: " +
                                        std::to_string(q));
        }
    }

    /**
     * @brief Check if this is a zero-order ARIMA model (all parameters are 0).
     * @return true if p, d, and q are all 0, false otherwise
     */
    [[nodiscard]] constexpr bool isZeroOrder() const noexcept { return p == 0 && d == 0 && q == 0; }

    /**
     * @brief Check if this specification includes differencing.
     * @return true if d > 0, false otherwise
     */
    [[nodiscard]] constexpr bool hasDifferencing() const noexcept { return d > 0; }

    /**
     * @brief Check if this specification includes autoregressive component.
     * @return true if p > 0, false otherwise
     */
    [[nodiscard]] constexpr bool hasAR() const noexcept { return p > 0; }

    /**
     * @brief Check if this specification includes moving average component.
     * @return true if q > 0, false otherwise
     */
    [[nodiscard]] constexpr bool hasMA() const noexcept { return q > 0; }
};

}  // namespace ag::models
