#pragma once

#include <stdexcept>
#include <string>

namespace ag::models {

/**
 * @brief Immutable specification for GARCH (Generalized Autoregressive Conditional
 * Heteroskedasticity) model.
 *
 * GarchSpec defines the order parameters (p, q) for a GARCH model:
 * - p: Order of the GARCH component (lagged conditional variances)
 * - q: Order of the ARCH component (lagged squared residuals)
 *
 * Both parameters must be >= 1 for a valid GARCH model.
 * Both parameters can be 0 to represent no GARCH component (ARIMA-only model).
 */
struct GarchSpec {
    const int p;  // GARCH order (must be >= 0)
    const int q;  // ARCH order (must be >= 0)

    /**
     * @brief Construct a GARCH specification with validation.
     * @param p_val Order of GARCH component (must be >= 0, both 0 for ARIMA-only)
     * @param q_val Order of ARCH component (must be >= 0, both 0 for ARIMA-only)
     * @throws std::invalid_argument if p < 0 or q < 0, or only one is 0
     */
    GarchSpec(int p_val, int q_val) : p(p_val), q(q_val) { validate(); }

    /**
     * @brief Validate GARCH specification parameters.
     * @throws std::invalid_argument if p < 0 or q < 0, or only one is 0
     */
    void validate() const {
        if (p < 0) {
            throw std::invalid_argument("GARCH parameter p must be >= 0, got: " +
                                        std::to_string(p));
        }
        if (q < 0) {
            throw std::invalid_argument("GARCH parameter q must be >= 0, got: " +
                                        std::to_string(q));
        }
        // Both must be 0 (ARIMA-only) or both must be >= 1 (GARCH model)
        if ((p == 0) != (q == 0)) {
            throw std::invalid_argument(
                "GARCH parameters must both be 0 (ARIMA-only) or both be >= 1, got p=" +
                std::to_string(p) + ", q=" + std::to_string(q));
        }
    }

    /**
     * @brief Check if this is a GARCH(1,1) model (the most common specification).
     * @return true if p == 1 and q == 1, false otherwise
     */
    [[nodiscard]] constexpr bool isGarch11() const noexcept { return p == 1 && q == 1; }

    /**
     * @brief Check if this represents no GARCH component (ARIMA-only model).
     * @return true if p == 0 and q == 0, false otherwise
     */
    [[nodiscard]] constexpr bool isNull() const noexcept { return p == 0 && q == 0; }
};

}  // namespace ag::models
