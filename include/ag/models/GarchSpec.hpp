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
 * Both parameters must be at least 1 for a valid GARCH model.
 */
struct GarchSpec {
    const int p;  // GARCH order (must be >= 1)
    const int q;  // ARCH order (must be >= 1)

    /**
     * @brief Construct a GARCH specification with validation.
     * @param p_val Order of GARCH component (must be >= 1)
     * @param q_val Order of ARCH component (must be >= 1)
     * @throws std::invalid_argument if p < 1 or q < 1
     */
    constexpr GarchSpec(int p_val, int q_val) : p(p_val), q(q_val) { validate(); }

    /**
     * @brief Validate GARCH specification parameters.
     * @throws std::invalid_argument if p < 1 or q < 1
     */
    constexpr void validate() const {
        if (p < 1) {
            throw std::invalid_argument("GARCH parameter p must be >= 1, got: " +
                                        std::to_string(p));
        }
        if (q < 1) {
            throw std::invalid_argument("GARCH parameter q must be >= 1, got: " +
                                        std::to_string(q));
        }
    }

    /**
     * @brief Check if this is a GARCH(1,1) model (the most common specification).
     * @return true if p == 1 and q == 1, false otherwise
     */
    [[nodiscard]] constexpr bool isGarch11() const noexcept { return p == 1 && q == 1; }
};

}  // namespace ag::models
