#pragma once

#include "ag/models/ArimaSpec.hpp"
#include "ag/models/GarchSpec.hpp"

namespace ag::models {

/**
 * @brief Immutable specification for combined ARIMA-GARCH model.
 *
 * ArimaGarchSpec combines ARIMA specification for the conditional mean
 * and GARCH specification for the conditional variance. This represents
 * a complete specification for modeling time series with both autocorrelation
 * in the mean and volatility clustering in the variance.
 *
 * The ARIMA component models the conditional mean:
 *   E[y_t | I_{t-1}] = ARIMA(p, d, q)
 *
 * The GARCH component models the conditional variance:
 *   Var[ε_t | I_{t-1}] = GARCH(P, Q)
 *
 * where ε_t are the residuals from the ARIMA model.
 */
struct ArimaGarchSpec {
    const ArimaSpec arimaSpec;  // Specification for conditional mean
    const GarchSpec garchSpec;  // Specification for conditional variance

    /**
     * @brief Construct an ARIMA-GARCH specification with validation.
     * @param arima ARIMA specification for the conditional mean
     * @param garch GARCH specification for the conditional variance
     */
    ArimaGarchSpec(const ArimaSpec& arima, const GarchSpec& garch)
        : arimaSpec(arima), garchSpec(garch) {
        // Both specs are validated in their constructors
    }

    /**
     * @brief Construct an ARIMA-GARCH specification directly from parameters.
     * @param arima_p ARIMA AR order
     * @param arima_d ARIMA differencing degree
     * @param arima_q ARIMA MA order
     * @param garch_p GARCH order
     * @param garch_q ARCH order
     */
    ArimaGarchSpec(int arima_p, int arima_d, int arima_q, int garch_p, int garch_q)
        : arimaSpec(arima_p, arima_d, arima_q), garchSpec(garch_p, garch_q) {}

    /**
     * @brief Get the total number of ARIMA parameters (p + q).
     * @return Number of ARIMA parameters (excluding d which is not estimated)
     */
    [[nodiscard]] constexpr int arimaParamCount() const noexcept {
        return arimaSpec.p + arimaSpec.q;
    }

    /**
     * @brief Get the total number of GARCH parameters (p + q).
     * @return Number of GARCH parameters
     */
    [[nodiscard]] constexpr int garchParamCount() const noexcept {
        return garchSpec.p + garchSpec.q;
    }

    /**
     * @brief Get the total number of model parameters.
     * @return Total number of parameters (ARIMA + GARCH + intercepts)
     */
    [[nodiscard]] constexpr int totalParamCount() const noexcept {
        // ARIMA: p AR params + q MA params + 1 intercept (if not zero-order)
        // GARCH: p GARCH params + q ARCH params + 1 omega (unless null GARCH)
        int arima_total = arimaSpec.isZeroOrder() ? 0 : (arimaSpec.p + arimaSpec.q + 1);
        int garch_total = garchSpec.isNull() ? 0 : (garchSpec.p + garchSpec.q + 1);
        return arima_total + garch_total;
    }
};

}  // namespace ag::models
