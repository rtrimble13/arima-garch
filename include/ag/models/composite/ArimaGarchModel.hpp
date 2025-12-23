#pragma once

#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/arima/ArimaModel.hpp"
#include "ag/models/arima/ArimaState.hpp"
#include "ag/models/garch/GarchModel.hpp"
#include "ag/models/garch/GarchState.hpp"

#include <cstddef>

namespace ag::models::composite {

/**
 * @brief Combined parameters for an ARIMA-GARCH model.
 *
 * Contains both ARIMA parameters (for conditional mean) and GARCH parameters
 * (for conditional variance).
 */
struct ArimaGarchParameters {
    arima::ArimaParameters arima_params;  // ARIMA parameters (intercept, AR, MA)
    garch::GarchParameters garch_params;  // GARCH parameters (omega, alpha, beta)

    /**
     * @brief Construct ARIMA-GARCH parameters with given specification.
     * @param spec ARIMA-GARCH specification
     */
    explicit ArimaGarchParameters(const ag::models::ArimaGarchSpec& spec)
        : arima_params(spec.arimaSpec.p, spec.arimaSpec.q),
          garch_params(spec.garchSpec.p, spec.garchSpec.q) {}
};

/**
 * @brief Output from a single ARIMA-GARCH model update.
 *
 * Contains the conditional mean (mu_t) and conditional variance (h_t)
 * produced by the model for a single observation.
 */
struct ArimaGarchOutput {
    double mu_t;  // Conditional mean at time t
    double h_t;   // Conditional variance at time t
};

/**
 * @brief ARIMA-GARCH model that encapsulates fitted parameters and state.
 *
 * ArimaGarchModel combines an ARIMA model for the conditional mean with a
 * GARCH model for the conditional variance. It maintains the state for both
 * components and provides an update() method for sequential processing of
 * time series observations.
 *
 * The model follows this two-step process:
 * 1. ARIMA component computes conditional mean μ_t and residual ε_t
 * 2. GARCH component computes conditional variance h_t from residuals
 *
 * The update() method processes a single new observation, updates both states,
 * and returns μ_t and h_t for that observation.
 */
class ArimaGarchModel {
public:
    /**
     * @brief Construct an ARIMA-GARCH model with given specification and parameters.
     * @param spec ARIMA-GARCH specification
     * @param params ARIMA-GARCH parameters (fitted coefficients)
     */
    ArimaGarchModel(const ag::models::ArimaGarchSpec& spec, const ArimaGarchParameters& params);

    /**
     * @brief Update the model with a new observation.
     *
     * This method processes a single new observation y_t through the ARIMA-GARCH
     * model:
     * 1. Computes the conditional mean μ_t using the ARIMA component
     * 2. Computes the residual ε_t = y_t - μ_t
     * 3. Computes the conditional variance h_t using the GARCH component
     * 4. Updates both ARIMA and GARCH states for the next observation
     *
     * The method is designed for sequential processing without reallocation,
     * making it efficient for online/streaming applications.
     *
     * @param y_t The new observation at time t
     * @return ArimaGarchOutput containing μ_t and h_t
     */
    ArimaGarchOutput update(double y_t);

    /**
     * @brief Get the ARIMA-GARCH specification for this model.
     * @return The ARIMA-GARCH specification
     */
    [[nodiscard]] const ag::models::ArimaGarchSpec& getSpec() const noexcept { return spec_; }

    /**
     * @brief Get the ARIMA parameters.
     * @return Reference to ARIMA parameters
     */
    [[nodiscard]] const arima::ArimaParameters& getArimaParams() const noexcept {
        return params_.arima_params;
    }

    /**
     * @brief Get the GARCH parameters.
     * @return Reference to GARCH parameters
     */
    [[nodiscard]] const garch::GarchParameters& getGarchParams() const noexcept {
        return params_.garch_params;
    }

    /**
     * @brief Get the current ARIMA state.
     * @return Reference to ARIMA state
     */
    [[nodiscard]] const arima::ArimaState& getArimaState() const noexcept { return mean_state_; }

    /**
     * @brief Get the current GARCH state.
     * @return Reference to GARCH state
     */
    [[nodiscard]] const garch::GarchState& getGarchState() const noexcept { return var_state_; }

private:
    ag::models::ArimaGarchSpec spec_;  // Model specification
    ArimaGarchParameters params_;      // Model parameters (fitted coefficients)
    arima::ArimaModel arima_model_;    // ARIMA model instance
    garch::GarchModel garch_model_;    // GARCH model instance
    arima::ArimaState mean_state_;     // State for ARIMA recursion
    garch::GarchState var_state_;      // State for GARCH recursion

    /**
     * @brief Compute conditional mean for the current observation.
     *
     * Uses the ARIMA model to compute the expected value based on historical
     * observations and residuals.
     *
     * @return The conditional mean μ_t
     */
    [[nodiscard]] double computeConditionalMean() const;
};

}  // namespace ag::models::composite
