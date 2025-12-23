#include "ag/models/composite/ArimaGarchModel.hpp"

#include <cmath>
#include <stdexcept>

namespace ag::models::composite {

ArimaGarchModel::ArimaGarchModel(const ag::models::ArimaGarchSpec& spec,
                                 const ArimaGarchParameters& params)
    : spec_(spec), params_(params), arima_model_(spec.arimaSpec), garch_model_(spec.garchSpec),
      mean_state_(spec.arimaSpec.p, spec.arimaSpec.d, spec.arimaSpec.q),
      var_state_(spec.garchSpec.p, spec.garchSpec.q) {
    // Validate specifications
    spec.arimaSpec.validate();
    spec.garchSpec.validate();

    // Validate parameter dimensions match spec
    if (static_cast<int>(params.arima_params.ar_coef.size()) != spec.arimaSpec.p) {
        throw std::invalid_argument("ARIMA AR coefficient count must match p");
    }
    if (static_cast<int>(params.arima_params.ma_coef.size()) != spec.arimaSpec.q) {
        throw std::invalid_argument("ARIMA MA coefficient count must match q");
    }
    if (static_cast<int>(params.garch_params.alpha_coef.size()) != spec.garchSpec.q) {
        throw std::invalid_argument("GARCH ARCH coefficient count must match q");
    }
    if (static_cast<int>(params.garch_params.beta_coef.size()) != spec.garchSpec.p) {
        throw std::invalid_argument("GARCH coefficient count must match p");
    }

    // Validate GARCH parameter constraints
    if (!params.garch_params.isPositive()) {
        throw std::invalid_argument(
            "GARCH parameters must satisfy positivity constraints: omega > 0, alpha >= 0, beta >= "
            "0");
    }

    // Initialize states with empty data (states will be updated as observations arrive)
    // For ARIMA state: initialize with zero history
    std::vector<double> dummy_data(1, 0.0);
    mean_state_.initialize(dummy_data.data(), dummy_data.size());

    // For GARCH state: initialize with unconditional variance if stationary
    std::vector<double> dummy_residuals(1, 0.0);
    double init_variance = params.garch_params.unconditionalVariance();
    var_state_.initialize(dummy_residuals.data(), dummy_residuals.size(), init_variance);
}

ArimaGarchOutput ArimaGarchModel::update(double y_t) {
    // Step 1: Compute conditional mean using ARIMA model
    double mu_t = computeConditionalMean();

    // Step 2: Compute residual
    double eps_t = y_t - mu_t;

    // Step 3: Compute conditional variance using GARCH model
    const auto& var_history = var_state_.getVarianceHistory();
    const auto& sq_res_history = var_state_.getSquaredResidualHistory();

    double h_t = params_.garch_params.omega;

    // Add ARCH component: α₁*ε²_{t-1} + α₂*ε²_{t-2} + ... + α_q*ε²_{t-q}
    for (int i = 0; i < spec_.garchSpec.q; ++i) {
        h_t += params_.garch_params.alpha_coef[i] * sq_res_history[spec_.garchSpec.q - 1 - i];
    }

    // Add GARCH component: β₁*h_{t-1} + β₂*h_{t-2} + ... + βₚ*h_{t-p}
    for (int i = 0; i < spec_.garchSpec.p; ++i) {
        h_t += params_.garch_params.beta_coef[i] * var_history[spec_.garchSpec.p - 1 - i];
    }

    // Ensure h_t > 0 (should be guaranteed by positive parameters, but guard against numerical
    // issues)
    h_t = std::max(h_t, 1e-10);

    // Step 4: Update ARIMA state
    mean_state_.update(y_t, eps_t);

    // Step 5: Update GARCH state
    double eps_squared = eps_t * eps_t;
    var_state_.update(h_t, eps_squared);

    // Return output
    return ArimaGarchOutput{mu_t, h_t};
}

double ArimaGarchModel::computeConditionalMean() const {
    double mean = params_.arima_params.intercept;

    // Add AR component: φ₁*y_{t-1} + φ₂*y_{t-2} + ... + φₚ*y_{t-p}
    const auto& obs_history = mean_state_.getObservationHistory();
    for (int i = 0; i < spec_.arimaSpec.p; ++i) {
        // History is stored with oldest first, so obs_history[0] is y_{t-p}
        // and obs_history[p-1] is y_{t-1}
        mean += params_.arima_params.ar_coef[i] * obs_history[spec_.arimaSpec.p - 1 - i];
    }

    // Add MA component: θ₁*ε_{t-1} + θ₂*ε_{t-2} + ... + θ_q*ε_{t-q}
    const auto& residual_history = mean_state_.getResidualHistory();
    for (int i = 0; i < spec_.arimaSpec.q; ++i) {
        // History is stored with oldest first, so residual_history[0] is ε_{t-q}
        // and residual_history[q-1] is ε_{t-1}
        mean += params_.arima_params.ma_coef[i] * residual_history[spec_.arimaSpec.q - 1 - i];
    }

    return mean;
}

}  // namespace ag::models::composite
