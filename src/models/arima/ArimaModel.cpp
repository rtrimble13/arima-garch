#include "ag/models/arima/ArimaModel.hpp"

#include <stdexcept>

namespace ag::models::arima {

ArimaModel::ArimaModel(const ag::models::ArimaSpec& spec) : spec_(spec) {
    spec.validate();
}

std::vector<double> ArimaModel::computeResiduals(const double* data, std::size_t size,
                                                 const ArimaParameters& params) const {
    if (data == nullptr) {
        throw std::invalid_argument("Data pointer cannot be null");
    }
    if (size == 0) {
        throw std::invalid_argument("Data size must be positive");
    }

    // Validate parameter dimensions
    if (static_cast<int>(params.ar_coef.size()) != spec_.p) {
        throw std::invalid_argument("AR coefficient count must match p");
    }
    if (static_cast<int>(params.ma_coef.size()) != spec_.q) {
        throw std::invalid_argument("MA coefficient count must match q");
    }

    // Initialize state
    ArimaState state(spec_.p, spec_.d, spec_.q);
    state.initialize(data, size);

    // Determine the working series (original or differenced)
    const double* working_data = data;
    std::size_t working_size = size;

    if (spec_.d > 0) {
        const auto& diff_series = state.getDifferencedSeries();
        working_data = diff_series.data();
        working_size = diff_series.size();
    }

    // Reserve space for residuals
    std::vector<double> residuals;
    residuals.reserve(working_size);

    // Compute residuals using recursion
    for (std::size_t t = 0; t < working_size; ++t) {
        // Compute conditional mean for this observation
        double conditional_mean = computeConditionalMean(state, params);

        // Compute residual: ε_t = y_t - E[y_t]
        double residual = working_data[t] - conditional_mean;
        residuals.push_back(residual);

        // Update state for next iteration
        state.update(working_data[t], residual);
    }

    return residuals;
}

double ArimaModel::computeConditionalMean(const ArimaState& state,
                                          const ArimaParameters& params) const {
    double mean = params.intercept;

    // Add AR component: φ₁*y_{t-1} + φ₂*y_{t-2} + ... + φₚ*y_{t-p}
    const auto& obs_history = state.getObservationHistory();
    for (int i = 0; i < spec_.p; ++i) {
        // History is stored with oldest first, so obs_history[0] is y_{t-p}
        // and obs_history[p-1] is y_{t-1}
        mean += params.ar_coef[i] * obs_history[spec_.p - 1 - i];
    }

    // Add MA component: θ₁*ε_{t-1} + θ₂*ε_{t-2} + ... + θ_q*ε_{t-q}
    const auto& residual_history = state.getResidualHistory();
    for (int i = 0; i < spec_.q; ++i) {
        // History is stored with oldest first, so residual_history[0] is ε_{t-q}
        // and residual_history[q-1] is ε_{t-1}
        mean += params.ma_coef[i] * residual_history[spec_.q - 1 - i];
    }

    return mean;
}

}  // namespace ag::models::arima
