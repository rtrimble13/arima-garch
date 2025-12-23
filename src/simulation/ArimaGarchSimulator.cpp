#include "ag/simulation/ArimaGarchSimulator.hpp"

#include <cmath>
#include <stdexcept>

namespace ag::simulation {

ArimaGarchSimulator::ArimaGarchSimulator(const ag::models::ArimaGarchSpec& spec,
                                         const ag::models::composite::ArimaGarchParameters& params)
    : spec_(spec), params_(params) {
    // Validate specifications
    spec_.arimaSpec.validate();
    spec_.garchSpec.validate();

    // Validate parameter dimensions
    if (static_cast<int>(params_.arima_params.ar_coef.size()) != spec_.arimaSpec.p) {
        throw std::invalid_argument("ARIMA AR coefficient count must match p");
    }
    if (static_cast<int>(params_.arima_params.ma_coef.size()) != spec_.arimaSpec.q) {
        throw std::invalid_argument("ARIMA MA coefficient count must match q");
    }
    if (static_cast<int>(params_.garch_params.alpha_coef.size()) != spec_.garchSpec.q) {
        throw std::invalid_argument("GARCH ARCH coefficient count must match q");
    }
    if (static_cast<int>(params_.garch_params.beta_coef.size()) != spec_.garchSpec.p) {
        throw std::invalid_argument("GARCH coefficient count must match p");
    }

    // Validate GARCH parameter constraints
    if (!params_.garch_params.isPositive()) {
        throw std::invalid_argument(
            "GARCH parameters must satisfy positivity constraints: omega > 0, alpha >= 0, beta >= "
            "0");
    }
}

SimulationResult ArimaGarchSimulator::simulate(int length, unsigned int seed) const {
    if (length <= 0) {
        throw std::invalid_argument("Simulation length must be positive");
    }

    // Initialize result
    SimulationResult result(length);

    // Create innovations generator
    Innovations innovations(seed);

    // Create model instance for this simulation
    ag::models::composite::ArimaGarchModel model(spec_, params_);

    // Simulate path
    for (int t = 0; t < length; ++t) {
        // Draw standard normal innovation
        double z_t = innovations.drawNormal();

        // Get current conditional mean and variance from model
        // For first few observations, the model will use initialization values
        // We need to "peek" at what the model would predict, then generate y_t

        // Create a temporary observation to get the current state
        // The approach: we'll compute conditional mean directly from current state,
        // then generate y_t, then update the model

        // Get current states before update
        const auto& mean_state = model.getArimaState();
        const auto& var_state = model.getGarchState();

        // Compute conditional mean
        double mu_t = params_.arima_params.intercept;

        const auto& obs_history = mean_state.getObservationHistory();
        for (int i = 0; i < spec_.arimaSpec.p; ++i) {
            mu_t += params_.arima_params.ar_coef[i] * obs_history[spec_.arimaSpec.p - 1 - i];
        }

        const auto& residual_history = mean_state.getResidualHistory();
        for (int i = 0; i < spec_.arimaSpec.q; ++i) {
            mu_t += params_.arima_params.ma_coef[i] * residual_history[spec_.arimaSpec.q - 1 - i];
        }

        // Compute conditional variance
        double h_t = params_.garch_params.omega;

        const auto& var_history = var_state.getVarianceHistory();
        const auto& sq_res_history = var_state.getSquaredResidualHistory();

        for (int i = 0; i < spec_.garchSpec.q; ++i) {
            h_t += params_.garch_params.alpha_coef[i] * sq_res_history[spec_.garchSpec.q - 1 - i];
        }

        for (int i = 0; i < spec_.garchSpec.p; ++i) {
            h_t += params_.garch_params.beta_coef[i] * var_history[spec_.garchSpec.p - 1 - i];
        }

        h_t = std::max(h_t, 1e-10);  // Ensure positive variance

        // Generate return: y_t = μ_t + sqrt(h_t) * z_t
        double y_t = mu_t + std::sqrt(h_t) * z_t;

        // Store results
        result.returns[t] = y_t;
        result.volatilities[t] = std::sqrt(h_t);

        // Update model state with the generated observation
        model.update(y_t);
    }

    return result;
}

}  // namespace ag::simulation
