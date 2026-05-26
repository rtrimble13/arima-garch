#include "ag/simulation/ArimaGarchSimulator.hpp"

#include "ag/util/NumericConstants.hpp"

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

SimulationResult ArimaGarchSimulator::simulate(int length, unsigned int seed,
                                               InnovationDistribution dist_type,
                                               std::optional<double> df) const {
    if (length <= 0) {
        throw std::invalid_argument("Simulation length must be positive");
    }

    // Validate Student-t parameters
    if (dist_type == InnovationDistribution::StudentT) {
        if (!df.has_value()) {
            throw std::invalid_argument(
                "Degrees of freedom must be specified for Student-t distribution");
        }
        if (df.value() <= 2.0) {
            throw std::invalid_argument(
                "Degrees of freedom must be > 2 for Student-t with finite variance");
        }
    }

    // Initialize result
    SimulationResult result(length);

    // Create innovations generator
    Innovations innovations(seed);

    // Create model instance for this simulation
    ag::models::composite::ArimaGarchModel model(spec_, params_);

    // Simulate path
    for (int t = 0; t < length; ++t) {
        // Draw innovation from specified distribution
        double z_t;
        if (dist_type == InnovationDistribution::Normal) {
            z_t = innovations.drawNormal();
        } else {
            z_t = innovations.drawStudentT(df.value());
        }

        // Compute conditional mean/variance from the model itself rather
        // than reimplementing the recursion here.
        const auto pred = model.predict();
        const double mu_t = pred.mu_t;
        const double h_t = std::max(pred.h_t, ag::util::MIN_VARIANCE);

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
