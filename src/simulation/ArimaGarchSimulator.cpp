#include "ag/simulation/ArimaGarchSimulator.hpp"

#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/util/Differencing.hpp"
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

    // The ARMA/GARCH recursion generates the stationary (differenced) process.
    // Build a d = 0 model for that recursion and integrate the simulated path
    // back to the level scale afterwards (Δ^d inverse, zero initial
    // conditions). For d == 0 the integrator is the identity, so behavior is
    // unchanged.
    ag::models::ArimaSpec stationary_arima(spec_.arimaSpec.p, 0, spec_.arimaSpec.q);
    ag::models::ArimaGarchSpec stationary_spec(stationary_arima, spec_.garchSpec);
    ag::models::composite::ArimaGarchModel model(stationary_spec, params_);

    ag::util::StreamingDifferencer integrator(spec_.arimaSpec.d);

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

        // Generate the differenced-scale value w_t = μ_t + sqrt(h_t) * z_t,
        // then integrate to the level scale for the reported return.
        double w_t = mu_t + std::sqrt(h_t) * z_t;
        double y_t = integrator.integrate(w_t);

        // Store results
        result.returns[t] = y_t;
        result.volatilities[t] = std::sqrt(h_t);

        // Advance the stationary recursion with the differenced value.
        model.update(w_t);
    }

    return result;
}

}  // namespace ag::simulation
