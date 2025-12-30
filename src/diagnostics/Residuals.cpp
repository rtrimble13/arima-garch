#include "ag/diagnostics/Residuals.hpp"

#include "ag/models/arima/ArimaState.hpp"
#include "ag/models/garch/GarchState.hpp"

#include <cmath>
#include <stdexcept>

namespace ag::diagnostics {

ResidualSeries computeResiduals(const ag::models::ArimaGarchSpec& spec,
                                const ag::models::composite::ArimaGarchParameters& params,
                                const double* data, std::size_t size) {
    // Validate inputs
    if (data == nullptr) {
        throw std::invalid_argument("Data pointer cannot be null");
    }
    if (size == 0) {
        throw std::invalid_argument("Data size must be greater than 0");
    }

    // Create the ARIMA-GARCH model (this validates parameters)
    ag::models::composite::ArimaGarchModel model(spec, params);

    // Allocate output vectors
    ResidualSeries result;
    result.eps_t.reserve(size);
    result.h_t.reserve(size);
    result.std_eps_t.reserve(size);

    // Filter each observation through the model
    for (std::size_t t = 0; t < size; ++t) {
        // Step 1: Predict conditional mean and variance using only past information
        auto output = model.predict();

        // Step 2: Compute residual: eps_t = y_t - mu_t
        double eps_t = data[t] - output.mu_t;

        // Step 3: Get conditional variance: h_t
        double h_t = output.h_t;

        // Step 4: Compute standardized residual: std_eps_t = eps_t / sqrt(h_t)
        // h_t should always be > 0 for valid GARCH parameters
        if (h_t <= 0.0) {
            throw std::runtime_error("Invalid conditional variance h_t <= 0 detected");
        }
        double std_eps_t = eps_t / std::sqrt(h_t);

        // Verify no NaNs or Infs in results
        if (!std::isfinite(eps_t) || !std::isfinite(h_t) || !std::isfinite(std_eps_t)) {
            throw std::runtime_error("Non-finite value detected in residual computation");
        }

        // Store results
        result.eps_t.push_back(eps_t);
        result.h_t.push_back(h_t);
        result.std_eps_t.push_back(std_eps_t);

        // Step 5: Manually update model state for next iteration
        // Get references to the state objects (non-const via const_cast since we need to update)
        auto& mean_state = const_cast<ag::models::arima::ArimaState&>(model.getArimaState());
        auto& var_state = const_cast<ag::models::garch::GarchState&>(model.getGarchState());

        // Update ARIMA state with current observation and residual
        mean_state.update(data[t], eps_t);

        // Update GARCH state with current variance and squared residual
        if (!spec.garchSpec.isNull()) {
            double eps_squared = eps_t * eps_t;
            var_state.update(h_t, eps_squared);
        }
    }

    return result;
}

}  // namespace ag::diagnostics
