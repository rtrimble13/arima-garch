#include "ag/diagnostics/Residuals.hpp"

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
        // Update model with current observation
        auto output = model.update(data[t]);

        // Compute residual: eps_t = y_t - mu_t
        double eps_t = data[t] - output.mu_t;

        // Get conditional variance: h_t
        double h_t = output.h_t;

        // Compute standardized residual: std_eps_t = eps_t / sqrt(h_t)
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
    }

    return result;
}

}  // namespace ag::diagnostics
