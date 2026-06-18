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

    // The first d observations only prime the differencing pipeline and yield
    // no innovation, so residuals exist for size - d observations (matching the
    // likelihood, which works on the differenced series).
    const std::size_t d = static_cast<std::size_t>(spec.arimaSpec.d);

    // Allocate output vectors
    ResidualSeries result;
    const std::size_t n_resid = (size > d) ? (size - d) : 0;
    result.eps_t.reserve(n_resid);
    result.h_t.reserve(n_resid);
    result.std_eps_t.reserve(n_resid);

    // Filter each observation through the d-aware model. update() computes the
    // conditional mean from past information only (then advances the state), so
    // eps_t = y_t - mu_t carries no look-ahead bias and is the differenced-scale
    // innovation.
    for (std::size_t t = 0; t < size; ++t) {
        auto output = model.update(data[t]);

        // Skip the priming observations (no innovation defined).
        if (t < d) {
            continue;
        }

        double eps_t = data[t] - output.mu_t;
        double h_t = output.h_t;

        // h_t should always be > 0 for valid GARCH parameters.
        if (h_t <= 0.0) {
            throw std::runtime_error("Invalid conditional variance h_t <= 0 detected");
        }
        double std_eps_t = eps_t / std::sqrt(h_t);

        // Verify no NaNs or Infs in results
        if (!std::isfinite(eps_t) || !std::isfinite(h_t) || !std::isfinite(std_eps_t)) {
            throw std::runtime_error("Non-finite value detected in residual computation");
        }

        result.eps_t.push_back(eps_t);
        result.h_t.push_back(h_t);
        result.std_eps_t.push_back(std_eps_t);
    }

    return result;
}

}  // namespace ag::diagnostics
