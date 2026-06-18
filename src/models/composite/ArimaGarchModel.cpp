#include "ag/models/composite/ArimaGarchModel.hpp"

#include "ag/util/NumericConstants.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ag::models::composite {

ArimaGarchModel::ArimaGarchModel(const ag::models::ArimaGarchSpec& spec,
                                 const ArimaGarchParameters& params)
    : spec_(spec), params_(params), arima_model_(spec.arimaSpec), garch_model_(spec.garchSpec),
      // The ARMA recursion operates on the already-differenced series, so the
      // mean state carries no differencing of its own (d = 0); differencing is
      // owned by differencer_ below.
      mean_state_(spec.arimaSpec.p, 0, spec.arimaSpec.q),
      var_state_(spec.garchSpec.p, spec.garchSpec.q), differencer_(spec.arimaSpec.d) {
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
    // Step 0: Difference the raw level. The ARMA/GARCH recursion runs on the
    // differenced (stationary) series w_t = Δ^d y_t, matching the fit. While
    // the differencing pipeline is priming (the first d observations) there is
    // no innovation yet: just consume the level and report the current state.
    double w_t = y_t;
    if (!differencer_.difference(y_t, w_t)) {
        double h_t = computeConditionalVariance();
        return ArimaGarchOutput{y_t, h_t};
    }

    // Step 1: Conditional mean on the differenced scale (μ_w).
    double mu_w = computeConditionalMean();

    // Step 2: Innovation on the differenced scale.
    double eps_t = w_t - mu_w;

    // Step 3: Conditional variance.
    double h_t = computeConditionalVariance();

    // Step 4: Update the ARIMA state with the differenced observation.
    mean_state_.update(w_t, eps_t);

    // Step 5: Update GARCH state (only if a GARCH component exists).
    if (!spec_.garchSpec.isNull()) {
        double eps_squared = eps_t * eps_t;
        var_state_.update(h_t, eps_squared);
    }

    // The conditional mean on the original level scale satisfies
    // y_t - mu_level = eps_t, so re-integrate by mu_level = y_t - eps_t.
    return ArimaGarchOutput{y_t - eps_t, h_t};
}

ArimaGarchOutput ArimaGarchModel::predict() const {
    // Compute conditional mean and variance using current state
    // without modifying the state
    double mu_t = computeConditionalMean();
    double h_t = computeConditionalVariance();

    return ArimaGarchOutput{mu_t, h_t};
}

double ArimaGarchModel::computeConditionalMean() const {
    return arima_model_.computeConditionalMean(mean_state_, params_.arima_params);
}

double ArimaGarchModel::computeConditionalVariance() const {
    if (spec_.garchSpec.isNull()) {
        // ARIMA-only models use a constant variance held in the state.
        return var_state_.getInitialVariance();
    }
    return std::max(garch_model_.computeConditionalVariance(var_state_, params_.garch_params),
                    ag::util::MIN_VARIANCE);
}

}  // namespace ag::models::composite
