#include "ag/diagnostics/DiagnosticReport.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ag::diagnostics {

DiagnosticReport computeDiagnostics(const ag::models::ArimaGarchSpec& spec,
                                    const ag::models::composite::ArimaGarchParameters& params,
                                    const std::vector<double>& data, std::size_t ljung_box_lags,
                                    bool include_adf) {
    // Validate input
    if (data.empty()) {
        throw std::invalid_argument("Cannot compute diagnostics for empty data");
    }

    if (ljung_box_lags == 0) {
        throw std::invalid_argument("Number of lags for Ljung-Box test must be positive");
    }

    if (ljung_box_lags >= data.size()) {
        throw std::invalid_argument(
            "Number of lags for Ljung-Box test must be less than data size");
    }

    // Step 1: Compute residuals from the fitted model
    ResidualSeries residuals = computeResiduals(spec, params, data);

    // Ensure we have enough data for the tests
    if (residuals.std_eps_t.size() < 4) {
        throw std::invalid_argument("Insufficient data for diagnostic tests (need at least 4)");
    }

    // Step 2: Compute squared residuals for ARCH effect test
    std::vector<double> squared_residuals(residuals.eps_t.size());
    std::transform(residuals.eps_t.begin(), residuals.eps_t.end(), squared_residuals.begin(),
                   [](double x) { return x * x; });

    // Step 3: Calculate degrees of freedom for Ljung-Box tests
    // DOF = lags - number_of_estimated_parameters
    // Use the spec's totalParamCount() which correctly handles zero-order ARIMA models
    std::size_t total_params = static_cast<std::size_t>(spec.totalParamCount());

    // Ensure DOF is positive and meaningful
    // If lags <= total_params, the Ljung-Box test lacks sufficient DOF
    if (ljung_box_lags <= total_params) {
        throw std::invalid_argument(
            "Number of lags for Ljung-Box test must be greater than the number of estimated "
            "parameters (" +
            std::to_string(total_params) + "). Increase lags or use a simpler model.");
    }

    std::size_t dof = ljung_box_lags - total_params;

    // Step 4: Perform Ljung-Box test on residuals
    stats::LjungBoxResult lb_residuals =
        stats::ljung_box_test(residuals.eps_t, ljung_box_lags, dof);

    // Step 5: Perform Ljung-Box test on squared residuals
    stats::LjungBoxResult lb_squared =
        stats::ljung_box_test(squared_residuals, ljung_box_lags, dof);

    // Step 6: Perform Jarque-Bera test on standardized residuals
    stats::JarqueBeraResult jb = stats::jarque_bera_test(residuals.std_eps_t);

    // Step 7: Optionally perform ADF test on residuals
    std::optional<stats::ADFResult> adf_result;
    if (include_adf) {
        // Use automatic lag selection and constant regression form
        adf_result = stats::adf_test(residuals.eps_t, 0, stats::ADFRegressionForm::Constant);
    }

    // Step 8: Construct and return the diagnostic report
    return DiagnosticReport{
        .ljung_box_residuals = lb_residuals,
        .ljung_box_squared = lb_squared,
        .jarque_bera = jb,
        .adf = adf_result,
    };
}

}  // namespace ag::diagnostics
