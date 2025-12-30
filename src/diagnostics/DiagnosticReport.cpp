#include "ag/diagnostics/DiagnosticReport.hpp"
#include "ag/stats/Bootstrap.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ag::diagnostics {

DiagnosticReport computeDiagnostics(const ag::models::ArimaGarchSpec& spec,
                                    const ag::models::composite::ArimaGarchParameters& params,
                                    const std::vector<double>& data, std::size_t ljung_box_lags,
                                    bool include_adf, const std::string& innovation_dist,
                                    double student_t_df, bool force_bootstrap,
                                    std::size_t n_bootstrap, unsigned int bootstrap_seed) {
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

    // Validate innovation distribution parameters
    bool use_student_t = (innovation_dist == "Student-t" || innovation_dist == "student-t" ||
                          innovation_dist == "StudentT");
    if (use_student_t && student_t_df <= 2.0) {
        throw std::invalid_argument(
            "Student-t degrees of freedom must be > 2.0 for finite variance");
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
    std::size_t total_params = static_cast<std::size_t>(spec.totalParamCount());

    if (ljung_box_lags <= total_params) {
        throw std::invalid_argument(
            "Number of lags for Ljung-Box test must be greater than the number of estimated "
            "parameters (" +
            std::to_string(total_params) + "). Increase lags or use a simpler model.");
    }

    std::size_t dof = ljung_box_lags - total_params;

    // Step 4: Determine whether to use bootstrap methods
    // Bootstrap is used if:
    // 1. Explicitly forced via force_bootstrap, OR
    // 2. Student-t distribution with df < 30 (heavy tails)
    bool use_bootstrap = force_bootstrap || (use_student_t && student_t_df < 30.0);
    std::string ljung_box_method = use_bootstrap ? "bootstrap" : "asymptotic";
    std::string adf_method = use_bootstrap ? "bootstrap" : "asymptotic";

    // Step 5: Perform Ljung-Box test on residuals
    stats::LjungBoxResult lb_residuals;
    if (use_bootstrap) {
        lb_residuals =
            stats::ljung_box_test_bootstrap(residuals.eps_t, ljung_box_lags, n_bootstrap,
                                             bootstrap_seed);
        // Adjust dof to match the asymptotic convention
        lb_residuals.dof = dof;
    } else {
        lb_residuals = stats::ljung_box_test(residuals.eps_t, ljung_box_lags, dof);
    }

    // Step 6: Perform Ljung-Box test on squared residuals
    stats::LjungBoxResult lb_squared;
    if (use_bootstrap) {
        lb_squared = stats::ljung_box_test_bootstrap(squared_residuals, ljung_box_lags,
                                                      n_bootstrap, bootstrap_seed + 1);
        // Adjust dof to match the asymptotic convention
        lb_squared.dof = dof;
    } else {
        lb_squared = stats::ljung_box_test(squared_residuals, ljung_box_lags, dof);
    }

    // Step 7: Perform Jarque-Bera test on standardized residuals
    stats::JarqueBeraResult jb = stats::jarque_bera_test(residuals.std_eps_t);

    // Step 8: Optionally perform ADF test on residuals
    std::optional<stats::ADFResult> adf_result;
    if (include_adf) {
        if (use_bootstrap) {
            // Use bootstrap ADF with automatic lag selection (lags=0)
            adf_result = stats::adf_test_bootstrap(residuals.eps_t, 0,
                                                    stats::ADFRegressionForm::Constant,
                                                    n_bootstrap, bootstrap_seed + 2);
        } else {
            // Use asymptotic ADF with automatic lag selection
            adf_result = stats::adf_test(residuals.eps_t, 0, stats::ADFRegressionForm::Constant);
        }
    }

    // Step 9: Construct and return the diagnostic report
    DiagnosticReport report{
        .ljung_box_residuals = lb_residuals,
        .ljung_box_squared = lb_squared,
        .jarque_bera = jb,
        .adf = adf_result,
        .ljung_box_method = ljung_box_method,
        .adf_method = adf_method,
    };

    // Set innovation distribution info if provided
    if (!innovation_dist.empty() && innovation_dist != "Normal") {
        report.innovation_distribution = innovation_dist;
        if (use_student_t) {
            report.student_t_df = student_t_df;
        }
    }

    return report;
}

}  // namespace ag::diagnostics
