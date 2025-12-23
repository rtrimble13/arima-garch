#include "ag/report/FitSummary.hpp"
#include "ag/diagnostics/DiagnosticReport.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"

#include <cmath>
#include <iostream>
#include <string>

#include "test_framework.hpp"

using ag::models::ArimaGarchSpec;
using ag::models::composite::ArimaGarchParameters;
using ag::report::FitSummary;
using ag::report::generateTextReport;

// ============================================================================
// Basic FitSummary Construction Tests
// ============================================================================

TEST(fit_summary_construction) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    FitSummary summary(spec);

    // Verify spec is copied correctly
    REQUIRE(summary.spec.arimaSpec.p == 1);
    REQUIRE(summary.spec.arimaSpec.d == 0);
    REQUIRE(summary.spec.arimaSpec.q == 1);
    REQUIRE(summary.spec.garchSpec.p == 1);
    REQUIRE(summary.spec.garchSpec.q == 1);

    // Verify default initialization
    REQUIRE(summary.neg_log_likelihood == 0.0);
    REQUIRE(summary.aic == 0.0);
    REQUIRE(summary.bic == 0.0);
    REQUIRE(summary.converged == false);
    REQUIRE(summary.iterations == 0);
    REQUIRE(summary.sample_size == 0);
    REQUIRE(!summary.diagnostics.has_value());
}

TEST(fit_summary_with_parameters) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    FitSummary summary(spec);

    // Set parameters
    summary.parameters.arima_params.intercept = 0.05;
    summary.parameters.arima_params.ar_coef[0] = 0.6;
    summary.parameters.arima_params.ma_coef[0] = 0.3;
    summary.parameters.garch_params.omega = 0.01;
    summary.parameters.garch_params.alpha_coef[0] = 0.1;
    summary.parameters.garch_params.beta_coef[0] = 0.85;

    // Set convergence info
    summary.converged = true;
    summary.iterations = 150;
    summary.message = "Converged";
    summary.sample_size = 1000;

    // Set information criteria
    summary.neg_log_likelihood = 500.0;
    summary.aic = 1012.0;
    summary.bic = 1048.0;

    // Verify all fields are set correctly
    REQUIRE(summary.parameters.arima_params.intercept == 0.05);
    REQUIRE(summary.parameters.arima_params.ar_coef[0] == 0.6);
    REQUIRE(summary.converged == true);
    REQUIRE(summary.iterations == 150);
    REQUIRE(summary.sample_size == 1000);
    REQUIRE(summary.neg_log_likelihood == 500.0);
}

// ============================================================================
// Text Report Generation Tests
// ============================================================================

TEST(generate_text_report_basic) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    FitSummary summary(spec);

    summary.parameters.arima_params.intercept = 0.05;
    summary.parameters.arima_params.ar_coef[0] = 0.6;
    summary.parameters.arima_params.ma_coef[0] = 0.3;
    summary.parameters.garch_params.omega = 0.01;
    summary.parameters.garch_params.alpha_coef[0] = 0.1;
    summary.parameters.garch_params.beta_coef[0] = 0.85;

    summary.converged = true;
    summary.iterations = 150;
    summary.message = "Converged";
    summary.sample_size = 1000;
    summary.neg_log_likelihood = 500.0;
    summary.aic = 1012.0;
    summary.bic = 1048.0;

    std::string report = generateTextReport(summary);

    // Verify report contains key sections
    REQUIRE(report.find("ARIMA-GARCH Model Fit Summary") != std::string::npos);
    REQUIRE(report.find("Model Specification") != std::string::npos);
    REQUIRE(report.find("Estimated Parameters") != std::string::npos);
    REQUIRE(report.find("Convergence Information") != std::string::npos);
    REQUIRE(report.find("Model Fit Statistics") != std::string::npos);

    // Verify model specification appears in report
    REQUIRE(report.find("ARIMA order:        (1,0,1)") != std::string::npos);
    REQUIRE(report.find("GARCH order:        (1,1)") != std::string::npos);
    REQUIRE(report.find("Sample size:        1000") != std::string::npos);

    // Verify convergence info
    REQUIRE(report.find("Converged") != std::string::npos);
    REQUIRE(report.find("Iterations:         150") != std::string::npos);

    // Verify information criteria
    REQUIRE(report.find("AIC:") != std::string::npos);
    REQUIRE(report.find("BIC:") != std::string::npos);
    REQUIRE(report.find("Log-likelihood:") != std::string::npos);
}

TEST(generate_text_report_with_parameters) {
    ArimaGarchSpec spec(2, 0, 2, 1, 1);
    FitSummary summary(spec);

    // Set multiple parameters
    summary.parameters.arima_params.intercept = 0.1;
    summary.parameters.arima_params.ar_coef[0] = 0.5;
    summary.parameters.arima_params.ar_coef[1] = 0.3;
    summary.parameters.arima_params.ma_coef[0] = 0.2;
    summary.parameters.arima_params.ma_coef[1] = 0.1;
    summary.parameters.garch_params.omega = 0.05;
    summary.parameters.garch_params.alpha_coef[0] = 0.15;
    summary.parameters.garch_params.beta_coef[0] = 0.80;

    summary.converged = true;
    summary.iterations = 200;
    summary.message = "Converged";
    summary.sample_size = 500;
    summary.neg_log_likelihood = 300.0;
    summary.aic = 616.0;
    summary.bic = 655.0;

    std::string report = generateTextReport(summary);

    // Verify AR coefficients appear
    REQUIRE(report.find("AR coefficients:") != std::string::npos);
    REQUIRE(report.find("0.500000") != std::string::npos);  // AR[0]
    REQUIRE(report.find("0.300000") != std::string::npos);  // AR[1]

    // Verify MA coefficients appear
    REQUIRE(report.find("MA coefficients:") != std::string::npos);
    REQUIRE(report.find("0.200000") != std::string::npos);  // MA[0]
    REQUIRE(report.find("0.100000") != std::string::npos);  // MA[1]

    // Verify GARCH parameters
    REQUIRE(report.find("Omega:") != std::string::npos);
    REQUIRE(report.find("ARCH (alpha):") != std::string::npos);
    REQUIRE(report.find("GARCH (beta):") != std::string::npos);
}

TEST(generate_text_report_white_noise) {
    // Test with white noise model (no AR/MA terms)
    ArimaGarchSpec spec(0, 0, 0, 1, 1);
    FitSummary summary(spec);

    summary.parameters.arima_params.intercept = 0.0;
    summary.parameters.garch_params.omega = 0.1;
    summary.parameters.garch_params.alpha_coef[0] = 0.1;
    summary.parameters.garch_params.beta_coef[0] = 0.8;

    summary.converged = true;
    summary.iterations = 100;
    summary.message = "Converged";
    summary.sample_size = 500;
    summary.neg_log_likelihood = 250.0;
    summary.aic = 506.0;
    summary.bic = 525.0;

    std::string report = generateTextReport(summary);

    // For white noise, AR/MA sections should not appear (or be empty)
    // Just verify key sections exist
    REQUIRE(report.find("ARIMA order:        (0,0,0)") != std::string::npos);
    REQUIRE(report.find("GARCH order:        (1,1)") != std::string::npos);
    REQUIRE(report.find("Intercept:") != std::string::npos);
}

// ============================================================================
// Text Report with Diagnostics Tests
// ============================================================================

TEST(generate_text_report_with_diagnostics) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    FitSummary summary(spec);

    summary.parameters.arima_params.intercept = 0.05;
    summary.parameters.arima_params.ar_coef[0] = 0.6;
    summary.parameters.arima_params.ma_coef[0] = 0.3;
    summary.parameters.garch_params.omega = 0.01;
    summary.parameters.garch_params.alpha_coef[0] = 0.1;
    summary.parameters.garch_params.beta_coef[0] = 0.85;

    summary.converged = true;
    summary.iterations = 150;
    summary.message = "Converged";
    summary.sample_size = 1000;
    summary.neg_log_likelihood = 500.0;
    summary.aic = 1012.0;
    summary.bic = 1048.0;

    // Create diagnostic report
    ag::diagnostics::DiagnosticReport diag;
    diag.ljung_box_residuals.lags = 10;
    diag.ljung_box_residuals.dof = 4;
    diag.ljung_box_residuals.statistic = 8.5;
    diag.ljung_box_residuals.p_value = 0.15;

    diag.ljung_box_squared.lags = 10;
    diag.ljung_box_squared.dof = 7;
    diag.ljung_box_squared.statistic = 5.2;
    diag.ljung_box_squared.p_value = 0.25;

    diag.jarque_bera.statistic = 2.5;
    diag.jarque_bera.p_value = 0.30;

    summary.diagnostics = diag;

    std::string report = generateTextReport(summary);

    // Verify diagnostic section appears
    REQUIRE(report.find("Diagnostic Tests") != std::string::npos);
    REQUIRE(report.find("Ljung-Box Test on Residuals") != std::string::npos);
    REQUIRE(report.find("Ljung-Box Test on Squared Residuals") != std::string::npos);
    REQUIRE(report.find("Jarque-Bera Test for Normality") != std::string::npos);

    // Verify diagnostic values appear
    REQUIRE(report.find("Lags:           10") != std::string::npos);
    REQUIRE(report.find("P-value:        0.15") != std::string::npos);
    REQUIRE(report.find("P-value:        0.25") != std::string::npos);
    REQUIRE(report.find("P-value:        0.30") != std::string::npos);

    // Verify interpretation guidance appears
    REQUIRE(report.find("Interpretation:") != std::string::npos);
}

TEST(generate_text_report_without_diagnostics) {
    ArimaGarchSpec spec(1, 0, 0, 1, 1);
    FitSummary summary(spec);

    summary.parameters.arima_params.intercept = 0.02;
    summary.parameters.arima_params.ar_coef[0] = 0.7;
    summary.parameters.garch_params.omega = 0.02;
    summary.parameters.garch_params.alpha_coef[0] = 0.15;
    summary.parameters.garch_params.beta_coef[0] = 0.80;

    summary.converged = true;
    summary.iterations = 120;
    summary.message = "Converged";
    summary.sample_size = 800;
    summary.neg_log_likelihood = 400.0;
    summary.aic = 810.0;
    summary.bic = 835.0;

    // No diagnostics
    summary.diagnostics = std::nullopt;

    std::string report = generateTextReport(summary);

    // Verify diagnostic section does NOT appear
    REQUIRE(report.find("Diagnostic Tests") == std::string::npos);
    REQUIRE(report.find("Ljung-Box") == std::string::npos);
    REQUIRE(report.find("Jarque-Bera") == std::string::npos);
}

// ============================================================================
// Information Criteria Tests
// ============================================================================

TEST(fit_summary_information_criteria) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    FitSummary summary(spec);

    // Verify total parameter count
    std::size_t k = spec.totalParamCount();
    REQUIRE(k == 6);  // intercept + AR(1) + MA(1) + omega + alpha + beta

    // Set sample size and NLL
    summary.sample_size = 1000;
    summary.neg_log_likelihood = 500.0;

    // Compute AIC and BIC
    summary.aic = 2.0 * k + 2.0 * summary.neg_log_likelihood;
    summary.bic = k * std::log(summary.sample_size) + 2.0 * summary.neg_log_likelihood;

    // Verify AIC = 2*6 + 2*500 = 12 + 1000 = 1012
    REQUIRE_APPROX(summary.aic, 1012.0, 0.001);

    // Verify BIC = 6*log(1000) + 2*500 = 6*6.907... + 1000 ≈ 1041.45
    REQUIRE_APPROX(summary.bic, 1041.45, 0.1);
}

// ============================================================================
// Main test runner
// ============================================================================

int main() {
    report_test_results("Report: FitSummary");
    return get_test_result();
}
