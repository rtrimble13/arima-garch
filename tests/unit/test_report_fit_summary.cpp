#include "ag/diagnostics/DiagnosticReport.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/report/FitSummary.hpp"
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
// Distribution Comparison Tests
// ============================================================================

TEST(fit_summary_distribution_comparison_construction) {
    using ag::report::DistributionComparison;

    DistributionComparison dc;
    dc.normal_log_likelihood = -500.0;
    dc.student_t_log_likelihood = -480.0;
    dc.student_t_df = 5.0;
    dc.lr_statistic = 40.0;
    dc.lr_p_value = 0.001;
    dc.prefer_student_t = true;
    dc.normal_aic = 1012.0;
    dc.student_t_aic = 974.0;
    dc.normal_bic = 1041.45;
    dc.student_t_bic = 1007.0;

    // Verify values are set correctly
    REQUIRE(dc.normal_log_likelihood == -500.0);
    REQUIRE(dc.student_t_log_likelihood == -480.0);
    REQUIRE(dc.student_t_df == 5.0);
    REQUIRE(dc.lr_statistic == 40.0);
    REQUIRE(dc.lr_p_value == 0.001);
    REQUIRE(dc.prefer_student_t == true);
}

TEST(fit_summary_with_distribution_comparison) {
    using ag::report::DistributionComparison;

    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    FitSummary summary(spec);

    // Populate basic summary fields
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

    // Create and add distribution comparison
    DistributionComparison dc;
    dc.normal_log_likelihood = -500.0;
    dc.student_t_log_likelihood = -480.0;
    dc.student_t_df = 5.0;
    dc.lr_statistic = 40.0;
    dc.lr_p_value = 0.001;
    dc.prefer_student_t = true;
    dc.normal_aic = 1012.0;
    dc.student_t_aic = 974.0;
    dc.normal_bic = 1048.0;
    dc.student_t_bic = 1011.0;

    summary.distribution_comparison = dc;

    // Verify distribution comparison is set
    REQUIRE(summary.distribution_comparison.has_value());
    REQUIRE(summary.distribution_comparison->prefer_student_t == true);
    REQUIRE(summary.distribution_comparison->student_t_df == 5.0);
}

TEST(generate_text_report_with_distribution_comparison) {
    using ag::report::DistributionComparison;

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

    // Add distribution comparison
    DistributionComparison dc;
    dc.normal_log_likelihood = -500.0;
    dc.student_t_log_likelihood = -480.0;
    dc.student_t_df = 5.5;
    dc.lr_statistic = 40.0;
    dc.lr_p_value = 0.001;
    dc.prefer_student_t = true;
    dc.normal_aic = 1012.0;
    dc.student_t_aic = 974.0;
    dc.normal_bic = 1048.0;
    dc.student_t_bic = 1011.0;

    summary.distribution_comparison = dc;

    std::string report = generateTextReport(summary);

    // Verify distribution comparison section appears
    REQUIRE(report.find("Innovation Distribution Comparison") != std::string::npos);
    REQUIRE(report.find("Gaussian log-likelihood:") != std::string::npos);
    REQUIRE(report.find("Student-t log-likelihood:") != std::string::npos);
    REQUIRE(report.find("Likelihood Ratio Test:") != std::string::npos);

    // Verify specific values appear
    REQUIRE(report.find("-500.0000") != std::string::npos);  // Normal LL
    REQUIRE(report.find("-480.0000") != std::string::npos);  // Student-T LL
    REQUIRE(report.find("5.50") != std::string::npos);       // df
    REQUIRE(report.find("40.0000") != std::string::npos);    // LR statistic
    REQUIRE(report.find("0.0010") != std::string::npos);     // p-value

    // Verify recommendation appears
    REQUIRE(report.find("RECOMMENDATION:") != std::string::npos);
    REQUIRE(report.find("Student-t distribution provides better fit") != std::string::npos);
}

TEST(generate_text_report_distribution_comparison_prefer_normal) {
    using ag::report::DistributionComparison;

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

    // Add distribution comparison where Normal is adequate
    DistributionComparison dc;
    dc.normal_log_likelihood = -500.0;
    dc.student_t_log_likelihood = -498.0;
    dc.student_t_df = 30.0;  // High df, close to Normal
    dc.lr_statistic = 4.0;
    dc.lr_p_value = 0.15;  // Not significant
    dc.prefer_student_t = false;
    dc.normal_aic = 1012.0;
    dc.student_t_aic = 1010.0;
    dc.normal_bic = 1048.0;
    dc.student_t_bic = 1050.0;  // BIC prefers simpler model

    summary.distribution_comparison = dc;

    std::string report = generateTextReport(summary);

    // Verify correct recommendation
    REQUIRE(report.find("Gaussian distribution is adequate") != std::string::npos);
}

TEST(generate_text_report_with_both_distribution_and_diagnostics) {
    using ag::report::DistributionComparison;

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

    // Add distribution comparison
    DistributionComparison dc;
    dc.normal_log_likelihood = -500.0;
    dc.student_t_log_likelihood = -480.0;
    dc.student_t_df = 5.0;
    dc.lr_statistic = 40.0;
    dc.lr_p_value = 0.001;
    dc.prefer_student_t = true;
    dc.normal_aic = 1012.0;
    dc.student_t_aic = 974.0;
    dc.normal_bic = 1048.0;
    dc.student_t_bic = 1011.0;
    summary.distribution_comparison = dc;

    // Add diagnostics
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

    // Verify both sections appear
    REQUIRE(report.find("6. Innovation Distribution Comparison") != std::string::npos);
    REQUIRE(report.find("7. Diagnostic Tests") != std::string::npos);

    // Verify diagnostic subsection numbering is correct
    REQUIRE(report.find("7.1 Ljung-Box Test on Residuals") != std::string::npos);
    REQUIRE(report.find("7.2 Ljung-Box Test on Squared Residuals") != std::string::npos);
    REQUIRE(report.find("7.3 Jarque-Bera Test for Normality") != std::string::npos);
}

// ============================================================================
// Bootstrap and Student-t Innovation Distribution Tests
// ============================================================================

TEST(generate_text_report_with_bootstrap_method) {
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
    summary.innovation_distribution = "Student-t";
    summary.student_t_df = 6.0;

    // Create diagnostic report with bootstrap method
    ag::diagnostics::DiagnosticReport diag;
    diag.ljung_box_method = "bootstrap";
    diag.adf_method = "bootstrap";
    diag.innovation_distribution = "Student-t";
    diag.student_t_df = 6.0;

    diag.ljung_box_residuals.lags = 10;
    diag.ljung_box_residuals.dof = 4;
    diag.ljung_box_residuals.statistic = 8.5;
    diag.ljung_box_residuals.p_value = 0.15;

    diag.ljung_box_squared.lags = 10;
    diag.ljung_box_squared.dof = 7;
    diag.ljung_box_squared.statistic = 5.2;
    diag.ljung_box_squared.p_value = 0.25;

    diag.jarque_bera.statistic = 15.5;
    diag.jarque_bera.p_value = 0.001;  // Low p-value - rejection expected for Student-t

    // Add ADF test
    ag::stats::ADFResult adf;
    adf.lags = 2;
    adf.statistic = -3.5;
    adf.p_value = 0.01;
    adf.critical_value_1pct = -3.43;
    adf.critical_value_5pct = -2.86;
    adf.critical_value_10pct = -2.57;
    diag.adf = adf;

    summary.diagnostics = diag;

    std::string report = generateTextReport(summary);

    // Verify method information appears
    REQUIRE(report.find("Method: Bootstrap") != std::string::npos);
    REQUIRE(report.find("Innovation Distribution: Student-t") != std::string::npos);
    REQUIRE(report.find("Student-t Degrees of Freedom: 6.00") != std::string::npos);

    // Verify test titles include method
    REQUIRE(report.find("Ljung-Box Test on Residuals (bootstrap)") != std::string::npos);
    REQUIRE(report.find("Ljung-Box Test on Squared Residuals (bootstrap)") != std::string::npos);
    REQUIRE(report.find("Augmented Dickey-Fuller Test (bootstrap)") != std::string::npos);

    // Verify Student-t specific Jarque-Bera interpretation
    REQUIRE(report.find("This is EXPECTED for Student-t innovations") != std::string::npos);
    REQUIRE(report.find("heavy tails by design") != std::string::npos);

    // Verify bootstrap interpretation
    REQUIRE(report.find("Bootstrap methods provide accurate p-values") != std::string::npos);
    REQUIRE(report.find("automatically used when Student-t df < 30") != std::string::npos);
}

TEST(generate_text_report_with_asymptotic_method) {
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

    // Create diagnostic report with asymptotic method
    ag::diagnostics::DiagnosticReport diag;
    diag.ljung_box_method = "asymptotic";
    diag.adf_method = "asymptotic";

    diag.ljung_box_residuals.lags = 10;
    diag.ljung_box_residuals.dof = 4;
    diag.ljung_box_residuals.statistic = 8.5;
    diag.ljung_box_residuals.p_value = 0.15;

    diag.ljung_box_squared.lags = 10;
    diag.ljung_box_squared.dof = 7;
    diag.ljung_box_squared.statistic = 5.2;
    diag.ljung_box_squared.p_value = 0.25;

    diag.jarque_bera.statistic = 12.5;
    diag.jarque_bera.p_value = 0.002;

    // Add ADF test
    ag::stats::ADFResult adf;
    adf.lags = 2;
    adf.statistic = -3.5;
    adf.p_value = 0.01;
    adf.critical_value_1pct = -3.43;
    adf.critical_value_5pct = -2.86;
    adf.critical_value_10pct = -2.57;
    diag.adf = adf;

    summary.diagnostics = diag;

    std::string report = generateTextReport(summary);

    // Verify method information appears
    REQUIRE(report.find("Method: Asymptotic") != std::string::npos);
    REQUIRE(report.find("chi-squared for Ljung-Box") != std::string::npos);
    REQUIRE(report.find("MacKinnon for ADF") != std::string::npos);

    // Verify test titles include method
    REQUIRE(report.find("Ljung-Box Test on Residuals (asymptotic)") != std::string::npos);
    REQUIRE(report.find("Ljung-Box Test on Squared Residuals (asymptotic)") != std::string::npos);
    REQUIRE(report.find("Augmented Dickey-Fuller Test (asymptotic)") != std::string::npos);

    // Verify Normal distribution Jarque-Bera interpretation (not Student-t specific)
    REQUIRE(report.find("Heavy tails are common in financial data") != std::string::npos);
    REQUIRE(report.find("This is EXPECTED for Student-t innovations") == std::string::npos);

    // Verify bootstrap interpretation is NOT present
    REQUIRE(report.find("Bootstrap methods provide accurate p-values") == std::string::npos);
}

TEST(generate_text_report_jarque_bera_pass_student_t) {
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
    summary.innovation_distribution = "Student-t";
    summary.student_t_df = 6.0;

    // Create diagnostic report with passing Jarque-Bera
    ag::diagnostics::DiagnosticReport diag;
    diag.ljung_box_method = "bootstrap";
    diag.adf_method = "bootstrap";
    diag.innovation_distribution = "Student-t";
    diag.student_t_df = 6.0;

    diag.ljung_box_residuals.lags = 10;
    diag.ljung_box_residuals.dof = 4;
    diag.ljung_box_residuals.statistic = 8.5;
    diag.ljung_box_residuals.p_value = 0.15;

    diag.ljung_box_squared.lags = 10;
    diag.ljung_box_squared.dof = 7;
    diag.ljung_box_squared.statistic = 5.2;
    diag.ljung_box_squared.p_value = 0.25;

    diag.jarque_bera.statistic = 2.5;
    diag.jarque_bera.p_value = 0.30;  // High p-value - pass

    summary.diagnostics = diag;

    std::string report = generateTextReport(summary);

    // When Jarque-Bera passes, it shouldn't show the Student-t specific note
    REQUIRE(report.find("✓ PASS - Residuals appear normally distributed") != std::string::npos);
    REQUIRE(report.find("This is EXPECTED for Student-t innovations") == std::string::npos);
}

// ============================================================================
// Unconditional Moments Tests
// ============================================================================

TEST(generate_text_report_unconditional_moments_stationary) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    FitSummary summary(spec);

    // Set stationary parameters
    summary.parameters.arima_params.intercept = 0.05;
    summary.parameters.arima_params.ar_coef[0] = 0.6;  // sum < 1, stationary
    summary.parameters.arima_params.ma_coef[0] = 0.3;
    summary.parameters.garch_params.omega = 0.01;
    summary.parameters.garch_params.alpha_coef[0] = 0.1;
    summary.parameters.garch_params.beta_coef[0] = 0.85;  // sum = 0.95 < 1, stationary

    summary.converged = true;
    summary.iterations = 150;
    summary.message = "Converged";
    summary.sample_size = 1000;
    summary.neg_log_likelihood = 500.0;
    summary.aic = 1012.0;
    summary.bic = 1048.0;

    std::string report = generateTextReport(summary);

    // Verify unconditional moments section appears
    REQUIRE(report.find("3. Unconditional Moments (Long-Run Properties)") != std::string::npos);
    REQUIRE(report.find("Unconditional mean:") != std::string::npos);
    REQUIRE(report.find("Unconditional variance:") != std::string::npos);

    // Calculate expected unconditional mean: 0.05 / (1 - 0.6) = 0.125
    REQUIRE(report.find("0.125000") != std::string::npos);

    // Calculate expected unconditional variance: 0.01 / (1 - 0.1 - 0.85) = 0.01 / 0.05 = 0.2
    REQUIRE(report.find("0.200000") != std::string::npos);

    // Verify explanatory notes
    REQUIRE(report.find("long-run average properties") != std::string::npos);
}

TEST(generate_text_report_unconditional_moments_nonstationary_arima) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    FitSummary summary(spec);

    // Set non-stationary ARIMA parameters
    summary.parameters.arima_params.intercept = 0.05;
    summary.parameters.arima_params.ar_coef[0] = 1.0;  // sum = 1, non-stationary
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

    // Verify unconditional mean doesn't exist
    REQUIRE(report.find("Unconditional mean:       Does not exist (non-stationary)") !=
            std::string::npos);

    // Verify unconditional variance still exists (GARCH is stationary)
    REQUIRE(report.find("0.200000") != std::string::npos);
}

TEST(generate_text_report_unconditional_moments_nonstationary_garch) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    FitSummary summary(spec);

    // Set non-stationary GARCH parameters
    summary.parameters.arima_params.intercept = 0.05;
    summary.parameters.arima_params.ar_coef[0] = 0.6;
    summary.parameters.arima_params.ma_coef[0] = 0.3;
    summary.parameters.garch_params.omega = 0.01;
    summary.parameters.garch_params.alpha_coef[0] = 0.5;
    summary.parameters.garch_params.beta_coef[0] = 0.5;  // sum = 1.0, non-stationary

    summary.converged = true;
    summary.iterations = 150;
    summary.message = "Converged";
    summary.sample_size = 1000;
    summary.neg_log_likelihood = 500.0;
    summary.aic = 1012.0;
    summary.bic = 1048.0;

    std::string report = generateTextReport(summary);

    // Verify unconditional mean exists
    REQUIRE(report.find("0.125000") != std::string::npos);

    // Verify unconditional variance doesn't exist
    REQUIRE(report.find("Unconditional variance:   Does not exist (non-stationary GARCH)") !=
            std::string::npos);
}

TEST(generate_text_report_unconditional_moments_no_ar_terms) {
    // Test with no AR terms (p=0)
    ArimaGarchSpec spec(0, 0, 1, 1, 1);
    FitSummary summary(spec);

    summary.parameters.arima_params.intercept = 0.05;
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

    // When p=0, unconditional mean = intercept
    REQUIRE(report.find("Unconditional mean:       0.050000") != std::string::npos);

    // Verify unconditional variance exists
    REQUIRE(report.find("0.200000") != std::string::npos);
}

TEST(generate_text_report_unconditional_moments_multiple_ar) {
    // Test with multiple AR terms
    ArimaGarchSpec spec(2, 0, 1, 1, 1);
    FitSummary summary(spec);

    summary.parameters.arima_params.intercept = 0.1;
    summary.parameters.arima_params.ar_coef[0] = 0.5;
    summary.parameters.arima_params.ar_coef[1] = 0.3;  // sum = 0.8 < 1, stationary
    summary.parameters.arima_params.ma_coef[0] = 0.2;
    summary.parameters.garch_params.omega = 0.05;
    summary.parameters.garch_params.alpha_coef[0] = 0.15;
    summary.parameters.garch_params.beta_coef[0] = 0.80;

    summary.converged = true;
    summary.iterations = 150;
    summary.message = "Converged";
    summary.sample_size = 1000;
    summary.neg_log_likelihood = 500.0;
    summary.aic = 1012.0;
    summary.bic = 1048.0;

    std::string report = generateTextReport(summary);

    // Calculate expected unconditional mean: 0.1 / (1 - 0.5 - 0.3) = 0.1 / 0.2 = 0.5
    REQUIRE(report.find("0.500000") != std::string::npos);

    // Calculate expected unconditional variance: 0.05 / (1 - 0.15 - 0.80) = 0.05 / 0.05 = 1.0
    REQUIRE(report.find("1.000000") != std::string::npos);
}

// ============================================================================
// Main test runner
// ============================================================================

int main() {
    report_test_results("Report: FitSummary");
    return get_test_result();
}
