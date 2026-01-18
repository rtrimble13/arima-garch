#include "ag/report/FitSummary.hpp"

#include <fmt/core.h>

namespace ag::report {

std::string generateTextReport(const FitSummary& summary) {
    std::string report;

    // Header
    report += "================================================================================\n";
    report += "                    ARIMA-GARCH Model Fit Summary\n";
    report +=
        "================================================================================\n\n";

    // 1. Model Specification
    report += "1. Model Specification\n";
    report += "   -------------------\n";
    report += fmt::format("   ARIMA order:        ({},{},{})\n", summary.spec.arimaSpec.p,
                          summary.spec.arimaSpec.d, summary.spec.arimaSpec.q);
    report += fmt::format("   GARCH order:        ({},{})\n", summary.spec.garchSpec.p,
                          summary.spec.garchSpec.q);
    report += fmt::format("   Total parameters:   {}\n", summary.spec.totalParamCount());
    report += fmt::format("   Sample size:        {}\n\n", summary.sample_size);

    // 2. Estimated Parameters
    report += "2. Estimated Parameters\n";
    report += "   --------------------\n";
    report += "   ARIMA:\n";
    report +=
        fmt::format("     Intercept:        {:.6f}\n", summary.parameters.arima_params.intercept);

    // AR coefficients
    if (summary.spec.arimaSpec.p > 0) {
        report += "     AR coefficients:  ";
        for (std::size_t i = 0; i < summary.spec.arimaSpec.p; ++i) {
            if (i > 0)
                report += ", ";
            report += fmt::format("{:.6f}", summary.parameters.arima_params.ar_coef[i]);
        }
        report += "\n";
    }

    // MA coefficients
    if (summary.spec.arimaSpec.q > 0) {
        report += "     MA coefficients:  ";
        for (std::size_t i = 0; i < summary.spec.arimaSpec.q; ++i) {
            if (i > 0)
                report += ", ";
            report += fmt::format("{:.6f}", summary.parameters.arima_params.ma_coef[i]);
        }
        report += "\n";
    }

    report += "\n   GARCH:\n";
    report += fmt::format("     Omega:            {:.6f}\n", summary.parameters.garch_params.omega);

    // ARCH (alpha) coefficients
    if (summary.spec.garchSpec.p > 0) {
        report += "     ARCH (alpha):     ";
        for (std::size_t i = 0; i < summary.spec.garchSpec.p; ++i) {
            if (i > 0)
                report += ", ";
            report += fmt::format("{:.6f}", summary.parameters.garch_params.alpha_coef[i]);
        }
        report += "\n";
    }

    // GARCH (beta) coefficients
    if (summary.spec.garchSpec.q > 0) {
        report += "     GARCH (beta):     ";
        for (std::size_t i = 0; i < summary.spec.garchSpec.q; ++i) {
            if (i > 0)
                report += ", ";
            report += fmt::format("{:.6f}", summary.parameters.garch_params.beta_coef[i]);
        }
        report += "\n";
    }
    report += "\n";

    // 3. Unconditional Moments (Long-Run Properties)
    report += "3. Unconditional Moments (Long-Run Properties)\n";
    report += "   -------------------------------------------\n";

    // Calculate unconditional mean
    double sum_ar = 0.0;
    if (summary.spec.arimaSpec.p > 0) {
        for (std::size_t i = 0; i < summary.spec.arimaSpec.p; ++i) {
            sum_ar += summary.parameters.arima_params.ar_coef[i];
        }
    }

    report += "   Unconditional mean:       ";
    if (summary.spec.arimaSpec.p == 0) {
        // No AR terms: unconditional mean = intercept
        report += fmt::format("{:.6f}\n", summary.parameters.arima_params.intercept);
    } else if (sum_ar < 1.0) {
        // Stationary: unconditional mean = intercept / (1 - sum_ar)
        double unconditional_mean = summary.parameters.arima_params.intercept / (1.0 - sum_ar);
        report += fmt::format("{:.6f}\n", unconditional_mean);
    } else {
        // Non-stationary: unconditional mean doesn't exist
        report += "Does not exist (non-stationary)\n";
    }

    // Calculate unconditional variance
    double unconditional_variance = summary.parameters.garch_params.unconditionalVariance();
    report += "   Unconditional variance:   ";
    if (unconditional_variance > 0.0) {
        report += fmt::format("{:.6f}\n", unconditional_variance);
    } else {
        report += "Does not exist (non-stationary GARCH)\n";
    }

    report += "\n";
    report += "   Note: Unconditional moments represent long-run average properties.\n";
    report += "         They exist only when the model is stationary.\n";
    report += "         For ARIMA: stationarity requires sum of AR coefficients < 1.\n";
    report += "         For GARCH: stationarity requires sum of ARCH + GARCH coefficients < 1.\n";
    report += "\n";

    // 4. Convergence Information
    report += "4. Convergence Information\n";
    report += "   -----------------------\n";
    report += fmt::format("   Status:             {}\n",
                          summary.converged ? "✓ Converged" : "✗ Not converged");
    report += fmt::format("   Iterations:         {}\n", summary.iterations);
    report += fmt::format("   Message:            {}\n\n", summary.message);

    // 5. Model Fit Statistics
    report += "5. Model Fit Statistics\n";
    report += "   --------------------\n";
    report += fmt::format("   Innovation dist.:   {}", summary.innovation_distribution);
    if (summary.innovation_distribution == "Student-t") {
        report += fmt::format(" (df={:.2f})", summary.student_t_df);
    }
    report += "\n";
    report += fmt::format("   Log-likelihood:     {:.6f}\n", -summary.neg_log_likelihood);
    report += fmt::format("   AIC:                {:.6f}\n", summary.aic);
    report += fmt::format("   BIC:                {:.6f}\n", summary.bic);
    report += "\n";
    report += "   Note: Lower AIC/BIC values indicate better models.\n";
    report += "         AIC = 2k + 2*NLL, BIC = k*log(n) + 2*NLL\n";
    report += "         where k = number of parameters, n = sample size\n\n";

    // 6. Innovation Distribution Comparison (if available)
    if (summary.distribution_comparison.has_value()) {
        const auto& dist = summary.distribution_comparison.value();

        report += "6. Innovation Distribution Comparison\n";
        report += "   ----------------------------------\n";
        report +=
            fmt::format("   Gaussian log-likelihood:    {:.4f}\n", dist.normal_log_likelihood);
        report +=
            fmt::format("   Student-t log-likelihood:   {:.4f}\n", dist.student_t_log_likelihood);
        report += fmt::format("   Estimated df (Student-t):   {:.2f}\n", dist.student_t_df);
        report += "\n";
        report += "   Likelihood Ratio Test:\n";
        report += fmt::format("     Statistic:  {:.4f}\n", dist.lr_statistic);
        report += fmt::format("     P-value:    {:.4f}\n", dist.lr_p_value);
        report += "\n";
        report += "   Information Criteria:\n";
        report += fmt::format("     AIC (Normal):     {:.4f}\n", dist.normal_aic);
        report += fmt::format("     AIC (Student-t):  {:.4f}\n", dist.student_t_aic);
        report += fmt::format("     BIC (Normal):     {:.4f}\n", dist.normal_bic);
        report += fmt::format("     BIC (Student-t):  {:.4f}\n", dist.student_t_bic);
        report += "\n";

        if (dist.prefer_student_t) {
            report += "   ✓ RECOMMENDATION: Student-t distribution provides better fit\n";
            report += "     Consider refitting with Student-t innovations.\n";
        } else {
            report += "   ✓ Gaussian distribution is adequate for this data\n";
        }
        report += "\n";
    }

    // 7. Diagnostic Tests (if available)
    int diag_section_num = summary.distribution_comparison.has_value() ? 7 : 6;
    if (summary.diagnostics.has_value()) {
        const auto& diag = summary.diagnostics.value();

        report += fmt::format("{}. Diagnostic Tests\n", diag_section_num);
        report += "   ----------------\n\n";

        // Display method and innovation distribution information
        if (diag.ljung_box_method == "bootstrap") {
            report += "   Method: Bootstrap (for Student-t or heavy-tailed distributions)\n";
            if (diag.innovation_distribution.has_value()) {
                report += fmt::format("   Innovation Distribution: {}\n",
                                      diag.innovation_distribution.value());
                if (diag.student_t_df.has_value()) {
                    report += fmt::format("   Student-t Degrees of Freedom: {:.2f}\n",
                                          diag.student_t_df.value());
                }
            }
        } else {
            report += "   Method: Asymptotic (chi-squared for Ljung-Box, MacKinnon for ADF)\n";
        }
        report += "\n";

        // Ljung-Box test on residuals
        report += fmt::format("   {}.1 Ljung-Box Test on Residuals ({})\n", diag_section_num,
                              diag.ljung_box_method);
        report += "       Tests for autocorrelation in conditional mean residuals.\n";
        report += fmt::format("       Lags:           {}\n", diag.ljung_box_residuals.lags);
        report += fmt::format("       DOF:            {}\n", diag.ljung_box_residuals.dof);
        report +=
            fmt::format("       Statistic:      {:.4f}\n", diag.ljung_box_residuals.statistic);
        report += fmt::format("       P-value:        {:.4f}\n", diag.ljung_box_residuals.p_value);
        if (diag.ljung_box_residuals.p_value > 0.05) {
            report += "       Result:         ✓ PASS - No significant autocorrelation\n";
        } else {
            report += "       Result:         ✗ FAIL - Significant autocorrelation detected\n";
        }
        report += "\n";

        // Ljung-Box test on squared residuals
        report += fmt::format("   {}.2 Ljung-Box Test on Squared Residuals ({})\n",
                              diag_section_num, diag.ljung_box_method);
        report += "       Tests for remaining ARCH effects (volatility clustering).\n";
        report += fmt::format("       Lags:           {}\n", diag.ljung_box_squared.lags);
        report += fmt::format("       DOF:            {}\n", diag.ljung_box_squared.dof);
        report += fmt::format("       Statistic:      {:.4f}\n", diag.ljung_box_squared.statistic);
        report += fmt::format("       P-value:        {:.4f}\n", diag.ljung_box_squared.p_value);
        if (diag.ljung_box_squared.p_value > 0.05) {
            report += "       Result:         ✓ PASS - No remaining ARCH effects\n";
        } else {
            report += "       Result:         ✗ FAIL - Remaining ARCH effects detected\n";
        }
        report += "\n";

        // Jarque-Bera test
        report += fmt::format("   {}.3 Jarque-Bera Test for Normality\n", diag_section_num);
        report += "       Tests whether standardized residuals are normally distributed.\n";
        report += fmt::format("       Statistic:      {:.4f}\n", diag.jarque_bera.statistic);
        report += fmt::format("       P-value:        {:.4f}\n", diag.jarque_bera.p_value);
        if (diag.jarque_bera.p_value > 0.05) {
            report += "       Result:         ✓ PASS - Residuals appear normally distributed\n";
        } else {
            report += "       Result:         ✗ FAIL - Residuals deviate from normality\n";
            if (diag.innovation_distribution.has_value() &&
                diag.innovation_distribution.value() == "Student-t") {
                report += "       Note:           This is EXPECTED for Student-t innovations "
                          "(heavy tails by design)\n";
            } else {
                report += "       Note:           Heavy tails are common in financial data\n";
            }
        }
        report += "\n";

        // ADF test (if available)
        if (diag.adf.has_value()) {
            report += fmt::format("   {}.4 Augmented Dickey-Fuller Test ({})\n", diag_section_num,
                                  diag.adf_method);
            report += "       Tests for stationarity of residuals.\n";
            report += fmt::format("       Lags:           {}\n", diag.adf->lags);
            report += fmt::format("       Statistic:      {:.4f}\n", diag.adf->statistic);
            report += fmt::format("       P-value:        {:.4f}\n", diag.adf->p_value);
            report += "       Critical values:\n";
            report += fmt::format("         1%:  {:.4f}\n", diag.adf->critical_value_1pct);
            report += fmt::format("         5%:  {:.4f}\n", diag.adf->critical_value_5pct);
            report += fmt::format("         10%: {:.4f}\n", diag.adf->critical_value_10pct);
            if (diag.adf->p_value < 0.05) {
                report += "       Result:         ✓ PASS - Residuals are stationary\n";
            } else {
                report += "       Result:         ✗ FAIL - Residuals may have unit root\n";
            }
            report += "\n";
        }

        report += "   Interpretation:\n";
        report += "   - High p-values (> 0.05) for Ljung-Box tests indicate no\n";
        report += "     remaining autocorrelation (desirable for well-specified models)\n";
        if (diag.innovation_distribution.has_value() &&
            diag.innovation_distribution.value() == "Student-t") {
            report += "   - Jarque-Bera rejection is EXPECTED when using Student-t innovations\n";
            report += "     (Student-t distribution has heavy tails by design)\n";
        } else {
            report += "   - Jarque-Bera rejection is common for financial data with heavy tails\n";
        }
        if (diag.adf.has_value()) {
            report += "   - Low ADF p-value (< 0.05) indicates stationarity (desirable)\n";
        }
        if (diag.ljung_box_method == "bootstrap") {
            report += "   - Bootstrap methods provide accurate p-values for heavy-tailed "
                      "distributions\n";
            report += "     (automatically used when Student-t df < 30)\n";
        }
        report += "\n";
    }

    report += "================================================================================\n";

    return report;
}

}  // namespace ag::report
