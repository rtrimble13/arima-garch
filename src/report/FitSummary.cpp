#include "ag/report/FitSummary.hpp"

#include <fmt/core.h>

namespace ag::report {

std::string generateTextReport(const FitSummary& summary) {
    std::string report;

    // Header
    report += "================================================================================\n";
    report += "                    ARIMA-GARCH Model Fit Summary\n";
    report += "================================================================================\n\n";

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
    report += fmt::format("     Intercept:        {:.6f}\n", summary.parameters.arima_params.intercept);

    // AR coefficients
    if (summary.spec.arimaSpec.p > 0) {
        report += "     AR coefficients:  ";
        for (std::size_t i = 0; i < summary.spec.arimaSpec.p; ++i) {
            if (i > 0) report += ", ";
            report += fmt::format("{:.6f}", summary.parameters.arima_params.ar_coef[i]);
        }
        report += "\n";
    }

    // MA coefficients
    if (summary.spec.arimaSpec.q > 0) {
        report += "     MA coefficients:  ";
        for (std::size_t i = 0; i < summary.spec.arimaSpec.q; ++i) {
            if (i > 0) report += ", ";
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
            if (i > 0) report += ", ";
            report += fmt::format("{:.6f}", summary.parameters.garch_params.alpha_coef[i]);
        }
        report += "\n";
    }

    // GARCH (beta) coefficients
    if (summary.spec.garchSpec.q > 0) {
        report += "     GARCH (beta):     ";
        for (std::size_t i = 0; i < summary.spec.garchSpec.q; ++i) {
            if (i > 0) report += ", ";
            report += fmt::format("{:.6f}", summary.parameters.garch_params.beta_coef[i]);
        }
        report += "\n";
    }
    report += "\n";

    // 3. Convergence Information
    report += "3. Convergence Information\n";
    report += "   -----------------------\n";
    report += fmt::format("   Status:             {}\n", summary.converged ? "✓ Converged" : "✗ Not converged");
    report += fmt::format("   Iterations:         {}\n", summary.iterations);
    report += fmt::format("   Message:            {}\n\n", summary.message);

    // 4. Model Fit Statistics
    report += "4. Model Fit Statistics\n";
    report += "   --------------------\n";
    report += fmt::format("   Log-likelihood:     {:.6f}\n", -summary.neg_log_likelihood);
    report += fmt::format("   AIC:                {:.6f}\n", summary.aic);
    report += fmt::format("   BIC:                {:.6f}\n", summary.bic);
    report += "\n";
    report += "   Note: Lower AIC/BIC values indicate better models.\n";
    report += "         AIC = 2k + 2*NLL, BIC = k*log(n) + 2*NLL\n";
    report += "         where k = number of parameters, n = sample size\n\n";

    // 5. Diagnostic Tests (if available)
    if (summary.diagnostics.has_value()) {
        const auto& diag = summary.diagnostics.value();
        
        report += "5. Diagnostic Tests\n";
        report += "   ----------------\n\n";

        // Ljung-Box test on residuals
        report += "   5.1 Ljung-Box Test on Residuals\n";
        report += "       Tests for autocorrelation in conditional mean residuals.\n";
        report += fmt::format("       Lags:           {}\n", diag.ljung_box_residuals.lags);
        report += fmt::format("       DOF:            {}\n", diag.ljung_box_residuals.dof);
        report += fmt::format("       Statistic:      {:.4f}\n", diag.ljung_box_residuals.statistic);
        report += fmt::format("       P-value:        {:.4f}\n", diag.ljung_box_residuals.p_value);
        if (diag.ljung_box_residuals.p_value > 0.05) {
            report += "       Result:         ✓ PASS - No significant autocorrelation\n";
        } else {
            report += "       Result:         ✗ FAIL - Significant autocorrelation detected\n";
        }
        report += "\n";

        // Ljung-Box test on squared residuals
        report += "   5.2 Ljung-Box Test on Squared Residuals\n";
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
        report += "   5.3 Jarque-Bera Test for Normality\n";
        report += "       Tests whether standardized residuals are normally distributed.\n";
        report += fmt::format("       Statistic:      {:.4f}\n", diag.jarque_bera.statistic);
        report += fmt::format("       P-value:        {:.4f}\n", diag.jarque_bera.p_value);
        if (diag.jarque_bera.p_value > 0.05) {
            report += "       Result:         ✓ PASS - Residuals appear normally distributed\n";
        } else {
            report += "       Result:         ✗ FAIL - Residuals deviate from normality\n";
            report += "       Note:           Heavy tails are common in financial data\n";
        }
        report += "\n";

        // ADF test (if available)
        if (diag.adf.has_value()) {
            report += "   5.4 Augmented Dickey-Fuller Test\n";
            report += "       Tests for stationarity of residuals.\n";
            report += fmt::format("       Lags:           {}\n", diag.adf->lags);
            report += fmt::format("       Statistic:      {:.4f}\n", diag.adf->statistic);
            report += fmt::format("       P-value:        {:.4f}\n", diag.adf->p_value);
            report += "       Critical values:\n";
            report += fmt::format("         1%%:  {:.4f}\n", diag.adf->critical_value_1pct);
            report += fmt::format("         5%%:  {:.4f}\n", diag.adf->critical_value_5pct);
            report += fmt::format("         10%%: {:.4f}\n", diag.adf->critical_value_10pct);
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
        report += "   - Jarque-Bera rejection is common for financial data with heavy tails\n";
        if (diag.adf.has_value()) {
            report += "   - Low ADF p-value (< 0.05) indicates stationarity (desirable)\n";
        }
        report += "\n";
    }

    report += "================================================================================\n";

    return report;
}

}  // namespace ag::report
