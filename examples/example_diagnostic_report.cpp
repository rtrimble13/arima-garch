// Example demonstrating comprehensive diagnostic report for ARIMA-GARCH models
// Compile: cmake --build build && ./build/examples/example_diagnostic_report

#include "ag/diagnostics/DiagnosticReport.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"

#include <iostream>

#include <fmt/core.h>

using ag::diagnostics::computeDiagnostics;
using ag::models::ArimaGarchSpec;
using ag::models::composite::ArimaGarchParameters;
using ag::simulation::ArimaGarchSimulator;

void printDiagnosticReport(const ag::diagnostics::DiagnosticReport& report,
                           const std::string& title) {
    fmt::print("\n=== {} ===\n", title);

    fmt::print("\n1. Ljung-Box Test on Residuals\n");
    fmt::print("   Tests for autocorrelation in the conditional mean residuals.\n");
    fmt::print("   Lags:       {}\n", report.ljung_box_residuals.lags);
    fmt::print("   DOF:        {}\n", report.ljung_box_residuals.dof);
    fmt::print("   Statistic:  {:.4f}\n", report.ljung_box_residuals.statistic);
    fmt::print("   P-value:    {:.4f}\n", report.ljung_box_residuals.p_value);
    if (report.ljung_box_residuals.p_value > 0.05) {
        fmt::print("   Result:     ✓ PASS - No significant autocorrelation detected\n");
    } else {
        fmt::print("   Result:     ✗ FAIL - Significant autocorrelation detected\n");
    }

    fmt::print("\n2. Ljung-Box Test on Squared Residuals\n");
    fmt::print("   Tests for remaining ARCH effects (volatility clustering).\n");
    fmt::print("   Lags:       {}\n", report.ljung_box_squared.lags);
    fmt::print("   DOF:        {}\n", report.ljung_box_squared.dof);
    fmt::print("   Statistic:  {:.4f}\n", report.ljung_box_squared.statistic);
    fmt::print("   P-value:    {:.4f}\n", report.ljung_box_squared.p_value);
    if (report.ljung_box_squared.p_value > 0.05) {
        fmt::print("   Result:     ✓ PASS - No remaining ARCH effects\n");
    } else {
        fmt::print("   Result:     ✗ FAIL - Remaining ARCH effects detected\n");
    }

    fmt::print("\n3. Jarque-Bera Test for Normality\n");
    fmt::print("   Tests whether standardized residuals are normally distributed.\n");
    fmt::print("   Statistic:  {:.4f}\n", report.jarque_bera.statistic);
    fmt::print("   P-value:    {:.4f}\n", report.jarque_bera.p_value);
    if (report.jarque_bera.p_value > 0.05) {
        fmt::print("   Result:     ✓ PASS - Residuals appear normally distributed\n");
    } else {
        fmt::print("   Result:     ✗ FAIL - Residuals deviate from normality\n");
        fmt::print("   Note:       Heavy tails are common in financial data\n");
    }

    if (report.adf.has_value()) {
        fmt::print("\n4. Augmented Dickey-Fuller Test\n");
        fmt::print("   Tests for stationarity of residuals.\n");
        fmt::print("   Lags:       {}\n", report.adf->lags);
        fmt::print("   Statistic:  {:.4f}\n", report.adf->statistic);
        fmt::print("   P-value:    {:.4f}\n", report.adf->p_value);
        fmt::print("   Critical values:\n");
        fmt::print("     1%%:  {:.4f}\n", report.adf->critical_value_1pct);
        fmt::print("     5%%:  {:.4f}\n", report.adf->critical_value_5pct);
        fmt::print("     10%%: {:.4f}\n", report.adf->critical_value_10pct);
        if (report.adf->p_value < 0.05) {
            fmt::print("   Result:     ✓ PASS - Residuals are stationary\n");
        } else {
            fmt::print("   Result:     ✗ FAIL - Residuals may have unit root\n");
        }
    }

    fmt::print("\n");
}

int main() {
    fmt::print("=== ARIMA-GARCH Diagnostic Report Example ===\n\n");

    // Example 1: Well-specified ARIMA(1,0,1)-GARCH(1,1) model
    fmt::print("Example 1: Correctly Specified Model\n");
    fmt::print("-------------------------------------\n");

    ArimaGarchSpec spec1(1, 0, 1, 1, 1);
    ArimaGarchParameters params1(spec1);

    params1.arima_params.intercept = 0.05;
    params1.arima_params.ar_coef[0] = 0.6;
    params1.arima_params.ma_coef[0] = 0.3;
    params1.garch_params.omega = 0.01;
    params1.garch_params.alpha_coef[0] = 0.1;
    params1.garch_params.beta_coef[0] = 0.85;

    fmt::print("Model: ARIMA({},{},{})-GARCH({},{})\n", spec1.arimaSpec.p, spec1.arimaSpec.d,
               spec1.arimaSpec.q, spec1.garchSpec.p, spec1.garchSpec.q);

    // Simulate data from the model
    ArimaGarchSimulator simulator1(spec1, params1);
    auto sim_result1 = simulator1.simulate(1000, 42);

    // Compute diagnostics using the same (correct) parameters
    auto report1 = computeDiagnostics(spec1, params1, sim_result1.returns, 10, true);

    printDiagnosticReport(report1, "Diagnostic Report for Correctly Specified Model");

    // Example 2: Simpler model - ARIMA(0,0,0)-GARCH(1,1) (white noise mean)
    fmt::print("\nExample 2: White Noise Mean with GARCH(1,1)\n");
    fmt::print("--------------------------------------------\n");

    ArimaGarchSpec spec2(0, 0, 0, 1, 1);
    ArimaGarchParameters params2(spec2);

    params2.arima_params.intercept = 0.0;
    params2.garch_params.omega = 0.05;
    params2.garch_params.alpha_coef[0] = 0.15;
    params2.garch_params.beta_coef[0] = 0.80;

    fmt::print("Model: ARIMA({},{},{})-GARCH({},{})\n", spec2.arimaSpec.p, spec2.arimaSpec.d,
               spec2.arimaSpec.q, spec2.garchSpec.p, spec2.garchSpec.q);

    // Simulate data
    ArimaGarchSimulator simulator2(spec2, params2);
    auto sim_result2 = simulator2.simulate(1000, 123);

    // Compute diagnostics without ADF test
    auto report2 = computeDiagnostics(spec2, params2, sim_result2.returns, 10, false);

    printDiagnosticReport(report2, "Diagnostic Report for White Noise Model");

    fmt::print("\n=== Interpretation Guidelines ===\n");
    fmt::print("1. Ljung-Box tests: High p-values (> 0.05) are desirable\n");
    fmt::print("   - Indicates no remaining autocorrelation in residuals\n");
    fmt::print("   - If test fails, consider increasing ARIMA or GARCH orders\n\n");

    fmt::print("2. Jarque-Bera test: Tests for normality\n");
    fmt::print("   - High p-value indicates normal distribution\n");
    fmt::print("   - Rejection is common for financial data (heavy tails)\n");
    fmt::print("   - Consider using t-distribution or other heavy-tailed distributions\n\n");

    fmt::print("3. ADF test: Tests for stationarity\n");
    fmt::print("   - Low p-value (< 0.05) indicates stationarity (desirable)\n");
    fmt::print("   - If test fails, may need differencing or trend removal\n\n");

    fmt::print("Note: These diagnostics help assess model adequacy but should be\n");
    fmt::print("      interpreted in context with domain knowledge and other criteria.\n");

    return 0;
}
