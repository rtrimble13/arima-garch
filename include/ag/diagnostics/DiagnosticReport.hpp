#pragma once

#include "ag/diagnostics/Residuals.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/stats/ADF.hpp"
#include "ag/stats/JarqueBera.hpp"
#include "ag/stats/LjungBox.hpp"

#include <optional>
#include <vector>

namespace ag::diagnostics {

/**
 * @brief Aggregated diagnostic test results for ARIMA-GARCH model residuals.
 *
 * This structure contains the results of several standard diagnostic tests
 * that are commonly used to assess the adequacy of a fitted ARIMA-GARCH model:
 *
 * - Ljung-Box test on residuals: Tests for remaining autocorrelation in the
 *   conditional mean residuals. A well-specified model should have residuals
 *   that are approximately white noise (high p-value).
 *
 * - Ljung-Box test on squared residuals: Tests for remaining ARCH effects
 *   (autocorrelation in volatility). A well-specified GARCH model should
 *   have squared residuals that are approximately white noise (high p-value).
 *
 * - Jarque-Bera test: Tests whether standardized residuals follow a normal
 *   distribution. Many financial time series have heavy-tailed distributions,
 *   so rejection of normality is common even for well-specified models.
 *
 * - ADF test: Tests whether residuals are stationary. This is optional and
 *   may not always be applicable depending on the modeling workflow.
 */
struct DiagnosticReport {
    /**
     * @brief Ljung-Box test result for raw residuals.
     *
     * Tests for autocorrelation in the conditional mean residuals.
     * High p-value (e.g., > 0.05) suggests residuals are white noise (good).
     */
    stats::LjungBoxResult ljung_box_residuals;

    /**
     * @brief Ljung-Box test result for squared residuals.
     *
     * Tests for remaining ARCH effects (autocorrelation in squared residuals).
     * High p-value (e.g., > 0.05) suggests no remaining volatility clustering (good).
     */
    stats::LjungBoxResult ljung_box_squared;

    /**
     * @brief Jarque-Bera test result for standardized residuals.
     *
     * Tests for normality of standardized residuals.
     * High p-value (e.g., > 0.05) suggests residuals are normally distributed.
     * Note: Rejection is common for financial data with heavy tails.
     */
    stats::JarqueBeraResult jarque_bera;

    /**
     * @brief ADF test result for raw residuals (optional).
     *
     * Tests for stationarity of residuals. This is optional and may be
     * omitted (std::nullopt) depending on the workflow.
     */
    std::optional<stats::ADFResult> adf;
};

/**
 * @brief Compute a comprehensive diagnostic report for ARIMA-GARCH model residuals.
 *
 * This function runs a battery of diagnostic tests on the residuals from a fitted
 * ARIMA-GARCH model. The tests help assess whether the model is adequately specified.
 *
 * The following tests are performed:
 * 1. Ljung-Box test on residuals (tests for autocorrelation in conditional mean)
 * 2. Ljung-Box test on squared residuals (tests for remaining ARCH effects)
 * 3. Jarque-Bera test on standardized residuals (tests for normality)
 * 4. ADF test on residuals (optional, tests for stationarity)
 *
 * Interpretation guidelines:
 * - For Ljung-Box tests: High p-values (> 0.05) are desirable, indicating no
 *   significant autocorrelation remaining in residuals
 * - For Jarque-Bera: High p-value indicates normality, but rejection is common
 *   for financial data even with well-specified models
 * - For ADF: Low p-value indicates stationarity (desirable)
 *
 * @param spec ARIMA-GARCH model specification
 * @param params Fitted model parameters
 * @param data Time series data used for fitting
 * @param ljung_box_lags Number of lags to use for Ljung-Box tests (default: 10)
 *                       Must be greater than the total number of model parameters
 * @param include_adf Whether to include ADF test in the report (default: false)
 * @return DiagnosticReport containing all test results
 * @throws std::invalid_argument if data is empty or parameters are invalid
 * @throws std::invalid_argument if ljung_box_lags <= number of model parameters
 *         (insufficient degrees of freedom for meaningful test results)
 *
 * @note The degrees of freedom for Ljung-Box tests are automatically adjusted
 *       to account for the number of estimated parameters in the model.
 *       For an ARIMA(p,d,q)-GARCH(P,Q) model, the total number of parameters is:
 *       (1 + p + q) + (1 + P + Q), accounting for intercept, AR, MA, omega, alpha, beta.
 */
[[nodiscard]] DiagnosticReport
computeDiagnostics(const ag::models::ArimaGarchSpec& spec,
                   const ag::models::composite::ArimaGarchParameters& params,
                   const std::vector<double>& data, std::size_t ljung_box_lags = 10,
                   bool include_adf = false);

}  // namespace ag::diagnostics
