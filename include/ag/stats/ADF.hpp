#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace ag::stats {

/**
 * @brief Regression form for the ADF test.
 *
 * Determines which deterministic components are included in the test regression:
 * - None: Δy_t = φy_{t-1} + ... (no constant or trend)
 * - Constant: Δy_t = α + φy_{t-1} + ... (includes constant term)
 * - ConstantAndTrend: Δy_t = α + βt + φy_{t-1} + ... (includes constant and trend)
 */
enum class ADFRegressionForm {
    None,             ///< No constant, no trend
    Constant,         ///< Constant term only
    ConstantAndTrend  ///< Constant and linear trend
};

/**
 * @brief Result of the Augmented Dickey-Fuller (ADF) test for stationarity.
 *
 * The ADF test is used to test the null hypothesis that a unit root is present
 * in a time series (i.e., the series is non-stationary).
 */
struct ADFResult {
    double statistic;                   ///< The ADF test statistic (t-statistic)
    double p_value;                     ///< Approximate p-value
    std::size_t lags;                   ///< Number of lags used in the test
    ADFRegressionForm regression_form;  ///< Regression form used
    double critical_value_1pct;         ///< Critical value at 1% significance
    double critical_value_5pct;         ///< Critical value at 5% significance
    double critical_value_10pct;        ///< Critical value at 10% significance
};

/**
 * @brief Perform the Augmented Dickey-Fuller (ADF) test for stationarity.
 *
 * The ADF test examines the null hypothesis that a unit root is present in the
 * time series. Under the null hypothesis, the series is non-stationary.
 *
 * The test regression is:
 * Δy_t = α + βt + φy_{t-1} + γ_1Δy_{t-1} + ... + γ_pΔy_{t-p} + ε_t
 *
 * where the null hypothesis is H₀: φ = 0 (unit root present).
 *
 * Interpretation:
 * - If statistic < critical value: Reject null hypothesis - series is stationary
 * - If statistic > critical value: Fail to reject - series has unit root (non-stationary)
 * - Low p-value (e.g., < 0.05): Evidence for stationarity
 * - High p-value (e.g., > 0.05): Evidence for unit root
 *
 * @param data Span of time series data
 * @param lags Number of lagged differences to include (if 0, automatically selected)
 * @param regression_form Type of deterministic components to include
 * @param max_lags Maximum lags to consider for automatic selection (default: 12*(n/100)^(1/4))
 * @return ADFResult containing the test statistic, p-value, and critical values
 * @throws std::invalid_argument if data is too short or parameters are invalid
 */
[[nodiscard]] ADFResult adf_test(std::span<const double> data, std::size_t lags = 0,
                                 ADFRegressionForm regression_form = ADFRegressionForm::Constant,
                                 std::size_t max_lags = 0);

/**
 * @brief Automatically select the best regression form for the ADF test.
 *
 * Uses a sequential testing procedure to determine whether to include
 * constant and/or trend terms in the test regression.
 *
 * @param data Span of time series data
 * @param lags Number of lagged differences to include (if 0, automatically selected)
 * @param max_lags Maximum lags to consider for automatic selection
 * @return ADFResult with automatically selected regression form
 * @throws std::invalid_argument if data is too short
 */
[[nodiscard]] ADFResult adf_test_auto(std::span<const double> data, std::size_t lags = 0,
                                      std::size_t max_lags = 0);

// Convenience overloads for std::vector
[[nodiscard]] inline ADFResult
adf_test(const std::vector<double>& data, std::size_t lags = 0,
         ADFRegressionForm regression_form = ADFRegressionForm::Constant,
         std::size_t max_lags = 0) {
    return adf_test(std::span<const double>(data), lags, regression_form, max_lags);
}

[[nodiscard]] inline ADFResult adf_test_auto(const std::vector<double>& data, std::size_t lags = 0,
                                             std::size_t max_lags = 0) {
    return adf_test_auto(std::span<const double>(data), lags, max_lags);
}

}  // namespace ag::stats
