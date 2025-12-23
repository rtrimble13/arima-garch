#pragma once

#include "ag/diagnostics/DiagnosticReport.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"

#include <optional>
#include <string>
#include <vector>

namespace ag::report {

/**
 * @brief Summary of ARIMA-GARCH model fitting results.
 *
 * FitSummary encapsulates all relevant information from fitting an ARIMA-GARCH
 * model to time series data, including:
 * - Model specification (ARIMA and GARCH orders)
 * - Estimated parameters
 * - Convergence information (iterations, status)
 * - Information criteria (AIC, BIC)
 * - Log-likelihood
 * - Diagnostic test results (optional)
 *
 * This structure is designed to provide a complete snapshot of the fitting
 * process that can be used for reporting, serialization, or further analysis.
 */
struct FitSummary {
    /**
     * @brief Model specification (ARIMA and GARCH orders).
     */
    ag::models::ArimaGarchSpec spec;

    /**
     * @brief Estimated model parameters.
     */
    ag::models::composite::ArimaGarchParameters parameters;

    /**
     * @brief Negative log-likelihood at the optimal parameters.
     *
     * Lower values indicate better fit (more likely parameters given the data).
     */
    double neg_log_likelihood;

    /**
     * @brief Akaike Information Criterion (AIC).
     *
     * AIC = 2k + 2*NLL, where k is the number of parameters and NLL is the
     * negative log-likelihood. Lower values indicate better models.
     */
    double aic;

    /**
     * @brief Bayesian Information Criterion (BIC).
     *
     * BIC = k*log(n) + 2*NLL, where k is the number of parameters, n is the
     * sample size, and NLL is the negative log-likelihood. Lower values
     * indicate better models. BIC penalizes model complexity more heavily
     * than AIC.
     */
    double bic;

    /**
     * @brief Whether the optimization algorithm converged.
     */
    bool converged;

    /**
     * @brief Number of iterations performed during optimization.
     */
    int iterations;

    /**
     * @brief Status message from the optimizer (e.g., "Converged", "Max iterations").
     */
    std::string message;

    /**
     * @brief Number of observations used for fitting.
     */
    std::size_t sample_size;

    /**
     * @brief Optional diagnostic test results.
     *
     * If diagnostics were computed after fitting, they are stored here.
     * This allows for comprehensive model assessment in a single report.
     */
    std::optional<ag::diagnostics::DiagnosticReport> diagnostics;

    /**
     * @brief Construct a FitSummary with given specification.
     * @param spec ARIMA-GARCH specification
     */
    explicit FitSummary(const ag::models::ArimaGarchSpec& spec)
        : spec(spec), parameters(spec), neg_log_likelihood(0.0), aic(0.0), bic(0.0),
          converged(false), iterations(0), sample_size(0) {}
};

/**
 * @brief Generate a human-readable text report from a FitSummary.
 *
 * This function formats the FitSummary into a clean, readable text report
 * suitable for console output or file writing. The report includes:
 *
 * 1. Model Specification:
 *    - ARIMA(p,d,q) and GARCH(p,q) orders
 *    - Total number of parameters
 *    - Sample size
 *
 * 2. Estimated Parameters:
 *    - ARIMA: intercept, AR coefficients, MA coefficients
 *    - GARCH: omega, ARCH coefficients, GARCH coefficients
 *
 * 3. Convergence Information:
 *    - Convergence status
 *    - Number of iterations
 *    - Status message
 *
 * 4. Model Fit Statistics:
 *    - Negative log-likelihood
 *    - AIC (Akaike Information Criterion)
 *    - BIC (Bayesian Information Criterion)
 *
 * 5. Diagnostic Tests (if available):
 *    - Ljung-Box test on residuals
 *    - Ljung-Box test on squared residuals
 *    - Jarque-Bera normality test
 *    - ADF stationarity test (if included)
 *
 * @param summary FitSummary containing all fit results
 * @return Formatted text report as a string
 *
 * @note The function uses the {fmt} library for formatting, which is consistent
 *       with the rest of the codebase.
 *
 * @example
 * ```cpp
 * FitSummary summary(spec);
 * // ... populate summary with fit results ...
 * std::string report = generateTextReport(summary);
 * std::cout << report << std::endl;
 * ```
 */
[[nodiscard]] std::string generateTextReport(const FitSummary& summary);

}  // namespace ag::report
