#pragma once

#include "ag/diagnostics/DiagnosticReport.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"

#include <optional>
#include <string>
#include <vector>

namespace ag::report {

/**
 * @brief Comparison between Normal and Student-T distribution fits.
 *
 * DistributionComparison contains statistics from fitting a model with both
 * Normal and Student-T innovations, enabling comparison of the two distributions.
 * This is useful for determining whether heavy-tailed Student-T distribution
 * provides a better fit to the data than the Normal distribution.
 */
struct DistributionComparison {
    /**
     * @brief Log-likelihood under Normal distribution assumption.
     */
    double normal_log_likelihood;

    /**
     * @brief Log-likelihood under Student-T distribution assumption.
     */
    double student_t_log_likelihood;

    /**
     * @brief Estimated degrees of freedom for Student-T distribution.
     *
     * Lower values indicate heavier tails. Values close to infinity approach Normal.
     */
    double student_t_df;

    /**
     * @brief Likelihood ratio test statistic.
     *
     * LR = 2 * (LL_studentT - LL_normal)
     * Under H0 (Normal is adequate), LR follows chi-squared distribution with 1 df.
     */
    double lr_statistic;

    /**
     * @brief P-value for likelihood ratio test.
     *
     * Tests null hypothesis that Normal distribution is adequate.
     * Low p-value (< 0.05) suggests Student-T provides significantly better fit.
     */
    double lr_p_value;

    /**
     * @brief Whether Student-T is preferred based on statistical tests.
     *
     * Based on likelihood ratio test and information criteria.
     * True if Student-T provides significantly better fit.
     */
    bool prefer_student_t;

    /**
     * @brief AIC for Normal distribution fit.
     */
    double normal_aic;

    /**
     * @brief AIC for Student-T distribution fit.
     */
    double student_t_aic;

    /**
     * @brief BIC for Normal distribution fit.
     */
    double normal_bic;

    /**
     * @brief BIC for Student-T distribution fit.
     */
    double student_t_bic;
};

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
     * @brief Optional distribution comparison results.
     *
     * If the model was fitted with both Normal and Student-T distributions,
     * the comparison statistics are stored here. This helps determine whether
     * heavy-tailed Student-T distribution is more appropriate for the data.
     */
    std::optional<DistributionComparison> distribution_comparison;

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
 * 5. Innovation Distribution Comparison (if available):
 *    - Gaussian vs. Student-t log-likelihoods
 *    - Estimated degrees of freedom for Student-t
 *    - Likelihood Ratio Test results
 *    - Information Criteria (AIC/BIC) for both distributions
 *    - Recommendation on which distribution to use
 *
 * 6. Diagnostic Tests (if available):
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
