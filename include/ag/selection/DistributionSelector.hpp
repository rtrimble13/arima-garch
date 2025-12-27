#pragma once

#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"

#include <cstddef>
#include <vector>

namespace ag::selection {

/**
 * @brief Result of distribution comparison test.
 *
 * Compares Gaussian vs Student-t innovations for ARIMA-GARCH models.
 */
struct DistributionTestResult {
    bool prefer_student_t;   // Recommendation based on tests
    double normal_ll;        // Gaussian log-likelihood
    double student_t_ll;     // Student-t log-likelihood
    double df;               // Estimated degrees of freedom
    double lr_statistic;     // Likelihood ratio test statistic
    double lr_p_value;       // LR test p-value
    double aic_improvement;  // AIC difference (positive = Student-t better)
    double bic_improvement;  // BIC difference (positive = Student-t better)
    double kurtosis;         // Excess kurtosis of standardized residuals
};

/**
 * @brief Compare Gaussian vs Student-t innovations for fitted model.
 *
 * This function:
 * 1. Computes standardized residuals from the fitted model
 * 2. Estimates Student-t degrees of freedom via maximum likelihood
 * 3. Computes log-likelihoods under both distributions
 * 4. Performs likelihood ratio test
 * 5. Compares AIC/BIC
 * 6. Checks kurtosis as supporting evidence
 *
 * Decision criteria:
 * - LR test p-value < 0.05: significant improvement with Student-t
 * - Student-t has better BIC AND excess kurtosis > 1: moderate evidence for Student-t
 *
 * Note: The function provides multiple metrics (LR test, AIC/BIC, kurtosis) to help
 * users make an informed decision. The prefer_student_t field is a recommendation
 * based on statistical significance and BIC comparison.
 *
 * @param spec Model specification
 * @param params Fitted parameters (assuming Gaussian)
 * @param data Time series data
 * @param size Number of observations
 * @return DistributionTestResult with recommendation
 */
[[nodiscard]] DistributionTestResult
compareDistributions(const ag::models::ArimaGarchSpec& spec,
                     const ag::models::composite::ArimaGarchParameters& params, const double* data,
                     std::size_t size);

/**
 * @brief Estimate Student-t degrees of freedom from standardized residuals.
 *
 * Uses maximum likelihood estimation to find optimal df parameter.
 *
 * @param standardized_residuals Vector of standardized residuals
 * @return Estimated degrees of freedom (typically between 3 and 30)
 */
[[nodiscard]] double estimateStudentTDF(const std::vector<double>& standardized_residuals);

}  // namespace ag::selection
