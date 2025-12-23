#pragma once

#include <cstddef>

namespace ag::selection {

/**
 * @brief Compute Akaike Information Criterion (AIC).
 *
 * The AIC is a measure of model quality that balances goodness of fit
 * with model complexity. It is defined as:
 *
 *   AIC = 2k - 2*loglik
 *
 * where:
 * - k is the number of estimated parameters
 * - loglik is the log-likelihood of the fitted model
 *
 * Lower AIC values indicate better models. AIC tends to favor more
 * complex models compared to BIC.
 *
 * @param log_likelihood The log-likelihood of the fitted model
 * @param k The number of estimated parameters
 * @return The AIC score
 *
 * @note This function assumes log_likelihood is the actual log-likelihood,
 *       not the negative log-likelihood. If you have negative log-likelihood
 *       (NLL), use -NLL as the log_likelihood parameter.
 */
[[nodiscard]] double computeAIC(double log_likelihood, int k);

/**
 * @brief Compute Bayesian Information Criterion (BIC).
 *
 * The BIC is a measure of model quality that balances goodness of fit
 * with model complexity. It is defined as:
 *
 *   BIC = k*log(n) - 2*loglik
 *
 * where:
 * - k is the number of estimated parameters
 * - n is the sample size
 * - loglik is the log-likelihood of the fitted model
 *
 * Lower BIC values indicate better models. BIC penalizes model complexity
 * more heavily than AIC, especially for larger sample sizes.
 *
 * @param log_likelihood The log-likelihood of the fitted model
 * @param k The number of estimated parameters
 * @param n The sample size (number of observations)
 * @return The BIC score
 *
 * @note This function assumes log_likelihood is the actual log-likelihood,
 *       not the negative log-likelihood. If you have negative log-likelihood
 *       (NLL), use -NLL as the log_likelihood parameter.
 */
[[nodiscard]] double computeBIC(double log_likelihood, int k, std::size_t n);

/**
 * @brief Compute corrected Akaike Information Criterion (AICc).
 *
 * The AICc is a corrected version of AIC that provides better performance
 * for small sample sizes. It is defined as:
 *
 *   AICc = AIC + 2k(k+1)/(n-k-1)
 *
 * where:
 * - AIC = 2k - 2*loglik
 * - k is the number of estimated parameters
 * - n is the sample size
 * - loglik is the log-likelihood of the fitted model
 *
 * As n → ∞, AICc converges to AIC. For small samples, AICc provides
 * a correction that prevents overfitting. It is recommended to use AICc
 * when n/k < 40.
 *
 * @param log_likelihood The log-likelihood of the fitted model
 * @param k The number of estimated parameters
 * @param n The sample size (number of observations)
 * @return The AICc score
 *
 * @note This function assumes log_likelihood is the actual log-likelihood,
 *       not the negative log-likelihood. If you have negative log-likelihood
 *       (NLL), use -NLL as the log_likelihood parameter.
 * @note The correction term 2k(k+1)/(n-k-1) requires n > k+1 to avoid
 *       division by zero or negative denominators.
 */
[[nodiscard]] double computeAICc(double log_likelihood, int k, std::size_t n);

}  // namespace ag::selection
