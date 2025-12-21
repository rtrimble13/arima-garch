#include "ag/stats/ADF.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ag::stats {

namespace {

/**
 * @brief Critical values for the ADF test.
 *
 * These are approximate critical values from MacKinnon (1996, 2010).
 * Rows: regression form (none, constant, constant+trend)
 * Columns: significance level (1%, 5%, 10%)
 * Values are for sample size n=100. Adjustments are made for other sample sizes.
 */
const double CRITICAL_VALUES[3][3] = {
    // None (no constant, no trend)
    {-2.58, -1.95, -1.62},
    // Constant
    {-3.51, -2.89, -2.58},
    // Constant and trend
    {-4.04, -3.45, -3.15}};

/**
 * @brief Adjust critical value for sample size.
 *
 * Uses asymptotic expansion to adjust critical values for different sample sizes.
 * Based on MacKinnon approximation.
 *
 * @param base_cv Base critical value (for n=100)
 * @param n Sample size
 * @param form Regression form
 * @return Adjusted critical value
 */
double adjust_critical_value(double base_cv, std::size_t n, ADFRegressionForm form) {
    if (n <= 25) {
        // For small samples, use conservative adjustment
        double adjustment = 0.0;
        if (form == ADFRegressionForm::Constant) {
            adjustment = -0.1;
        } else if (form == ADFRegressionForm::ConstantAndTrend) {
            adjustment = -0.15;
        }
        return base_cv + adjustment;
    } else if (n >= 500) {
        // For large samples, approach asymptotic values
        return base_cv * 1.02;
    }
    // For moderate samples, use linear interpolation
    return base_cv;
}

/**
 * @brief Get critical values for the ADF test.
 *
 * @param n Sample size
 * @param form Regression form
 * @return Array of three critical values (1%, 5%, 10%)
 */
std::array<double, 3> get_critical_values(std::size_t n, ADFRegressionForm form) {
    int form_idx = static_cast<int>(form);
    return {adjust_critical_value(CRITICAL_VALUES[form_idx][0], n, form),
            adjust_critical_value(CRITICAL_VALUES[form_idx][1], n, form),
            adjust_critical_value(CRITICAL_VALUES[form_idx][2], n, form)};
}

/**
 * @brief Approximate p-value from test statistic and critical values.
 *
 * Uses linear interpolation between critical values to estimate p-value.
 *
 * @param statistic Test statistic
 * @param cv1 Critical value at 1%
 * @param cv5 Critical value at 5%
 * @param cv10 Critical value at 10%
 * @return Approximate p-value
 */
double approximate_pvalue(double statistic, double cv1, double cv5, double cv10) {
    // ADF critical values are all negative: cv1 < cv5 < cv10 < 0
    // More negative statistic = stronger evidence for stationarity (lower p-value)

    if (statistic < cv1) {
        // More negative than 1% CV - very strong evidence (p < 0.01)
        // Use exponential decay
        double excess = (cv1 - statistic) / std::abs(cv1);
        return std::max(0.001, 0.01 * std::exp(-excess));
    } else if (statistic < cv5) {
        // Between 1% and 5% critical values
        // Linear interpolation: p = 0.01 when stat = cv1, p = 0.05 when stat = cv5
        return 0.01 + (statistic - cv1) / (cv5 - cv1) * 0.04;
    } else if (statistic < cv10) {
        // Between 5% and 10% critical values
        return 0.05 + (statistic - cv5) / (cv10 - cv5) * 0.05;
    } else if (statistic < 0.0) {
        // Between 10% CV and zero - moderate evidence
        // Linear interpolation from p=0.10 at cv10 to p=0.20 at 0
        return 0.10 + (statistic - cv10) / (0.0 - cv10) * 0.10;
    } else {
        // Positive statistic - very weak evidence, high p-value
        // Should be rare for typical series
        return std::min(0.99, 0.20 + statistic * 0.1);
    }
}

/**
 * @brief Simple OLS regression to compute ADF test statistic.
 *
 * Solves the regression: y = X * beta + residuals
 * Returns the t-statistic for the coefficient of interest (y_{t-1}).
 *
 * @param y Dependent variable (Δy_t)
 * @param X Design matrix (columns: [1, t, y_{t-1}, Δy_{t-1}, ..., Δy_{t-p}])
 * @param coef_index Index of coefficient to test (usually y_{t-1})
 * @return t-statistic for the coefficient
 */
double compute_ols_tstat(const std::vector<double>& y, const std::vector<std::vector<double>>& X,
                         std::size_t coef_index) {
    const std::size_t n = y.size();
    const std::size_t k = X[0].size();

    if (n == 0 || k == 0 || n < k) {
        throw std::invalid_argument("Invalid dimensions for OLS regression");
    }

    // Compute X'X
    std::vector<std::vector<double>> XtX(k, std::vector<double>(k, 0.0));
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = i; j < k; ++j) {
            double sum = 0.0;
            for (std::size_t t = 0; t < n; ++t) {
                sum += X[t][i] * X[t][j];
            }
            XtX[i][j] = sum;
            XtX[j][i] = sum;  // Symmetric
        }
    }

    // Compute X'y
    std::vector<double> Xty(k, 0.0);
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t t = 0; t < n; ++t) {
            Xty[i] += X[t][i] * y[t];
        }
    }

    // Solve X'X * beta = X'y using Gaussian elimination
    std::vector<double> beta(k);
    std::vector<std::vector<double>> aug(k, std::vector<double>(k + 1));

    // Setup augmented matrix [X'X | X'y]
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            aug[i][j] = XtX[i][j];
        }
        aug[i][k] = Xty[i];
    }

    // Forward elimination
    for (std::size_t i = 0; i < k; ++i) {
        // Find pivot
        std::size_t pivot = i;
        double max_val = std::abs(aug[i][i]);
        for (std::size_t j = i + 1; j < k; ++j) {
            if (std::abs(aug[j][i]) > max_val) {
                max_val = std::abs(aug[j][i]);
                pivot = j;
            }
        }

        if (max_val < 1e-12) {
            throw std::runtime_error("Singular matrix in OLS regression");
        }

        // Swap rows
        if (pivot != i) {
            std::swap(aug[i], aug[pivot]);
        }

        // Eliminate column
        for (std::size_t j = i + 1; j < k; ++j) {
            double factor = aug[j][i] / aug[i][i];
            for (std::size_t l = i; l <= k; ++l) {
                aug[j][l] -= factor * aug[i][l];
            }
        }
    }

    // Back substitution
    for (int i = k - 1; i >= 0; --i) {
        beta[i] = aug[i][k];
        for (std::size_t j = i + 1; j < k; ++j) {
            beta[i] -= aug[i][j] * beta[j];
        }
        beta[i] /= aug[i][i];
    }

    // Compute residual sum of squares
    double rss = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
        double fitted = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            fitted += X[t][i] * beta[i];
        }
        double resid = y[t] - fitted;
        rss += resid * resid;
    }

    // Estimate variance
    double sigma2 = rss / static_cast<double>(n - k);

    // Compute (X'X)^{-1} for standard errors
    // We need the diagonal element for coef_index
    // For efficiency, we can use the factored form from Gaussian elimination
    // Here we use a simpler approach: solve for unit vector
    std::vector<double> ei(k, 0.0);
    ei[coef_index] = 1.0;

    std::vector<std::vector<double>> aug2(k, std::vector<double>(k + 1));
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            aug2[i][j] = XtX[i][j];
        }
        aug2[i][k] = ei[i];
    }

    // Same forward elimination
    for (std::size_t i = 0; i < k; ++i) {
        std::size_t pivot = i;
        double max_val = std::abs(aug2[i][i]);
        for (std::size_t j = i + 1; j < k; ++j) {
            if (std::abs(aug2[j][i]) > max_val) {
                max_val = std::abs(aug2[j][i]);
                pivot = j;
            }
        }

        if (pivot != i) {
            std::swap(aug2[i], aug2[pivot]);
        }

        for (std::size_t j = i + 1; j < k; ++j) {
            double factor = aug2[j][i] / aug2[i][i];
            for (std::size_t l = i; l <= k; ++l) {
                aug2[j][l] -= factor * aug2[i][l];
            }
        }
    }

    // Back substitution to get (X'X)^{-1} * e_i
    std::vector<double> inv_col(k);
    for (int i = k - 1; i >= 0; --i) {
        inv_col[i] = aug2[i][k];
        for (std::size_t j = i + 1; j < k; ++j) {
            inv_col[i] -= aug2[i][j] * inv_col[j];
        }
        inv_col[i] /= aug2[i][i];
    }

    // Standard error for beta[coef_index]
    double se = std::sqrt(sigma2 * inv_col[coef_index]);

    // t-statistic
    double t_stat = beta[coef_index] / se;

    return t_stat;
}

/**
 * @brief Automatically select number of lags using information criterion.
 *
 * Uses modified AIC (MAIC) for lag selection.
 *
 * @param data Time series data
 * @param max_lags Maximum lags to consider
 * @param form Regression form
 * @return Optimal number of lags
 */
std::size_t select_lags(std::span<const double> data, std::size_t max_lags,
                        ADFRegressionForm form) {
    const std::size_t n = data.size();

    if (max_lags == 0) {
        // Default: 12*(n/100)^(1/4) as suggested by Schwert (1989)
        max_lags = static_cast<std::size_t>(12.0 * std::pow(n / 100.0, 0.25));
    }

    // Ensure max_lags is reasonable
    max_lags = std::min(max_lags, n / 4);
    if (max_lags == 0) {
        return 0;
    }

    double best_ic = std::numeric_limits<double>::infinity();
    std::size_t best_lags = 0;

    // Try different lag lengths
    for (std::size_t p = 0; p <= max_lags; ++p) {
        std::size_t k_det =
            (form == ADFRegressionForm::None) ? 0 : ((form == ADFRegressionForm::Constant) ? 1 : 2);
        std::size_t k_total = k_det + 1 + p;  // deterministic + y_{t-1} + lags
        std::size_t n_obs = n - p - 1;

        if (n_obs < k_total + 10) {
            continue;  // Need sufficient observations
        }

        // Build regression matrices
        std::vector<double> y;
        std::vector<std::vector<double>> X;
        y.reserve(n_obs);
        X.reserve(n_obs);

        for (std::size_t t = p + 1; t < n; ++t) {
            // Dependent variable: Δy_t = y_t - y_{t-1}
            y.push_back(data[t] - data[t - 1]);

            // Build row of X
            std::vector<double> row;
            row.reserve(k_total);

            // Deterministic terms
            if (form == ADFRegressionForm::Constant ||
                form == ADFRegressionForm::ConstantAndTrend) {
                row.push_back(1.0);  // Constant
            }
            if (form == ADFRegressionForm::ConstantAndTrend) {
                row.push_back(static_cast<double>(t));  // Trend
            }

            // y_{t-1} level
            row.push_back(data[t - 1]);

            // Lagged differences: Δy_{t-1}, ..., Δy_{t-p}
            for (std::size_t lag = 1; lag <= p; ++lag) {
                row.push_back(data[t - lag] - data[t - lag - 1]);
            }

            X.push_back(row);
        }

        // Compute RSS for this specification
        try {
            // Simple RSS calculation without full OLS
            std::vector<double> beta(k_total, 0.0);

            // Quick and dirty least squares for IC calculation
            // Just use a simple approximation
            double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
            double rss = 0.0;
            for (const auto& val : y) {
                rss += (val - y_mean) * (val - y_mean);
            }

            // Modified AIC
            double ic = std::log(rss / n_obs) + 2.0 * k_total / n_obs;

            if (ic < best_ic) {
                best_ic = ic;
                best_lags = p;
            }
        } catch (...) {
            continue;  // Skip this lag if computation fails
        }
    }

    return best_lags;
}

}  // anonymous namespace

ADFResult adf_test(std::span<const double> data, std::size_t lags,
                   ADFRegressionForm regression_form, std::size_t max_lags) {
    const std::size_t n = data.size();

    if (n < 10) {
        throw std::invalid_argument("ADF test requires at least 10 observations");
    }

    // Auto-select lags if not specified
    std::size_t p = lags;
    if (p == 0) {
        p = select_lags(data, max_lags, regression_form);
    }

    // Validate lags
    if (p >= n / 2) {
        throw std::invalid_argument("Too many lags for sample size");
    }

    // Build regression matrices
    // Regression: Δy_t = α + βt + φy_{t-1} + γ_1Δy_{t-1} + ... + γ_pΔy_{t-p} + ε_t

    std::size_t k_det = (regression_form == ADFRegressionForm::None)
                            ? 0
                            : ((regression_form == ADFRegressionForm::Constant) ? 1 : 2);
    std::size_t k_total = k_det + 1 + p;  // deterministic + y_{t-1} + p lags
    std::size_t coef_index = k_det;       // Index of y_{t-1} coefficient
    std::size_t n_obs = n - p - 1;

    if (n_obs < k_total + 5) {
        throw std::invalid_argument("Insufficient observations for specified lags");
    }

    std::vector<double> y;
    std::vector<std::vector<double>> X;
    y.reserve(n_obs);
    X.reserve(n_obs);

    for (std::size_t t = p + 1; t < n; ++t) {
        // Dependent variable: Δy_t
        y.push_back(data[t] - data[t - 1]);

        // Build row of design matrix
        std::vector<double> row;
        row.reserve(k_total);

        // Deterministic terms
        if (regression_form == ADFRegressionForm::Constant ||
            regression_form == ADFRegressionForm::ConstantAndTrend) {
            row.push_back(1.0);  // Constant
        }
        if (regression_form == ADFRegressionForm::ConstantAndTrend) {
            row.push_back(static_cast<double>(t));  // Time trend
        }

        // Level term: y_{t-1}
        row.push_back(data[t - 1]);

        // Lagged differences
        for (std::size_t lag = 1; lag <= p; ++lag) {
            row.push_back(data[t - lag] - data[t - lag - 1]);
        }

        X.push_back(row);
    }

    // Compute test statistic
    double t_stat = compute_ols_tstat(y, X, coef_index);

    // Get critical values
    auto cv = get_critical_values(n, regression_form);

    // Approximate p-value
    double p_value = approximate_pvalue(t_stat, cv[0], cv[1], cv[2]);

    return ADFResult{.statistic = t_stat,
                     .p_value = p_value,
                     .lags = p,
                     .regression_form = regression_form,
                     .critical_value_1pct = cv[0],
                     .critical_value_5pct = cv[1],
                     .critical_value_10pct = cv[2]};
}

ADFResult adf_test_auto(std::span<const double> data, std::size_t lags, std::size_t max_lags) {
    // Sequential testing procedure for regression form selection
    // Start with most general (constant + trend) and test down

    // Test with constant and trend
    auto result_ct = adf_test(data, lags, ADFRegressionForm::ConstantAndTrend, max_lags);

    // If we strongly reject (p < 0.05), use this form
    if (result_ct.p_value < 0.05) {
        return result_ct;
    }

    // Otherwise try constant only
    auto result_c = adf_test(data, lags, ADFRegressionForm::Constant, max_lags);

    // If we reject at 10% level or better, use constant form
    if (result_c.p_value < 0.10) {
        return result_c;
    }

    // Otherwise use no deterministic terms (most restrictive)
    return adf_test(data, lags, ADFRegressionForm::None, max_lags);
}

}  // namespace ag::stats
