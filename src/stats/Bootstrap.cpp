#include "ag/stats/Bootstrap.hpp"

#include "ag/stats/ACF.hpp"
#include "ag/stats/Descriptive.hpp"
#include "ag/util/LinearAlgebra.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace ag::stats {

namespace {

// Numerical tolerance for matrix singularity detection
constexpr double MATRIX_SINGULARITY_TOLERANCE = 1e-10;

/**
 * @brief Resample data with replacement using a random number generator.
 *
 * @param data Original data to resample from
 * @param rng Random number generator
 * @return Vector of resampled data (same size as input)
 */
std::vector<double> resample_with_replacement(std::span<const double> data, std::mt19937& rng) {
    const std::size_t n = data.size();
    std::vector<double> resampled(n);
    std::uniform_int_distribution<std::size_t> dist(0, n - 1);

    for (std::size_t i = 0; i < n; ++i) {
        resampled[i] = data[dist(rng)];
    }

    return resampled;
}

/**
 * @brief Compute Ljung-Box Q statistic (without p-value calculation).
 *
 * This is a helper that computes just the statistic for bootstrap purposes.
 */
double compute_ljung_box_q(std::span<const double> residuals, std::size_t lags) {
    const std::size_t n = residuals.size();

    if (lags >= n) {
        throw std::invalid_argument("Number of lags must be less than sample size");
    }

    // Compute ACF for the residuals
    std::vector<double> acf_values = acf(residuals, lags);

    // Calculate the Ljung-Box Q statistic
    double q = 0.0;
    for (std::size_t k = 1; k <= lags; ++k) {
        double rho_k = acf_values[k];
        q += (rho_k * rho_k) / static_cast<double>(n - k);
    }

    q *= static_cast<double>(n * (n + 2));

    return q;
}

/**
 * @brief Fit AR model and compute residuals for sieve bootstrap.
 *
 * @param data Time series data
 * @param p AR order
 * @return Pair of (AR coefficients, residuals)
 */
std::pair<std::vector<double>, std::vector<double>> fit_ar_model(std::span<const double> data,
                                                                 std::size_t p) {
    const std::size_t n = data.size();

    if (p == 0 || n <= p) {
        // Cannot fit AR model, return empty coefficients and original data as residuals
        return {{}, std::vector<double>(data.begin(), data.end())};
    }

    // Build design matrix X and response vector y for OLS
    // y_t = φ_1 * y_{t-1} + ... + φ_p * y_{t-p} + ε_t
    const std::size_t n_obs = n - p;
    std::vector<std::vector<double>> X(n_obs, std::vector<double>(p));
    std::vector<double> y(n_obs);

    for (std::size_t t = 0; t < n_obs; ++t) {
        y[t] = data[p + t];
        for (std::size_t j = 0; j < p; ++j) {
            X[t][j] = data[p + t - j - 1];
        }
    }

    // Solve least squares using utility function
    std::vector<double> phi = ag::util::solveLeastSquares(X, y, MATRIX_SINGULARITY_TOLERANCE);

    if (phi.empty()) {
        // Singular matrix, return empty coefficients
        return {{}, std::vector<double>(data.begin(), data.end())};
    }

    // Compute residuals
    std::vector<double> residuals(n_obs);
    for (std::size_t t = 0; t < n_obs; ++t) {
        residuals[t] = y[t];
        for (std::size_t j = 0; j < p; ++j) {
            residuals[t] -= phi[j] * X[t][j];
        }
    }

    return {phi, residuals};
}

/**
 * @brief Generate bootstrap sample from AR model with resampled residuals.
 *
 * @param phi AR coefficients
 * @param residuals Residuals to resample from
 * @param n Length of series to generate
 * @param rng Random number generator
 * @return Bootstrap time series
 */
std::vector<double> generate_ar_bootstrap_sample(const std::vector<double>& phi,
                                                 std::span<const double> residuals, std::size_t n,
                                                 std::mt19937& rng) {
    const std::size_t p = phi.size();
    std::vector<double> y_star(n, 0.0);

    if (p == 0) {
        // No AR structure, just return resampled residuals
        return resample_with_replacement(residuals, rng);
    }

    // Resample residuals
    auto resampled_residuals = resample_with_replacement(residuals, rng);

    // Generate bootstrap series
    // Initialize first p values to zero (or could use actual initial values)
    for (std::size_t t = p; t < n; ++t) {
        y_star[t] = resampled_residuals[t];
        for (std::size_t j = 0; j < p; ++j) {
            y_star[t] += phi[j] * y_star[t - j - 1];
        }
    }

    return y_star;
}

/**
 * @brief Generate bootstrap sample imposing unit root null hypothesis.
 *
 * This function implements the sieve bootstrap for unit root testing by:
 * 1. Generating differences from an AR(p) model with resampled residuals
 * 2. Integrating the differences to obtain levels (imposing unit root)
 *
 * The key difference from generate_ar_bootstrap_sample is that this generates
 * an I(1) process by construction, suitable for testing the unit root null.
 *
 * @param phi_diff AR coefficients estimated from first differences
 * @param residuals Centered residuals to resample from
 * @param n Length of series to generate
 * @param rng Random number generator
 * @return Bootstrap time series with unit root imposed
 */
std::vector<double> generate_unit_root_bootstrap_sample(const std::vector<double>& phi_diff,
                                                        std::span<const double> residuals,
                                                        std::size_t n, std::mt19937& rng) {
    const std::size_t p = phi_diff.size();

    // We need to generate n points, so we need n resampled residuals
    // Resample with replacement to get exactly n residuals from the available pool
    const std::size_t n_resid = residuals.size();
    std::uniform_int_distribution<std::size_t> dist(0, n_resid - 1);
    std::vector<double> resampled_residuals(n);
    for (std::size_t i = 0; i < n; ++i) {
        resampled_residuals[i] = residuals[dist(rng)];
    }

    // Step 2: Generate differences Δy*_t from AR(p) model
    std::vector<double> dy_star(n, 0.0);

    if (p == 0) {
        // No AR structure in differences, just use resampled residuals
        dy_star = resampled_residuals;
    } else {
        // Generate AR process for differences
        // Δy*_t = φ̂₁Δy*_{t-1} + ... + φ̂ₚΔy*_{t-p} + ε*_t
        for (std::size_t t = p; t < n; ++t) {
            dy_star[t] = resampled_residuals[t];
            for (std::size_t j = 0; j < p; ++j) {
                dy_star[t] += phi_diff[j] * dy_star[t - j - 1];
            }
        }
    }

    // Step 3: Integrate differences to get levels (imposing unit root)
    // y*_t = y*_{t-1} + Δy*_t, with y*_0 = 0
    std::vector<double> y_star(n, 0.0);
    for (std::size_t t = 1; t < n; ++t) {
        y_star[t] = y_star[t - 1] + dy_star[t];
    }

    return y_star;
}

/**
 * @brief Compute ADF test statistic (without p-value calculation).
 *
 * This is a simplified version that just computes the t-statistic for φ in the
 * ADF regression: Δy_t = α + βt + φy_{t-1} + Σγ_jΔy_{t-j} + ε_t
 *
 * Returns the t-statistic for testing H0: φ = 0 (unit root).
 */
double compute_adf_statistic(std::span<const double> data, std::size_t lags,
                             ADFRegressionForm form) {
    const std::size_t n = data.size();

    if (n <= lags + 2) {
        throw std::invalid_argument("Insufficient data for ADF test");
    }

    // Compute first differences
    std::vector<double> dy(n - 1);
    for (std::size_t i = 0; i < n - 1; ++i) {
        dy[i] = data[i + 1] - data[i];
    }

    // Build regression: dy[t] = α + βt + φ*y[t-1] + Σγ_j*dy[t-j] + ε
    // We need at least lags + 1 observations for lagged differences
    const std::size_t n_obs = dy.size() - lags;

    // Count regressors
    std::size_t n_regressors = 1;  // φ (coefficient on y_{t-1})
    if (form == ADFRegressionForm::Constant || form == ADFRegressionForm::ConstantAndTrend) {
        n_regressors += 1;  // α (constant)
    }
    if (form == ADFRegressionForm::ConstantAndTrend) {
        n_regressors += 1;  // β (trend)
    }
    n_regressors += lags;  // γ_1, ..., γ_lags

    if (n_obs <= n_regressors) {
        throw std::invalid_argument("Insufficient observations for ADF regression");
    }

    // Build design matrix X and response vector y_vec
    std::vector<std::vector<double>> X(n_obs, std::vector<double>(n_regressors, 0.0));
    std::vector<double> y_vec(n_obs);

    for (std::size_t t = 0; t < n_obs; ++t) {
        const std::size_t data_idx = lags + t;
        y_vec[t] = dy[data_idx];

        std::size_t col = 0;

        // Constant term
        if (form == ADFRegressionForm::Constant || form == ADFRegressionForm::ConstantAndTrend) {
            X[t][col++] = 1.0;
        }

        // Trend term
        if (form == ADFRegressionForm::ConstantAndTrend) {
            X[t][col++] = static_cast<double>(data_idx + 1);
        }

        // y_{t-1} (level term)
        X[t][col++] = data[data_idx];

        // Lagged differences: dy[t-1], dy[t-2], ..., dy[t-lags]
        for (std::size_t j = 1; j <= lags; ++j) {
            X[t][col++] = dy[data_idx - j];
        }
    }

    // The φ coefficient (on y_{t-1}) follows the deterministic terms.
    std::size_t phi_idx = 0;
    if (form == ADFRegressionForm::Constant) {
        phi_idx = 1;
    } else if (form == ADFRegressionForm::ConstantAndTrend) {
        phi_idx = 2;
    }

    // Use the shared OLS t-statistic so the observed and bootstrap ADF
    // statistics are produced by one implementation (the bespoke Gauss-Jordan
    // inverse previously here has been removed). A singular regression on a
    // bootstrap replicate is treated as an uninformative 0, matching the prior
    // behavior, rather than aborting the whole bootstrap.
    try {
        return ag::util::olsTStatistic(X, y_vec, phi_idx, MATRIX_SINGULARITY_TOLERANCE);
    } catch (const std::exception&) {
        return 0.0;
    }
}

}  // anonymous namespace

LjungBoxResult ljung_box_test_bootstrap(std::span<const double> residuals, std::size_t lags,
                                        std::size_t dof, std::size_t n_bootstrap,
                                        unsigned int seed) {
    const std::size_t n = residuals.size();

    if (n == 0) {
        throw std::invalid_argument("Cannot compute bootstrap Ljung-Box test for empty residuals");
    }

    if (lags == 0) {
        throw std::invalid_argument("Number of lags must be positive");
    }

    if (lags >= n) {
        throw std::invalid_argument("Number of lags must be less than sample size");
    }

    if (n_bootstrap == 0) {
        throw std::invalid_argument("Number of bootstrap replications must be positive");
    }

    // Step 1: Compute observed Q statistic
    double q_observed = compute_ljung_box_q(residuals, lags);

    // Step 2: Center the residuals
    double mean = std::accumulate(residuals.begin(), residuals.end(), 0.0) / n;
    std::vector<double> centered_residuals(n);
    for (std::size_t i = 0; i < n; ++i) {
        centered_residuals[i] = residuals[i] - mean;
    }

    // Step 3: Bootstrap
    std::mt19937 rng(seed);
    std::size_t count_greater_equal = 0;

    for (std::size_t b = 0; b < n_bootstrap; ++b) {
        // Resample centered residuals with replacement
        auto resampled = resample_with_replacement(centered_residuals, rng);

        // Compute Q* on resampled data
        double q_star = compute_ljung_box_q(resampled, lags);

        // Count cases where Q* >= Q_observed
        if (q_star >= q_observed) {
            ++count_greater_equal;
        }
    }

    // Step 4: Compute bootstrap p-value
    double p_value = static_cast<double>(count_greater_equal) / static_cast<double>(n_bootstrap);

    // Step 5: Set degrees of freedom
    // If dof is 0, default to lags (no parameter adjustment)
    std::size_t effective_dof = (dof == 0) ? lags : dof;

    return LjungBoxResult{
        .statistic = q_observed,
        .p_value = p_value,
        .lags = lags,
        .dof = effective_dof,
    };
}

ADFResult adf_test_bootstrap(std::span<const double> data, std::size_t lags,
                             ADFRegressionForm regression_form, std::size_t n_bootstrap,
                             unsigned int seed) {
    const std::size_t n = data.size();

    if (n <= 10) {
        throw std::invalid_argument("Insufficient data for bootstrap ADF test (need at least 10)");
    }

    if (n_bootstrap == 0) {
        throw std::invalid_argument("Number of bootstrap replications must be positive");
    }

    // Step 1: Compute observed ADF statistic
    double tau_observed = compute_adf_statistic(data, lags, regression_form);

    // Step 2: Select AR order for sieve bootstrap
    // Use either specified lags or automatic selection (here we use specified lags)
    std::size_t ar_order = lags;
    if (ar_order == 0) {
        // If lags is 0, use a default based on sample size
        ar_order =
            std::max(1UL, static_cast<std::size_t>(std::floor(std::pow(n / 100.0, 0.25) * 12)));
        ar_order = std::min(ar_order, n / 4);
    }

    // Step 3: Take first differences (impose unit root null hypothesis)
    std::vector<double> differences(n - 1);
    for (std::size_t i = 0; i < n - 1; ++i) {
        differences[i] = data[i + 1] - data[i];
    }

    // Step 4: Fit AR model to differences (not levels)
    auto [phi_diff, residuals] = fit_ar_model(differences, ar_order);

    // Center the residuals
    double mean = std::accumulate(residuals.begin(), residuals.end(), 0.0) / residuals.size();
    std::vector<double> centered_residuals(residuals.size());
    for (std::size_t i = 0; i < residuals.size(); ++i) {
        centered_residuals[i] = residuals[i] - mean;
    }

    // Step 5: Bootstrap under unit root null
    std::mt19937 rng(seed);
    std::vector<double> bootstrap_statistics(n_bootstrap);

    for (std::size_t b = 0; b < n_bootstrap; ++b) {
        // Generate bootstrap sample imposing unit root
        auto y_star = generate_unit_root_bootstrap_sample(phi_diff, centered_residuals, n, rng);

        // Compute ADF statistic on bootstrap sample
        try {
            bootstrap_statistics[b] = compute_adf_statistic(y_star, lags, regression_form);
        } catch (const std::exception&) {
            // If computation fails, use a default value
            bootstrap_statistics[b] = 0.0;
        }
    }

    // Step 6: Compute bootstrap p-value and critical values
    // Sort bootstrap statistics
    std::sort(bootstrap_statistics.begin(), bootstrap_statistics.end());

    // P-value: proportion of τ* <= τ_observed
    // (For ADF, more negative is more evidence against unit root)
    std::size_t count_less_equal = 0;
    for (double tau_star : bootstrap_statistics) {
        if (tau_star <= tau_observed) {
            ++count_less_equal;
        }
    }

    double p_value = static_cast<double>(count_less_equal) / static_cast<double>(n_bootstrap);

    // Compute empirical critical values from bootstrap distribution
    // 1% critical value: 1st percentile
    // 5% critical value: 5th percentile
    // 10% critical value: 10th percentile
    std::size_t idx_1pct = static_cast<std::size_t>(0.01 * n_bootstrap);
    std::size_t idx_5pct = static_cast<std::size_t>(0.05 * n_bootstrap);
    std::size_t idx_10pct = static_cast<std::size_t>(0.10 * n_bootstrap);

    idx_1pct = std::min(idx_1pct, n_bootstrap - 1);
    idx_5pct = std::min(idx_5pct, n_bootstrap - 1);
    idx_10pct = std::min(idx_10pct, n_bootstrap - 1);

    double cv_1pct = bootstrap_statistics[idx_1pct];
    double cv_5pct = bootstrap_statistics[idx_5pct];
    double cv_10pct = bootstrap_statistics[idx_10pct];

    return ADFResult{
        .statistic = tau_observed,
        .p_value = p_value,
        .lags = lags,
        .regression_form = regression_form,
        .critical_value_1pct = cv_1pct,
        .critical_value_5pct = cv_5pct,
        .critical_value_10pct = cv_10pct,
    };
}

}  // namespace ag::stats
