#include "ag/selection/DistributionSelector.hpp"

#include "ag/diagnostics/Residuals.hpp"
#include "ag/stats/Descriptive.hpp"

#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace ag::selection {

namespace {

/**
 * @brief Compute ln(Γ(x)) using Lanczos approximation.
 * Implementation borrowed from LjungBox for consistency.
 */
double log_gamma_lanczos(double x) {
    if (x <= 0.0) {
        throw std::invalid_argument("Gamma function undefined for non-positive values");
    }

    // Lanczos coefficients for g=7
    const double coef[] = {0.99999999999980993,  676.5203681218851,     -1259.1392167224028,
                           771.32342877765313,   -176.61502916214059,   12.507343278686905,
                           -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};

    if (x < 0.5) {
        // Use reflection formula: Γ(x) * Γ(1-x) = π / sin(πx)
        double sin_val = std::sin(std::numbers::pi * x);
        if (std::abs(sin_val) < 1e-15) {
            throw std::invalid_argument("Gamma function evaluation unstable for x very close to 0");
        }
        return std::log(std::numbers::pi) - std::log(std::abs(sin_val)) -
               log_gamma_lanczos(1.0 - x);
    }

    x -= 1.0;
    double sum = coef[0];
    for (int i = 1; i < 9; ++i) {
        sum += coef[i] / (x + i);
    }

    const double t = x + 7.5;
    const double log_sqrt_2pi = 0.91893853320467274178;
    return log_sqrt_2pi + std::log(sum) + (x + 0.5) * std::log(t) - t;
}

/**
 * @brief Compute continued fraction for Q(a, z) using Lentz's method.
 * Implementation borrowed from LjungBox for consistency.
 */
double continued_fraction_q(double a, double z) {
    const int max_iter = 200;
    const double eps = 1e-15;
    const double tiny = 1e-30;

    double b = z + 1.0 - a;
    double c = 1.0 / tiny;
    double d = 1.0 / b;
    double h = d;

    for (int i = 1; i <= max_iter; ++i) {
        double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (std::abs(d) < tiny)
            d = tiny;
        c = b + an / c;
        if (std::abs(c) < tiny)
            c = tiny;
        d = 1.0 / d;
        double delta = d * c;
        h *= delta;
        if (std::abs(delta - 1.0) < eps) {
            break;
        }
    }

    return h;
}

/**
 * @brief Compute the complementary chi-square CDF (upper tail probability).
 * Implementation borrowed from LjungBox for consistency.
 */
double chi_square_ccdf(double x, double k) {
    if (x <= 0.0) {
        return 1.0;
    }
    if (k <= 0.0) {
        throw std::invalid_argument("Degrees of freedom must be positive");
    }

    const double a = k / 2.0;
    const double z = x / 2.0;

    if (z > 500.0) {
        return 0.0;
    }

    double log_term = a * std::log(z) - z - log_gamma_lanczos(a);
    double cf = continued_fraction_q(a, z);
    double result = std::exp(log_term) * cf;

    result = std::max(0.0, std::min(1.0, result));

    return result;
}

/**
 * @brief Compute chi-square CDF (lower tail probability).
 */
double chi_square_cdf(double x, double k) {
    return 1.0 - chi_square_ccdf(x, k);
}

}  // anonymous namespace

double estimateStudentTDF(const std::vector<double>& std_residuals) {
    if (std_residuals.empty()) {
        throw std::invalid_argument("Cannot estimate degrees of freedom from empty residuals");
    }

    // Objective: maximize Student-t log-likelihood w.r.t. df
    auto objective = [&](double df) -> double {
        if (df <= 2.0 || df > 100.0)
            return 1e10;  // Invalid df

        double ll = 0.0;

        // Log-likelihood for Student-t(0,1,df)
        for (double z : std_residuals) {
            ll += std::lgamma((df + 1.0) / 2.0) - std::lgamma(df / 2.0) -
                  0.5 * std::log(df * std::numbers::pi) -
                  ((df + 1.0) / 2.0) * std::log(1.0 + z * z / df);
        }

        return -ll;  // Negate for minimization
    };

    // Grid search followed by optimization
    double best_df = 5.0;
    double best_ll = objective(5.0);

    // Try common df values
    for (double df : {3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0}) {
        double ll = objective(df);
        if (ll < best_ll) {
            best_ll = ll;
            best_df = df;
        }
    }

    // Fine-tune with simple golden section search
    double a = std::max(3.0, best_df - 5.0);
    double b = std::min(100.0, best_df + 5.0);
    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
    const double resphi = 2.0 - phi;
    const double tol = 1e-5;

    double x1 = a + resphi * (b - a);
    double x2 = b - resphi * (b - a);
    double f1 = objective(x1);
    double f2 = objective(x2);

    while (std::abs(b - a) > tol) {
        if (f1 < f2) {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + resphi * (b - a);
            f1 = objective(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = b - resphi * (b - a);
            f2 = objective(x2);
        }
    }

    return (a + b) / 2.0;
}

DistributionTestResult
compareDistributions(const ag::models::ArimaGarchSpec& spec,
                     const ag::models::composite::ArimaGarchParameters& params, const double* data,
                     std::size_t size) {
    if (data == nullptr) {
        throw std::invalid_argument("Data pointer cannot be null");
    }
    if (size == 0) {
        throw std::invalid_argument("Data size must be positive");
    }

    // Step 1: Compute standardized residuals
    auto residuals = ag::diagnostics::computeResiduals(spec, params, data, size);

    // Step 2: Estimate Student-t df
    double df = estimateStudentTDF(residuals.std_eps_t);

    // Step 3: Compute log-likelihoods
    double n = static_cast<double>(size);

    // Gaussian log-likelihood
    double normal_ll = -0.5 * n * std::log(2.0 * std::numbers::pi);
    for (double z : residuals.std_eps_t) {
        normal_ll -= 0.5 * z * z;
    }

    // Student-t log-likelihood
    double student_t_ll = 0.0;
    for (double z : residuals.std_eps_t) {
        student_t_ll += std::lgamma((df + 1.0) / 2.0) - std::lgamma(df / 2.0) -
                        0.5 * std::log(df * std::numbers::pi) -
                        ((df + 1.0) / 2.0) * std::log(1.0 + z * z / df);
    }

    // Step 4: Likelihood ratio test
    // H0: Gaussian (restricted), H1: Student-t (unrestricted)
    // LR = 2 * (LL_t - LL_n) ~ chi^2(1) under H0
    double lr_stat = 2.0 * (student_t_ll - normal_ll);
    double lr_p_value = 1.0 - chi_square_cdf(lr_stat, 1.0);

    // Step 5: Information criteria
    int k_normal = spec.totalParamCount();
    int k_student_t = k_normal + 1;  // Extra df parameter

    double normal_aic = -2.0 * normal_ll + 2.0 * k_normal;
    double student_t_aic = -2.0 * student_t_ll + 2.0 * k_student_t;

    double normal_bic = -2.0 * normal_ll + k_normal * std::log(n);
    double student_t_bic = -2.0 * student_t_ll + k_student_t * std::log(n);

    // Step 6: Compute kurtosis
    double kurt = ag::stats::kurtosis(residuals.std_eps_t);

    // Step 7: Make recommendation
    bool prefer_student_t = (lr_p_value < 0.05) || ((student_t_bic < normal_bic) && (kurt > 1.0));

    return DistributionTestResult{.prefer_student_t = prefer_student_t,
                                  .normal_ll = normal_ll,
                                  .student_t_ll = student_t_ll,
                                  .df = df,
                                  .lr_statistic = lr_stat,
                                  .lr_p_value = lr_p_value,
                                  .aic_improvement = normal_aic - student_t_aic,
                                  .bic_improvement = normal_bic - student_t_bic,
                                  .kurtosis = kurt};
}

}  // namespace ag::selection
