#include "ag/stats/JarqueBera.hpp"

#include "ag/stats/Descriptive.hpp"

#include <cmath>
#include <stdexcept>

namespace ag::stats {

namespace {

/**
 * @brief Compute ln(Γ(x)) using Lanczos approximation.
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
        // For numerical stability, avoid values very close to 0 where sin(πx) ≈ 0
        double sin_val = std::sin(M_PI * x);
        if (std::abs(sin_val) < 1e-15) {
            throw std::invalid_argument("Gamma function evaluation unstable for x very close to 0");
        }
        return std::log(M_PI) - std::log(std::abs(sin_val)) - log_gamma_lanczos(1.0 - x);
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
 */
double continued_fraction_q(double a, double z) {
    const int max_iter = 200;
    const double eps = 1e-15;
    const double tiny = 1e-30;

    // Continued fraction: Q = e^(-z) * z^a / Γ(a) * (1/(z+1-a+1/(1+1/(z+3-a+...))))
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
 *
 * Returns P(X > x) where X ~ χ²(k).
 *
 * Uses the regularized incomplete gamma function Q(k/2, x/2).
 * For chi-square with k degrees of freedom, P(X > x) = Q(k/2, x/2).
 *
 * Implementation uses continued fraction for better numerical stability.
 *
 * @param x The test statistic value
 * @param k Degrees of freedom
 * @return Upper tail probability (p-value)
 */
double chi_square_ccdf(double x, double k) {
    if (x <= 0.0) {
        return 1.0;
    }
    if (k <= 0.0) {
        throw std::invalid_argument("Degrees of freedom must be positive");
    }

    // For chi-square distribution, we need Q(k/2, x/2)
    // where Q is the regularized upper incomplete gamma function
    const double a = k / 2.0;
    const double z = x / 2.0;

    // Handle edge cases
    if (z > 500.0) {
        return 0.0;  // Very large x, p-value is essentially 0
    }

    // Compute Q(a, z) = e^(-z) * z^a / Γ(a) * continued_fraction
    double log_term = a * std::log(z) - z - log_gamma_lanczos(a);
    double cf = continued_fraction_q(a, z);
    double result = std::exp(log_term) * cf;

    // Clamp to [0, 1] to handle numerical errors
    result = std::max(0.0, std::min(1.0, result));

    return result;
}

}  // anonymous namespace

double jarque_bera_statistic(std::span<const double> data) {
    const std::size_t n = data.size();

    if (n < 4) {
        throw std::invalid_argument(
            "Cannot compute Jarque-Bera statistic with fewer than 4 data points");
    }

    // Compute sample skewness and kurtosis
    double S = skewness(data);
    double K = kurtosis(data);

    // Calculate the Jarque-Bera statistic
    // JB = n/6 * (S² + K²/4)
    double jb = (static_cast<double>(n) / 6.0) * (S * S + (K * K) / 4.0);

    return jb;
}

JarqueBeraResult jarque_bera_test(std::span<const double> data) {
    // Compute the test statistic
    double jb = jarque_bera_statistic(data);

    // Compute p-value using chi-square distribution with 2 degrees of freedom
    double p_value = chi_square_ccdf(jb, 2.0);

    return JarqueBeraResult{
        .statistic = jb,
        .p_value = p_value,
    };
}

}  // namespace ag::stats
