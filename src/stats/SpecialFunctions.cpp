#include "ag/stats/SpecialFunctions.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace ag::stats {

namespace {

/**
 * @brief Compute the continued fraction for Q(a, z) using Lentz's method.
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

}  // anonymous namespace

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

double chi_square_ccdf(double x, double k) {
    if (x <= 0.0) {
        return 1.0;
    }
    if (k <= 0.0) {
        throw std::invalid_argument("Degrees of freedom must be positive");
    }

    // For the chi-square distribution, P(X > x) = Q(k/2, x/2),
    // the regularized upper incomplete gamma function.
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

}  // namespace ag::stats
