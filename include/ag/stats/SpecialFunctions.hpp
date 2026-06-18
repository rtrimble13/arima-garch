/**
 * @file stats/SpecialFunctions.hpp
 * @brief Shared numerical special functions for statistical tests.
 *
 * Provides a single implementation of the gamma / chi-square tail routines that
 * were previously copy-pasted across JarqueBera, LjungBox, and
 * DistributionSelector. Keeping them in one place avoids the accuracy drift that
 * arises when the copies are maintained independently.
 */

#pragma once

namespace ag::stats {

/**
 * @brief Compute ln(Γ(x)) using the Lanczos approximation (g = 7).
 *
 * Uses the reflection formula for x < 0.5.
 *
 * @param x Argument (must be positive after reflection; throws otherwise)
 * @return ln(Γ(x))
 * @throws std::invalid_argument for non-positive arguments or values where the
 *         reflection formula is numerically unstable (x very close to 0).
 */
double log_gamma_lanczos(double x);

/**
 * @brief Complementary chi-square CDF (upper-tail probability).
 *
 * Returns P(X > x) where X ~ χ²(k), i.e. the regularized upper incomplete gamma
 * function Q(k/2, x/2). This is the quantity needed for a chi-square p-value.
 *
 * @param x Test statistic value
 * @param k Degrees of freedom (must be positive)
 * @return Upper-tail probability in [0, 1]
 * @throws std::invalid_argument if k <= 0
 */
double chi_square_ccdf(double x, double k);

}  // namespace ag::stats
