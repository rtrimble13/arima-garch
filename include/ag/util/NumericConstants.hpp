#pragma once

namespace ag::util {

// Lower bound applied to GARCH conditional variances to guard against
// numerical underflow during recursion. Positivity of h_t is enforced
// upstream by parameter constraints; this is purely a numerical floor.
inline constexpr double MIN_VARIANCE = 1e-10;

}  // namespace ag::util
