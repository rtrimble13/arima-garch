#pragma once

#include <stdexcept>
#include <string>

namespace ag::estimation {

/**
 * @brief Innovation distribution type for likelihood and simulation.
 *
 * Kept as the canonical enumeration. The simulation namespace exposes an
 * alias so legacy call sites continue to compile.
 */
enum class InnovationDistribution {
    Normal,   // Standard normal N(0,1)
    StudentT  // Standardized Student-t with df degrees of freedom
};

/**
 * @brief Value type bundling an innovation distribution with its
 *        distribution-specific parameters.
 *
 * Replaces the unstructured (bool use_student_t, double df) parameter pair
 * that previously appeared in fit, simulate, and diagnostics call sites.
 * The invariant "df is only meaningful for StudentT" is encoded by
 * construction: the factory functions are the only well-formed entry
 * points, and ::studentT validates df >= 2.0001 up-front (any value <=2
 * yields infinite variance for the standardized Student-t parameterization
 * used here).
 */
struct InnovationSpec {
    InnovationDistribution type = InnovationDistribution::Normal;
    double df = 0.0;  // Only meaningful when type == StudentT.

    [[nodiscard]] static InnovationSpec normal() noexcept {
        return InnovationSpec{InnovationDistribution::Normal, 0.0};
    }

    [[nodiscard]] static InnovationSpec studentT(double df) {
        if (!(df > 2.0)) {
            throw std::invalid_argument(
                "InnovationSpec::studentT: degrees of freedom must be > 2.0, got " +
                std::to_string(df));
        }
        return InnovationSpec{InnovationDistribution::StudentT, df};
    }

    [[nodiscard]] bool isStudentT() const noexcept {
        return type == InnovationDistribution::StudentT;
    }
};

}  // namespace ag::estimation
