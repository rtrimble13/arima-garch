#pragma once

#include "ag/models/ArimaGarchSpec.hpp"

#include <vector>

namespace ag::selection {

/**
 * @brief Configuration for candidate grid generation.
 *
 * Defines the maximum orders and optional restrictions for generating
 * a grid of candidate ARIMA-GARCH model specifications.
 */
struct CandidateGridConfig {
    int max_p;        // Maximum ARIMA AR order
    int max_d;        // Maximum ARIMA differencing degree
    int max_q;        // Maximum ARIMA MA order
    int max_p_garch;  // Maximum GARCH order
    int max_q_garch;  // Maximum ARCH order

    // Optional restrictions
    bool restrict_d_to_01;   // If true, only allow d in {0, 1}
    bool restrict_pq_total;  // If true, enforce p + q <= max_pq_total
    int max_pq_total;  // Maximum sum of ARIMA p and q (only used if restrict_pq_total is true)

    /**
     * @brief Construct a candidate grid configuration with default restrictions disabled.
     * @param p Maximum ARIMA AR order (must be >= 0)
     * @param d Maximum ARIMA differencing degree (must be >= 0)
     * @param q Maximum ARIMA MA order (must be >= 0)
     * @param p_garch Maximum GARCH order (must be >= 1)
     * @param q_garch Maximum ARCH order (must be >= 1)
     */
    CandidateGridConfig(int p, int d, int q, int p_garch, int q_garch)
        : max_p(p), max_d(d), max_q(q), max_p_garch(p_garch), max_q_garch(q_garch),
          restrict_d_to_01(false), restrict_pq_total(false), max_pq_total(0) {
        validate();
    }

    /**
     * @brief Validate the configuration parameters.
     * @throws std::invalid_argument if parameters are invalid
     */
    void validate() const;
};

/**
 * @brief Generate a grid of candidate ARIMA-GARCH specifications.
 *
 * CandidateGrid generates all valid combinations of ARIMA(p,d,q)-GARCH(P,Q)
 * specifications within the bounds and restrictions specified in the configuration.
 *
 * The generation is deterministic: given the same configuration, the same
 * candidate list will be generated in the same order.
 */
class CandidateGrid {
public:
    /**
     * @brief Construct a CandidateGrid with the specified configuration.
     * @param config Configuration specifying bounds and restrictions
     */
    explicit CandidateGrid(const CandidateGridConfig& config);

    /**
     * @brief Generate all candidate specifications.
     * @return Vector of candidate ARIMA-GARCH specifications
     *
     * The candidates are generated in a deterministic order:
     * - ARIMA orders iterate with p as the outer loop, d as middle, q as inner
     * - GARCH orders iterate with p_garch as outer loop, q_garch as inner
     * - ARIMA combinations are iterated first, then GARCH combinations
     */
    [[nodiscard]] std::vector<models::ArimaGarchSpec> generate() const;

    /**
     * @brief Get the number of candidate specifications that will be generated.
     * @return Count of candidates matching the configuration
     */
    [[nodiscard]] size_t candidateCount() const;

private:
    CandidateGridConfig config_;

    /**
     * @brief Check if an ARIMA specification satisfies the restrictions.
     * @param p ARIMA AR order
     * @param d ARIMA differencing degree
     * @param q ARIMA MA order
     * @return true if the specification satisfies all restrictions
     */
    [[nodiscard]] bool satisfiesRestrictions(int p, int d, int q) const;
};

}  // namespace ag::selection
