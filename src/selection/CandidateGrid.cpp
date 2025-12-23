#include "ag/selection/CandidateGrid.hpp"

#include <stdexcept>
#include <string>

namespace ag::selection {

void CandidateGridConfig::validate() const {
    if (max_p < 0) {
        throw std::invalid_argument("max_p must be non-negative, got: " + std::to_string(max_p));
    }
    if (max_d < 0) {
        throw std::invalid_argument("max_d must be non-negative, got: " + std::to_string(max_d));
    }
    if (max_q < 0) {
        throw std::invalid_argument("max_q must be non-negative, got: " + std::to_string(max_q));
    }
    if (max_p_garch < 1) {
        throw std::invalid_argument("max_p_garch must be >= 1, got: " +
                                    std::to_string(max_p_garch));
    }
    if (max_q_garch < 1) {
        throw std::invalid_argument("max_q_garch must be >= 1, got: " +
                                    std::to_string(max_q_garch));
    }
    if (restrict_pq_total && max_pq_total < 0) {
        throw std::invalid_argument("max_pq_total must be non-negative when restrict_pq_total "
                                    "is enabled, got: " +
                                    std::to_string(max_pq_total));
    }
}

CandidateGrid::CandidateGrid(const CandidateGridConfig& config) : config_(config) {
    config_.validate();
}

bool CandidateGrid::satisfiesRestrictions(int p, int d, int q) const {
    // Check d restriction: only allow d in {0, 1}
    if (config_.restrict_d_to_01 && d > 1) {
        return false;
    }

    // Check p+q restriction
    if (config_.restrict_pq_total && (p + q) > config_.max_pq_total) {
        return false;
    }

    return true;
}

size_t CandidateGrid::candidateCount() const {
    size_t count = 0;

    // Iterate through all ARIMA combinations
    for (int p = 0; p <= config_.max_p; ++p) {
        for (int d = 0; d <= config_.max_d; ++d) {
            for (int q = 0; q <= config_.max_q; ++q) {
                if (satisfiesRestrictions(p, d, q)) {
                    // For each valid ARIMA spec, count all GARCH combinations
                    // GARCH orders start from 1
                    count += static_cast<size_t>(config_.max_p_garch) *
                             static_cast<size_t>(config_.max_q_garch);
                }
            }
        }
    }

    return count;
}

std::vector<models::ArimaGarchSpec> CandidateGrid::generate() const {
    std::vector<models::ArimaGarchSpec> candidates;
    candidates.reserve(candidateCount());

    // Iterate through all ARIMA combinations (p, d, q)
    // Order: p as outer loop, d as middle, q as inner
    for (int p = 0; p <= config_.max_p; ++p) {
        for (int d = 0; d <= config_.max_d; ++d) {
            for (int q = 0; q <= config_.max_q; ++q) {
                // Check if this ARIMA spec satisfies restrictions
                if (!satisfiesRestrictions(p, d, q)) {
                    continue;
                }

                // For each valid ARIMA spec, generate all GARCH combinations
                // GARCH orders start from 1 (p_garch, q_garch both must be >= 1)
                for (int p_garch = 1; p_garch <= config_.max_p_garch; ++p_garch) {
                    for (int q_garch = 1; q_garch <= config_.max_q_garch; ++q_garch) {
                        candidates.emplace_back(p, d, q, p_garch, q_garch);
                    }
                }
            }
        }
    }

    return candidates;
}

}  // namespace ag::selection
