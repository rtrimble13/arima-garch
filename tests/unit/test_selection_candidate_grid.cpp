#include "ag/selection/CandidateGrid.hpp"

#include <algorithm>
#include <stdexcept>

#include "test_framework.hpp"

using ag::models::ArimaGarchSpec;
using ag::selection::CandidateGrid;
using ag::selection::CandidateGridConfig;

// ============================================================================
// CandidateGridConfig Tests
// ============================================================================

// Test valid configuration
TEST(candidate_grid_config_valid) {
    CandidateGridConfig config(2, 1, 2, 1, 1);
    REQUIRE(config.max_p == 2);
    REQUIRE(config.max_d == 1);
    REQUIRE(config.max_q == 2);
    REQUIRE(config.max_p_garch == 1);
    REQUIRE(config.max_q_garch == 1);
    REQUIRE(!config.restrict_d_to_01);
    REQUIRE(!config.restrict_pq_total);
}

// Test invalid configuration: negative max_p
TEST(candidate_grid_config_negative_max_p) {
    bool caught_exception = false;
    try {
        CandidateGridConfig config(-1, 1, 1, 1, 1);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("max_p") != std::string::npos);
        REQUIRE(msg.find("non-negative") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// Test invalid configuration: negative max_d
TEST(candidate_grid_config_negative_max_d) {
    bool caught_exception = false;
    try {
        CandidateGridConfig config(1, -1, 1, 1, 1);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("max_d") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// Test invalid configuration: negative max_q
TEST(candidate_grid_config_negative_max_q) {
    bool caught_exception = false;
    try {
        CandidateGridConfig config(1, 1, -1, 1, 1);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("max_q") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// Test invalid configuration: max_p_garch < 1
TEST(candidate_grid_config_zero_max_p_garch) {
    bool caught_exception = false;
    try {
        CandidateGridConfig config(1, 1, 1, 0, 1);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("max_p_garch") != std::string::npos);
        REQUIRE(msg.find(">= 1") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// Test invalid configuration: max_q_garch < 1
TEST(candidate_grid_config_zero_max_q_garch) {
    bool caught_exception = false;
    try {
        CandidateGridConfig config(1, 1, 1, 1, 0);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("max_q_garch") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// ============================================================================
// CandidateGrid Basic Generation Tests
// ============================================================================

// Test minimal grid: ARIMA(0,0,0)-GARCH(1,1) only
TEST(candidate_grid_minimal) {
    CandidateGridConfig config(0, 0, 0, 1, 1);
    CandidateGrid grid(config);

    // Should generate only 1 candidate: (0,0,0)-(1,1)
    REQUIRE(grid.candidateCount() == 1);

    auto candidates = grid.generate();
    REQUIRE(candidates.size() == 1);

    const auto& spec = candidates[0];
    REQUIRE(spec.arimaSpec.p == 0);
    REQUIRE(spec.arimaSpec.d == 0);
    REQUIRE(spec.arimaSpec.q == 0);
    REQUIRE(spec.garchSpec.p == 1);
    REQUIRE(spec.garchSpec.q == 1);
}

// Test small grid without restrictions
TEST(candidate_grid_small_no_restrictions) {
    CandidateGridConfig config(1, 1, 1, 1, 1);
    CandidateGrid grid(config);

    // ARIMA: (p,d,q) with p in {0,1}, d in {0,1}, q in {0,1} -> 2*2*2 = 8 combinations
    // GARCH: (p_g,q_g) with p_g=1, q_g=1 -> 1*1 = 1 combination
    // Total: 8 * 1 = 8
    REQUIRE(grid.candidateCount() == 8);

    auto candidates = grid.generate();
    REQUIRE(candidates.size() == 8);
}

// Test grid with multiple GARCH orders
TEST(candidate_grid_multiple_garch_orders) {
    CandidateGridConfig config(0, 0, 0, 2, 2);
    CandidateGrid grid(config);

    // ARIMA: only (0,0,0) -> 1 combination
    // GARCH: p_g in {1,2}, q_g in {1,2} -> 2*2 = 4 combinations
    // Total: 1 * 4 = 4
    REQUIRE(grid.candidateCount() == 4);

    auto candidates = grid.generate();
    REQUIRE(candidates.size() == 4);

    // Verify all GARCH combinations are present
    std::vector<std::pair<int, int>> garch_orders;
    for (const auto& spec : candidates) {
        REQUIRE(spec.arimaSpec.p == 0);
        REQUIRE(spec.arimaSpec.d == 0);
        REQUIRE(spec.arimaSpec.q == 0);
        garch_orders.emplace_back(spec.garchSpec.p, spec.garchSpec.q);
    }

    // Check that we have all expected GARCH combinations
    REQUIRE(std::find(garch_orders.begin(), garch_orders.end(), std::make_pair(1, 1)) !=
            garch_orders.end());
    REQUIRE(std::find(garch_orders.begin(), garch_orders.end(), std::make_pair(1, 2)) !=
            garch_orders.end());
    REQUIRE(std::find(garch_orders.begin(), garch_orders.end(), std::make_pair(2, 1)) !=
            garch_orders.end());
    REQUIRE(std::find(garch_orders.begin(), garch_orders.end(), std::make_pair(2, 2)) !=
            garch_orders.end());
}

// Test larger grid
TEST(candidate_grid_larger) {
    CandidateGridConfig config(2, 1, 2, 1, 1);
    CandidateGrid grid(config);

    // ARIMA: p in {0,1,2}, d in {0,1}, q in {0,1,2} -> 3*2*3 = 18 combinations
    // GARCH: p_g=1, q_g=1 -> 1 combination
    // Total: 18 * 1 = 18
    REQUIRE(grid.candidateCount() == 18);

    auto candidates = grid.generate();
    REQUIRE(candidates.size() == 18);
}

// ============================================================================
// CandidateGrid Restriction Tests
// ============================================================================

// Test restriction: d in {0, 1}
TEST(candidate_grid_restrict_d_to_01) {
    CandidateGridConfig config(1, 2, 1, 1, 1);
    config.restrict_d_to_01 = true;

    CandidateGrid grid(config);

    // Without restriction: p in {0,1}, d in {0,1,2}, q in {0,1} -> 2*3*2 = 12
    // With restriction d in {0,1}: p in {0,1}, d in {0,1}, q in {0,1} -> 2*2*2 = 8
    REQUIRE(grid.candidateCount() == 8);

    auto candidates = grid.generate();
    REQUIRE(candidates.size() == 8);

    // Verify that all candidates have d in {0, 1}
    for (const auto& spec : candidates) {
        REQUIRE(spec.arimaSpec.d <= 1);
    }
}

// Test restriction: p + q <= max_pq_total
TEST(candidate_grid_restrict_pq_total) {
    CandidateGridConfig config(2, 0, 2, 1, 1);
    config.restrict_pq_total = true;
    config.max_pq_total = 2;

    CandidateGrid grid(config);

    // Without restriction: p in {0,1,2}, d=0, q in {0,1,2} -> 3*1*3 = 9
    // With restriction p+q <= 2:
    //   (0,0,0), (0,0,1), (0,0,2),  // p=0: q in {0,1,2}, all valid (0+q <= 2)
    //   (1,0,0), (1,0,1),            // p=1: q in {0,1} (1+q <= 2)
    //   (2,0,0)                       // p=2: q=0 only (2+q <= 2)
    // Total: 6 ARIMA specs * 1 GARCH = 6
    REQUIRE(grid.candidateCount() == 6);

    auto candidates = grid.generate();
    REQUIRE(candidates.size() == 6);

    // Verify that all candidates satisfy p + q <= 2
    for (const auto& spec : candidates) {
        REQUIRE(spec.arimaSpec.p + spec.arimaSpec.q <= 2);
    }
}

// Test combined restrictions
TEST(candidate_grid_combined_restrictions) {
    CandidateGridConfig config(2, 2, 2, 1, 2);
    config.restrict_d_to_01 = true;
    config.restrict_pq_total = true;
    config.max_pq_total = 2;

    CandidateGrid grid(config);

    auto candidates = grid.generate();

    // Verify all candidates satisfy both restrictions
    for (const auto& spec : candidates) {
        REQUIRE(spec.arimaSpec.d <= 1);
        REQUIRE(spec.arimaSpec.p + spec.arimaSpec.q <= 2);
    }

    // With both restrictions:
    // ARIMA: p in {0,1,2}, d in {0,1}, q in {0,1,2}, p+q <= 2
    // Valid ARIMA combinations:
    //   d=0: (0,0,0), (0,0,1), (0,0,2), (1,0,0), (1,0,1), (2,0,0) = 6
    //   d=1: (0,1,0), (0,1,1), (0,1,2), (1,1,0), (1,1,1), (2,1,0) = 6
    // Total ARIMA: 12
    // GARCH: p_g=1, q_g in {1,2} -> 2 combinations
    // Total: 12 * 2 = 24
    REQUIRE(grid.candidateCount() == 24);
    REQUIRE(candidates.size() == 24);
}

// ============================================================================
// CandidateGrid Determinism Tests
// ============================================================================

// Test deterministic generation
TEST(candidate_grid_deterministic) {
    CandidateGridConfig config(2, 1, 2, 2, 2);

    CandidateGrid grid(config);

    auto candidates1 = grid.generate();
    auto candidates2 = grid.generate();

    REQUIRE(candidates1.size() == candidates2.size());

    // Verify that the candidates are generated in the same order
    for (size_t i = 0; i < candidates1.size(); ++i) {
        REQUIRE(candidates1[i].arimaSpec.p == candidates2[i].arimaSpec.p);
        REQUIRE(candidates1[i].arimaSpec.d == candidates2[i].arimaSpec.d);
        REQUIRE(candidates1[i].arimaSpec.q == candidates2[i].arimaSpec.q);
        REQUIRE(candidates1[i].garchSpec.p == candidates2[i].garchSpec.p);
        REQUIRE(candidates1[i].garchSpec.q == candidates2[i].garchSpec.q);
    }
}

// Test ordering: ARIMA combinations iterate with p outer, d middle, q inner
TEST(candidate_grid_ordering_arima) {
    CandidateGridConfig config(1, 1, 1, 1, 1);
    CandidateGrid grid(config);

    auto candidates = grid.generate();

    // Expected order (p, d, q):
    // (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    REQUIRE(candidates.size() == 8);

    REQUIRE(candidates[0].arimaSpec.p == 0);
    REQUIRE(candidates[0].arimaSpec.d == 0);
    REQUIRE(candidates[0].arimaSpec.q == 0);

    REQUIRE(candidates[1].arimaSpec.p == 0);
    REQUIRE(candidates[1].arimaSpec.d == 0);
    REQUIRE(candidates[1].arimaSpec.q == 1);

    REQUIRE(candidates[2].arimaSpec.p == 0);
    REQUIRE(candidates[2].arimaSpec.d == 1);
    REQUIRE(candidates[2].arimaSpec.q == 0);

    REQUIRE(candidates[3].arimaSpec.p == 0);
    REQUIRE(candidates[3].arimaSpec.d == 1);
    REQUIRE(candidates[3].arimaSpec.q == 1);

    REQUIRE(candidates[4].arimaSpec.p == 1);
    REQUIRE(candidates[4].arimaSpec.d == 0);
    REQUIRE(candidates[4].arimaSpec.q == 0);

    REQUIRE(candidates[5].arimaSpec.p == 1);
    REQUIRE(candidates[5].arimaSpec.d == 0);
    REQUIRE(candidates[5].arimaSpec.q == 1);

    REQUIRE(candidates[6].arimaSpec.p == 1);
    REQUIRE(candidates[6].arimaSpec.d == 1);
    REQUIRE(candidates[6].arimaSpec.q == 0);

    REQUIRE(candidates[7].arimaSpec.p == 1);
    REQUIRE(candidates[7].arimaSpec.d == 1);
    REQUIRE(candidates[7].arimaSpec.q == 1);
}

// Test ordering: GARCH combinations iterate with p_garch outer, q_garch inner
TEST(candidate_grid_ordering_garch) {
    CandidateGridConfig config(0, 0, 0, 2, 2);
    CandidateGrid grid(config);

    auto candidates = grid.generate();

    // Expected GARCH order (p_g, q_g):
    // (1,1), (1,2), (2,1), (2,2)
    REQUIRE(candidates.size() == 4);

    REQUIRE(candidates[0].garchSpec.p == 1);
    REQUIRE(candidates[0].garchSpec.q == 1);

    REQUIRE(candidates[1].garchSpec.p == 1);
    REQUIRE(candidates[1].garchSpec.q == 2);

    REQUIRE(candidates[2].garchSpec.p == 2);
    REQUIRE(candidates[2].garchSpec.q == 1);

    REQUIRE(candidates[3].garchSpec.p == 2);
    REQUIRE(candidates[3].garchSpec.q == 2);
}

int main() {
    report_test_results("Candidate Grid Selection");
    return get_test_result();
}
