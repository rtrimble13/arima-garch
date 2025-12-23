#include "ag/selection/CandidateGrid.hpp"

#include <iostream>

#include <fmt/core.h>

using ag::selection::CandidateGrid;
using ag::selection::CandidateGridConfig;

int main() {
    std::cout << "CandidateGrid Example - Generating Model Specification Candidates\n";
    std::cout << "==================================================================\n\n";

    // Example 1: Basic grid without restrictions
    std::cout << "Example 1: Basic grid (max ARIMA orders: 2,1,2, max GARCH orders: 1,1)\n";
    {
        CandidateGridConfig config(2, 1, 2, 1, 1);
        CandidateGrid grid(config);

        fmt::print("Number of candidates: {}\n", grid.candidateCount());
        fmt::print("First 5 candidates:\n");

        auto candidates = grid.generate();
        for (size_t i = 0; i < std::min(size_t(5), candidates.size()); ++i) {
            const auto& spec = candidates[i];
            fmt::print("  ARIMA({},{},{})-GARCH({},{})\n", spec.arimaSpec.p, spec.arimaSpec.d,
                       spec.arimaSpec.q, spec.garchSpec.p, spec.garchSpec.q);
        }
        std::cout << "\n";
    }

    // Example 2: Grid with d restricted to {0, 1}
    std::cout << "Example 2: Restrict d to {0,1} (common in practice)\n";
    {
        CandidateGridConfig config(2, 2, 2, 1, 1);
        config.restrict_d_to_01 = true;

        CandidateGrid grid(config);
        fmt::print("Without restriction: {} candidates (if max_d=2)\n",
                   (2 + 1) * (2 + 1) * (2 + 1) * 1 * 1);
        fmt::print("With d in {{0,1}} restriction: {} candidates\n", grid.candidateCount());
        std::cout << "\n";
    }

    // Example 3: Grid with p+q total restriction
    std::cout << "Example 3: Restrict p+q <= 3 (limit model complexity)\n";
    {
        CandidateGridConfig config(3, 1, 3, 1, 1);
        config.restrict_pq_total = true;
        config.max_pq_total = 3;

        CandidateGrid grid(config);
        fmt::print("Number of candidates: {}\n", grid.candidateCount());

        auto candidates = grid.generate();
        fmt::print("Sample candidates (showing first 6):\n");
        for (size_t i = 0; i < std::min(size_t(6), candidates.size()); ++i) {
            const auto& spec = candidates[i];
            fmt::print("  ARIMA({},{},{})-GARCH({},{}) [p+q={}]\n", spec.arimaSpec.p,
                       spec.arimaSpec.d, spec.arimaSpec.q, spec.garchSpec.p, spec.garchSpec.q,
                       spec.arimaSpec.p + spec.arimaSpec.q);
        }
        std::cout << "\n";
    }

    // Example 4: Small grid with multiple GARCH orders
    std::cout << "Example 4: Multiple GARCH orders (exploring volatility model complexity)\n";
    {
        CandidateGridConfig config(1, 1, 1, 2, 2);
        CandidateGrid grid(config);

        fmt::print("Number of candidates: {}\n", grid.candidateCount());

        auto candidates = grid.generate();
        fmt::print("All candidates (8 ARIMA x 4 GARCH combinations):\n");
        for (size_t i = 0; i < candidates.size(); ++i) {
            const auto& spec = candidates[i];
            if (i % 4 == 0 && i > 0) {
                std::cout << "\n";
            }
            fmt::print("  ARIMA({},{},{})-GARCH({},{}) ", spec.arimaSpec.p, spec.arimaSpec.d,
                       spec.arimaSpec.q, spec.garchSpec.p, spec.garchSpec.q);
        }
        std::cout << "\n\n";
    }

    // Example 5: Combined restrictions
    std::cout << "Example 5: Combined restrictions (d in {0,1} and p+q <= 2)\n";
    {
        CandidateGridConfig config(2, 2, 2, 1, 2);
        config.restrict_d_to_01 = true;
        config.restrict_pq_total = true;
        config.max_pq_total = 2;

        CandidateGrid grid(config);
        fmt::print("Number of candidates: {}\n", grid.candidateCount());

        auto candidates = grid.generate();
        fmt::print("First 8 candidates:\n");
        for (size_t i = 0; i < std::min(size_t(8), candidates.size()); ++i) {
            const auto& spec = candidates[i];
            fmt::print("  ARIMA({},{},{})-GARCH({},{}) [d={}, p+q={}]\n", spec.arimaSpec.p,
                       spec.arimaSpec.d, spec.arimaSpec.q, spec.garchSpec.p, spec.garchSpec.q,
                       spec.arimaSpec.d, spec.arimaSpec.p + spec.arimaSpec.q);
        }
        std::cout << "\n";
    }

    std::cout << "Use Case: Model Selection\n";
    std::cout << "--------------------------\n";
    std::cout << "CandidateGrid is useful for:\n";
    std::cout << "  - Automatic model selection (fit all candidates, choose best AIC/BIC)\n";
    std::cout << "  - Grid search over model specifications\n";
    std::cout << "  - Systematic exploration of model space\n";
    std::cout << "  - Reproducible model selection workflows\n";

    return 0;
}
