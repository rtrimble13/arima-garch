#include "ag/api/Engine.hpp"
#include "ag/cli/CliUtils.hpp"
#include "ag/io/Json.hpp"
#include "ag/report/FitSummary.hpp"
#include "ag/selection/CandidateGrid.hpp"
#include "ag/selection/ModelSelector.hpp"

#include <algorithm>
#include <string>

#include "Handlers.hpp"
#include <fmt/core.h>

namespace ag::cli {

int handleSelect(const std::string& dataFile, int maxP, int maxD, int maxQ, int maxGarchP,
                 int maxGarchQ, const std::string& criterion, const std::string& outputFile,
                 int topK, bool no_header) {
    return executeWithErrorHandling([&]() {
        fmt::print("Loading data from {}...\n", dataFile);
        auto data = loadData(dataFile, !no_header);
        fmt::print("Loaded {} observations\n", data.size());

        ag::selection::CandidateGridConfig config(maxP, maxD, maxQ, maxGarchP, maxGarchQ);
        ag::selection::CandidateGrid grid(config);
        auto candidates = grid.generate();

        fmt::print("Generated {} candidate models\n", candidates.size());
        fmt::print("Performing model selection using {}...\n", criterion);

        ag::selection::SelectionCriterion crit = ag::selection::SelectionCriterion::BIC;
        if (criterion == "AIC") {
            crit = ag::selection::SelectionCriterion::AIC;
        } else if (criterion == "AICc") {
            crit = ag::selection::SelectionCriterion::AICc;
        } else if (criterion == "CV") {
            crit = ag::selection::SelectionCriterion::CV;
        }

        ag::api::Engine engine;
        const bool build_ranking = (topK > 0);
        auto select_result = engine.auto_select(data, candidates, crit, build_ranking);
        if (!select_result) {
            fmt::print("Error: {}\n", select_result.error().message);
            return 1;
        }

        auto& result = select_result.value();
        const auto& spec = result.selected_spec;

        fmt::print("✅ Model selection completed\n");
        fmt::print("Best model: ARIMA({},{},{})-GARCH({},{})\n", spec.arimaSpec.p, spec.arimaSpec.d,
                   spec.arimaSpec.q, spec.garchSpec.p, spec.garchSpec.q);
        fmt::print("Candidates evaluated: {}\n", result.candidates_evaluated);
        fmt::print("Candidates failed: {}\n", result.candidates_failed);
        fmt::print("AIC: {:.4f}\n", result.summary.aic);
        fmt::print("BIC: {:.4f}\n", result.summary.bic);

        if (topK > 0 && !result.ranking.empty()) {
            const int display_count = std::min(topK, static_cast<int>(result.ranking.size()));
            fmt::print("\n=== Model Ranking (Top {}) ===\n", display_count);

            const int rank_width = 6;
            const int model_width = 20;
            const int score_width = 12;
            const int converged_width = 12;
            const int total_width = rank_width + model_width + score_width + converged_width;

            fmt::print("{:<6} {:<20} {:<12} {:<12}\n", "Rank", "Model", criterion, "Converged");
            fmt::print("{:-<{}}\n", "", total_width);

            int rank = 1;
            for (int i = 0; i < display_count; ++i) {
                const auto& entry = result.ranking[i];
                const std::string model_str =
                    fmt::format("ARIMA({},{},{})-GARCH({},{})", entry.p, entry.d, entry.q,
                                entry.garch_p, entry.garch_q);
                fmt::print("{:<6} {:<20} {:<12.4f} {:<12}\n", rank++, model_str, entry.score,
                           entry.converged ? "Yes" : "No");
            }
            fmt::print("\n");
        }

        if (!outputFile.empty()) {
            auto save_result = ag::io::JsonWriter::saveModel(outputFile, *result.model);
            if (save_result) {
                fmt::print("Model saved to {}\n", outputFile);
            } else {
                fmt::print("Warning: Failed to save model to {}\n", outputFile);
            }
        }

        fmt::print("\n{}\n", ag::report::generateTextReport(result.summary));
        return 0;
    });
}

}  // namespace ag::cli
