#include "ag/api/Engine.hpp"
#include "ag/cli/CliUtils.hpp"
#include "ag/io/Json.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/report/FitSummary.hpp"

#include <string>

#include "Handlers.hpp"
#include <fmt/core.h>

namespace ag::cli {

int handleFit(const std::string& dataFile, const std::string& arimaOrder,
              const std::string& garchOrder, const std::string& outputFile, bool no_header,
              bool use_student_t, double student_t_df) {
    return executeWithErrorHandling([&]() {
        fmt::print("Loading data from {}...\n", dataFile);
        auto data = loadData(dataFile, !no_header);
        fmt::print("Loaded {} observations\n", data.size());

        int p = 0, d = 0, q = 0;
        int P = 0, Q = 0;

        if (!arimaOrder.empty()) {
            auto arima_tuple = parseArimaOrder(arimaOrder);
            p = std::get<0>(arima_tuple);
            d = std::get<1>(arima_tuple);
            q = std::get<2>(arima_tuple);
        }
        if (!garchOrder.empty()) {
            auto garch_tuple = parseGarchOrder(garchOrder);
            P = std::get<0>(garch_tuple);
            Q = std::get<1>(garch_tuple);
        }

        if (arimaOrder.empty() && garchOrder.empty()) {
            fmt::print("Error: Must specify at least --arima or --garch parameters\n");
            return 1;
        }

        ag::models::ArimaGarchSpec spec(p, d, q, P, Q);

        const std::string dist_str = use_student_t
                                         ? fmt::format(" with Student-t(df={:.1f})", student_t_df)
                                         : " with Gaussian innovations";
        if (arimaOrder.empty()) {
            fmt::print(
                "Fitting GARCH({},{}) model (ARIMA component uses defaults: ARIMA(0,0,0)){}...\n",
                P, Q, dist_str);
        } else if (garchOrder.empty()) {
            fmt::print("Fitting ARIMA({},{},{}) model (no GARCH component){}...\n", p, d, q,
                       dist_str);
        } else {
            fmt::print("Fitting ARIMA({},{},{})-GARCH({},{}) model{}...\n", p, d, q, P, Q,
                       dist_str);
        }

        ag::api::Engine engine;
        auto fit_result = engine.fit(data, spec, true, use_student_t, student_t_df);
        if (!fit_result) {
            fmt::print("Error: {}\n", fit_result.error().message);
            return 1;
        }

        fmt::print("✅ Model fitted successfully\n");
        fmt::print("Converged: {}\n", fit_result.value().summary.converged);
        fmt::print("Iterations: {}\n", fit_result.value().summary.iterations);
        fmt::print("AIC: {:.4f}\n", fit_result.value().summary.aic);
        fmt::print("BIC: {:.4f}\n", fit_result.value().summary.bic);

        if (!outputFile.empty()) {
            auto save_result = ag::io::JsonWriter::saveModel(outputFile, *fit_result.value().model);
            if (save_result) {
                fmt::print("Model saved to {}\n", outputFile);
            } else {
                fmt::print("Warning: Failed to save model to {}\n", outputFile);
            }
        }

        fmt::print("\n{}\n", ag::report::generateTextReport(fit_result.value().summary));
        return 0;
    });
}

}  // namespace ag::cli
