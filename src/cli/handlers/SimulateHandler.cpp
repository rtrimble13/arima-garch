#include "ag/api/Engine.hpp"
#include "ag/cli/CliUtils.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"

#include <fstream>
#include <string>

#include "Handlers.hpp"
#include <fmt/core.h>

namespace ag::cli {

int handleSimulate(const std::string& arimaOrder, const std::string& garchOrder, int length,
                   unsigned int seed, const std::string& outputFile, bool use_student_t,
                   double student_t_df) {
    return executeWithErrorHandling([&]() {
        auto [p, d, q] = parseArimaOrder(arimaOrder);
        auto [P, Q] = parseGarchOrder(garchOrder);
        ag::models::ArimaGarchSpec spec(p, d, q, P, Q);

        ag::models::composite::ArimaGarchParameters params(spec);
        params.arima_params.intercept = 0.0;
        if (p > 0)
            params.arima_params.ar_coef[0] = 0.5;
        if (q > 0)
            params.arima_params.ma_coef[0] = 0.3;
        params.garch_params.omega = 0.01;
        if (P > 0)
            params.garch_params.alpha_coef[0] = 0.1;
        if (Q > 0)
            params.garch_params.beta_coef[0] = 0.85;

        const std::string dist_str = use_student_t
                                         ? fmt::format(" with Student-t(df={:.1f})", student_t_df)
                                         : " with Gaussian innovations";
        fmt::print("Simulating {} observations from ARIMA({},{},{})-GARCH({},{}) model{}...\n",
                   length, p, d, q, P, Q, dist_str);

        ag::api::Engine engine;
        auto sim_result = engine.simulate(spec, params, length, seed, use_student_t, student_t_df);
        if (!sim_result) {
            fmt::print("Error: {}\n", sim_result.error().message);
            return 1;
        }

        fmt::print("✅ Simulation completed\n");

        if (!outputFile.empty()) {
            std::ofstream file(outputFile);
            if (!file) {
                fmt::print("Warning: Failed to open output file {}\n", outputFile);
                return 0;
            }
            file << "observation,return,volatility\n";
            for (size_t i = 0; i < sim_result.value().returns.size(); ++i) {
                file << (i + 1) << "," << sim_result.value().returns[i] << ","
                     << sim_result.value().volatilities[i] << "\n";
            }
            fmt::print("Simulation saved to {}\n", outputFile);
        }

        return 0;
    });
}

}  // namespace ag::cli
