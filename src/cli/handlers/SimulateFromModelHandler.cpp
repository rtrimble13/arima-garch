#include "ag/io/Json.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"
#include "ag/stats/Descriptive.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "Handlers.hpp"
#include <fmt/core.h>

namespace ag::cli {

int handleSimulateFromModel(const std::string& modelFile, int numPaths, int length,
                            unsigned int seed, const std::string& outputFile, bool computeStats) {
    return executeWithErrorHandling([&]() {
        fmt::print("Loading model from {}...\n", modelFile);
        auto model_result = ag::io::JsonReader::loadModel(modelFile);
        if (!model_result) {
            fmt::print("Error: Failed to load model from {}\n", modelFile);
            return 1;
        }

        auto& model = *model_result;
        const auto& spec = model.getSpec();
        fmt::print("Model: ARIMA({},{},{})-GARCH({},{})\n", spec.arimaSpec.p, spec.arimaSpec.d,
                   spec.arimaSpec.q, spec.garchSpec.p, spec.garchSpec.q);

        ag::models::composite::ArimaGarchParameters params(spec);
        params.arima_params = model.getArimaParams();
        params.garch_params = model.getGarchParams();

        fmt::print("Simulating {} paths of {} observations each (seed={})...\n", numPaths, length,
                   seed);

        ag::simulation::ArimaGarchSimulator simulator(spec, params);

        std::vector<ag::simulation::SimulationResult> all_paths;
        all_paths.reserve(numPaths);
        for (int path = 0; path < numPaths; ++path) {
            // Hash-based seeding avoids overflow and yields a good
            // distribution while keeping each path reproducible from the
            // base seed.
            const unsigned int path_seed = seed ^ (std::hash<int>{}(path) + 0x9e3779b9);
            all_paths.push_back(simulator.simulate(length, path_seed));
        }

        fmt::print("✅ Simulation completed\n");

        if (!outputFile.empty()) {
            std::ofstream file(outputFile);
            if (!file) {
                fmt::print("Error: Failed to open output file {}\n", outputFile);
                return 1;
            }
            file << "path,observation,return,volatility\n";
            for (int path = 0; path < numPaths; ++path) {
                const auto& result = all_paths[path];
                for (std::size_t i = 0; i < result.returns.size(); ++i) {
                    file << (path + 1) << "," << (i + 1) << "," << result.returns[i] << ","
                         << result.volatilities[i] << "\n";
                }
            }
            fmt::print("Simulation results saved to {}\n", outputFile);
        }

        if (computeStats) {
            fmt::print("\n=== Summary Statistics Across All Paths ===\n");

            std::vector<double> all_returns;
            all_returns.reserve(static_cast<std::size_t>(numPaths) * length);
            for (const auto& result : all_paths) {
                all_returns.insert(all_returns.end(), result.returns.begin(), result.returns.end());
            }

            const double mean_ret = ag::stats::mean(all_returns);
            const double std_ret = std::sqrt(ag::stats::variance(all_returns));
            const double min_ret = *std::min_element(all_returns.begin(), all_returns.end());
            const double max_ret = *std::max_element(all_returns.begin(), all_returns.end());
            const double skew_ret = ag::stats::skewness(all_returns);
            const double kurt_ret = ag::stats::kurtosis(all_returns);

            fmt::print("Returns (aggregated over {} paths):\n", numPaths);
            fmt::print("  Mean:     {:.6f}\n", mean_ret);
            fmt::print("  Std Dev:  {:.6f}\n", std_ret);
            fmt::print("  Min:      {:.6f}\n", min_ret);
            fmt::print("  Max:      {:.6f}\n", max_ret);
            fmt::print("  Skewness: {:.6f}\n", skew_ret);
            fmt::print("  Kurtosis: {:.6f}\n", kurt_ret);

            fmt::print("\nFirst path statistics (for reproducibility check):\n");
            const auto& first_path = all_paths[0];
            fmt::print("  First 5 returns: ");
            for (int i = 0; i < std::min(5, static_cast<int>(first_path.returns.size())); ++i) {
                fmt::print("{:.6f} ", first_path.returns[i]);
            }
            fmt::print("\n");
        }

        return 0;
    });
}

}  // namespace ag::cli
