#include "ag/api/Engine.hpp"
#include "ag/io/Json.hpp"

#include <cmath>
#include <fstream>
#include <string>

#include "Handlers.hpp"
#include <fmt/core.h>

namespace ag::cli {

int handleForecast(const std::string& modelFile, int horizon, const std::string& outputFile) {
    return executeWithErrorHandling([&]() {
        fmt::print("Loading model from {}...\n", modelFile);
        auto model_result = ag::io::JsonReader::loadModel(modelFile);
        if (!model_result) {
            fmt::print("Error: Failed to load model from {}\n", modelFile);
            return 1;
        }

        fmt::print("Generating {}-step ahead forecasts...\n", horizon);

        ag::api::Engine engine;
        auto forecast_result = engine.forecast(*model_result, horizon);
        if (!forecast_result) {
            fmt::print("Error: {}\n", forecast_result.error().message);
            return 1;
        }

        fmt::print("✅ Forecasts generated\n\n");
        fmt::print("Step  Mean Forecast  Std Dev\n");
        fmt::print("----  -------------  -------\n");

        for (int i = 0; i < horizon; ++i) {
            fmt::print("{:4d}  {:13.6f}  {:7.6f}\n", i + 1,
                       forecast_result.value().mean_forecasts[i],
                       std::sqrt(forecast_result.value().variance_forecasts[i]));
        }

        if (!outputFile.empty()) {
            std::ofstream file(outputFile);
            if (!file) {
                fmt::print("Warning: Failed to open output file {}\n", outputFile);
                return 0;
            }
            file << "step,mean,variance,std_dev\n";
            for (int i = 0; i < horizon; ++i) {
                file << (i + 1) << "," << forecast_result.value().mean_forecasts[i] << ","
                     << forecast_result.value().variance_forecasts[i] << ","
                     << std::sqrt(forecast_result.value().variance_forecasts[i]) << "\n";
            }
            fmt::print("\nForecasts saved to {}\n", outputFile);
        }

        return 0;
    });
}

}  // namespace ag::cli
