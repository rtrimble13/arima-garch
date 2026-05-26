#include "ag/cli/CliUtils.hpp"
#include "ag/diagnostics/DiagnosticReport.hpp"
#include "ag/io/Json.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <string>

#include "Handlers.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace ag::cli {

int handleDiagnostics(const std::string& modelFile, const std::string& dataFile,
                      const std::string& outputFile, bool no_header) {
    return executeWithErrorHandling([&]() {
        fmt::print("Loading model from {}...\n", modelFile);
        auto model_result = ag::io::JsonReader::loadModel(modelFile);
        if (!model_result) {
            fmt::print("Error: Failed to load model from {}\n", modelFile);
            return 1;
        }

        fmt::print("Loading data from {}...\n", dataFile);
        auto data = loadData(dataFile, !no_header);
        fmt::print("Loaded {} observations\n", data.size());

        fmt::print("Running diagnostic tests...\n");

        auto& model = *model_result;
        const std::size_t ljung_box_lags = std::min(static_cast<std::size_t>(10), data.size() / 5);

        ag::models::composite::ArimaGarchParameters params(model.getSpec());
        params.arima_params = model.getArimaParams();
        params.garch_params = model.getGarchParams();

        auto diagnostics = ag::diagnostics::computeDiagnostics(model.getSpec(), params, data,
                                                               ljung_box_lags, true);

        fmt::print("✅ Diagnostics completed\n\n");
        fmt::print("=== Diagnostic Tests ===\n\n");

        fmt::print("Ljung-Box Test (raw residuals):\n");
        fmt::print("  Statistic: {:.4f}\n", diagnostics.ljung_box_residuals.statistic);
        fmt::print("  P-value: {:.4f}\n", diagnostics.ljung_box_residuals.p_value);
        fmt::print("  DOF: {}\n", diagnostics.ljung_box_residuals.dof);
        fmt::print("  Lags: {}\n\n", diagnostics.ljung_box_residuals.lags);

        fmt::print("Ljung-Box Test (squared residuals):\n");
        fmt::print("  Statistic: {:.4f}\n", diagnostics.ljung_box_squared.statistic);
        fmt::print("  P-value: {:.4f}\n", diagnostics.ljung_box_squared.p_value);
        fmt::print("  DOF: {}\n", diagnostics.ljung_box_squared.dof);
        fmt::print("  Lags: {}\n\n", diagnostics.ljung_box_squared.lags);

        fmt::print("Jarque-Bera Test:\n");
        fmt::print("  Statistic: {:.4f}\n", diagnostics.jarque_bera.statistic);
        fmt::print("  P-value: {:.4f}\n\n", diagnostics.jarque_bera.p_value);

        if (diagnostics.adf.has_value()) {
            fmt::print("Augmented Dickey-Fuller Test:\n");
            fmt::print("  Statistic: {:.4f}\n", diagnostics.adf->statistic);
            fmt::print("  P-value: {:.4f}\n", diagnostics.adf->p_value);
            fmt::print("  Lags: {}\n", diagnostics.adf->lags);
            fmt::print("  Critical values:\n");
            fmt::print("    1%:  {:.4f}\n", diagnostics.adf->critical_value_1pct);
            fmt::print("    5%:  {:.4f}\n", diagnostics.adf->critical_value_5pct);
            fmt::print("    10%: {:.4f}\n\n", diagnostics.adf->critical_value_10pct);
        }

        if (!outputFile.empty()) {
            nlohmann::json j;
            j["ljung_box_residuals"]["statistic"] = diagnostics.ljung_box_residuals.statistic;
            j["ljung_box_residuals"]["p_value"] = diagnostics.ljung_box_residuals.p_value;
            j["ljung_box_residuals"]["dof"] = diagnostics.ljung_box_residuals.dof;
            j["ljung_box_residuals"]["lags"] = diagnostics.ljung_box_residuals.lags;

            j["ljung_box_squared"]["statistic"] = diagnostics.ljung_box_squared.statistic;
            j["ljung_box_squared"]["p_value"] = diagnostics.ljung_box_squared.p_value;
            j["ljung_box_squared"]["dof"] = diagnostics.ljung_box_squared.dof;
            j["ljung_box_squared"]["lags"] = diagnostics.ljung_box_squared.lags;

            j["jarque_bera"]["statistic"] = diagnostics.jarque_bera.statistic;
            j["jarque_bera"]["p_value"] = diagnostics.jarque_bera.p_value;

            if (diagnostics.adf.has_value()) {
                j["adf"]["statistic"] = diagnostics.adf->statistic;
                j["adf"]["p_value"] = diagnostics.adf->p_value;
                j["adf"]["lags"] = diagnostics.adf->lags;
                j["adf"]["critical_value_1pct"] = diagnostics.adf->critical_value_1pct;
                j["adf"]["critical_value_5pct"] = diagnostics.adf->critical_value_5pct;
                j["adf"]["critical_value_10pct"] = diagnostics.adf->critical_value_10pct;
            }

            std::ofstream file(outputFile);
            if (file) {
                file << j.dump(2);
                fmt::print("Diagnostics saved to {}\n", outputFile);
            } else {
                fmt::print("Warning: Failed to save diagnostics to {}\n", outputFile);
            }
        }

        return 0;
    });
}

}  // namespace ag::cli
