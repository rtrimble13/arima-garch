/**
 * @file cli/main.cpp
 * @brief Command-line interface for ARIMA-GARCH modeling.
 *
 * Provides subcommands for:
 * - fit: Fit ARIMA-GARCH model to data
 * - select: Automatic model selection
 * - forecast: Generate forecasts
 * - simulate: Simulate synthetic data
 * - diagnostics: Run diagnostic tests
 */

#include "ag/api/Engine.hpp"
#include "ag/io/CsvReader.hpp"
#include "ag/io/Json.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/report/FitSummary.hpp"
#include "ag/selection/CandidateGrid.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

using ag::api::Engine;
using ag::models::ArimaGarchSpec;

// Helper function to parse ARIMA order string (e.g., "1,1,1" -> p=1, d=1, q=1)
std::tuple<int, int, int> parseArimaOrder(const std::string& order) {
    std::istringstream iss(order);
    int p, d, q;
    char comma1, comma2;
    if (!(iss >> p >> comma1 >> d >> comma2 >> q) || comma1 != ',' || comma2 != ',') {
        throw std::invalid_argument("Invalid ARIMA order format. Use p,d,q (e.g., 1,1,1)");
    }
    return {p, d, q};
}

// Helper function to parse GARCH order string (e.g., "1,1" -> p=1, q=1)
std::tuple<int, int> parseGarchOrder(const std::string& order) {
    std::istringstream iss(order);
    int p, q;
    char comma;
    if (!(iss >> p >> comma >> q) || comma != ',') {
        throw std::invalid_argument("Invalid GARCH order format. Use p,q (e.g., 1,1)");
    }
    return {p, q};
}

// Load data from CSV file
std::vector<double> loadData(const std::string& filepath) {
    ag::io::CsvReaderOptions options;
    options.has_header = true;
    options.value_column = 0;

    auto result = ag::io::CsvReader::read(filepath, options);
    if (!result) {
        throw std::runtime_error("Failed to read data from file: " + filepath);
    }

    // Convert TimeSeries to vector
    std::vector<double> data;
    const auto& ts = *result;
    data.reserve(ts.size());
    for (std::size_t i = 0; i < ts.size(); ++i) {
        data.push_back(ts[i]);
    }
    return data;
}

// Fit subcommand handler
int handleFit(const std::string& dataFile, const std::string& arimaOrder,
              const std::string& garchOrder, const std::string& outputFile) {
    try {
        fmt::print("Loading data from {}...\n", dataFile);
        auto data = loadData(dataFile);
        fmt::print("Loaded {} observations\n", data.size());

        // Parse model specification
        auto [p, d, q] = parseArimaOrder(arimaOrder);
        auto [P, Q] = parseGarchOrder(garchOrder);
        ArimaGarchSpec spec(p, d, q, P, Q);

        fmt::print("Fitting ARIMA({},{},{})-GARCH({},{}) model...\n", p, d, q, P, Q);

        Engine engine;
        auto fit_result = engine.fit(data, spec, true);

        if (!fit_result) {
            fmt::print("Error: {}\n", fit_result.error().message);
            return 1;
        }

        fmt::print("✅ Model fitted successfully\n");
        fmt::print("Converged: {}\n", fit_result.value().summary.converged);
        fmt::print("Iterations: {}\n", fit_result.value().summary.iterations);
        fmt::print("AIC: {:.4f}\n", fit_result.value().summary.aic);
        fmt::print("BIC: {:.4f}\n", fit_result.value().summary.bic);

        // Save model to file
        if (!outputFile.empty()) {
            auto save_result = ag::io::JsonWriter::saveModel(outputFile, *fit_result.value().model);
            if (save_result) {
                fmt::print("Model saved to {}\n", outputFile);
            } else {
                fmt::print("Warning: Failed to save model to {}\n", outputFile);
            }
        }

        // Print fit summary report
        fmt::print("\n{}\n", ag::report::generateTextReport(fit_result.value().summary));

        return 0;
    } catch (const std::exception& e) {
        fmt::print("Error: {}\n", e.what());
        return 1;
    }
}

// Select subcommand handler
int handleSelect(const std::string& dataFile, int maxP, int maxD, int maxQ, int maxGarchP,
                 int maxGarchQ, const std::string& criterion, const std::string& outputFile) {
    try {
        fmt::print("Loading data from {}...\n", dataFile);
        auto data = loadData(dataFile);
        fmt::print("Loaded {} observations\n", data.size());

        // Generate candidate grid
        ag::selection::CandidateGridConfig config(maxP, maxD, maxQ, maxGarchP, maxGarchQ);
        ag::selection::CandidateGrid grid(config);
        auto candidates = grid.generate();

        fmt::print("Generated {} candidate models\n", candidates.size());
        fmt::print("Performing model selection using {}...\n", criterion);

        // Parse selection criterion
        ag::selection::SelectionCriterion crit = ag::selection::SelectionCriterion::BIC;
        if (criterion == "AIC") {
            crit = ag::selection::SelectionCriterion::AIC;
        } else if (criterion == "AICc") {
            crit = ag::selection::SelectionCriterion::AICc;
        } else if (criterion == "CV") {
            crit = ag::selection::SelectionCriterion::CV;
        }

        Engine engine;
        auto select_result = engine.auto_select(data, candidates, crit);

        if (!select_result) {
            fmt::print("Error: {}\n", select_result.error().message);
            return 1;
        }

        auto& result = select_result.value();
        auto& spec = result.selected_spec;

        fmt::print("✅ Model selection completed\n");
        fmt::print("Best model: ARIMA({},{},{})-GARCH({},{})\n", spec.arimaSpec.p, spec.arimaSpec.d,
                   spec.arimaSpec.q, spec.garchSpec.p, spec.garchSpec.q);
        fmt::print("Candidates evaluated: {}\n", result.candidates_evaluated);
        fmt::print("Candidates failed: {}\n", result.candidates_failed);
        fmt::print("AIC: {:.4f}\n", result.summary.aic);
        fmt::print("BIC: {:.4f}\n", result.summary.bic);

        // Save model to file
        if (!outputFile.empty()) {
            auto save_result = ag::io::JsonWriter::saveModel(outputFile, *result.model);
            if (save_result) {
                fmt::print("Model saved to {}\n", outputFile);
            } else {
                fmt::print("Warning: Failed to save model to {}\n", outputFile);
            }
        }

        // Print fit summary
        fmt::print("\n{}\n", ag::report::generateTextReport(result.summary));

        return 0;
    } catch (const std::exception& e) {
        fmt::print("Error: {}\n", e.what());
        return 1;
    }
}

// Forecast subcommand handler
int handleForecast(const std::string& modelFile, int horizon, const std::string& outputFile) {
    try {
        fmt::print("Loading model from {}...\n", modelFile);
        auto model_result = ag::io::JsonReader::loadModel(modelFile);
        if (!model_result) {
            fmt::print("Error: Failed to load model from {}\n", modelFile);
            return 1;
        }

        fmt::print("Generating {}-step ahead forecasts...\n", horizon);

        Engine engine;
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

        // Save forecasts to file
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
    } catch (const std::exception& e) {
        fmt::print("Error: {}\n", e.what());
        return 1;
    }
}

// Simulate subcommand handler
int handleSimulate(const std::string& arimaOrder, const std::string& garchOrder, int length,
                   unsigned int seed, const std::string& outputFile) {
    try {
        // Parse model specification
        auto [p, d, q] = parseArimaOrder(arimaOrder);
        auto [P, Q] = parseGarchOrder(garchOrder);
        ArimaGarchSpec spec(p, d, q, P, Q);

        // Use default parameters for simulation
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

        fmt::print("Simulating {} observations from ARIMA({},{},{})-GARCH({},{}) model...\n",
                   length, p, d, q, P, Q);

        Engine engine;
        auto sim_result = engine.simulate(spec, params, length, seed);

        if (!sim_result) {
            fmt::print("Error: {}\n", sim_result.error().message);
            return 1;
        }

        fmt::print("✅ Simulation completed\n");

        // Save simulation to file
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
    } catch (const std::exception& e) {
        fmt::print("Error: {}\n", e.what());
        return 1;
    }
}

// Diagnostics subcommand handler
int handleDiagnostics(const std::string& modelFile, const std::string& dataFile,
                      const std::string& outputFile) {
    try {
        fmt::print("Loading model from {}...\n", modelFile);
        auto model_result = ag::io::JsonReader::loadModel(modelFile);
        if (!model_result) {
            fmt::print("Error: Failed to load model from {}\n", modelFile);
            return 1;
        }

        fmt::print("Loading data from {}...\n", dataFile);
        auto data = loadData(dataFile);
        fmt::print("Loaded {} observations\n", data.size());

        // Run diagnostics
        fmt::print("Running diagnostic tests...\n");

        auto& model = *model_result;
        std::size_t ljung_box_lags = std::min(static_cast<std::size_t>(10), data.size() / 5);

        // Reconstruct parameters from the model
        ag::models::composite::ArimaGarchParameters params(model.getSpec());
        params.arima_params = model.getArimaParams();
        params.garch_params = model.getGarchParams();

        auto diagnostics = ag::diagnostics::computeDiagnostics(model.getSpec(), params, data,
                                                               ljung_box_lags, true);

        fmt::print("✅ Diagnostics completed\n\n");

        // Print diagnostics report
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

        // Save diagnostics to JSON file if requested
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

            std::ofstream file(outputFile);
            if (file) {
                file << j.dump(2);
                fmt::print("Diagnostics saved to {}\n", outputFile);
            } else {
                fmt::print("Warning: Failed to save diagnostics to {}\n", outputFile);
            }
        }

        return 0;
    } catch (const std::exception& e) {
        fmt::print("Error: {}\n", e.what());
        return 1;
    }
}

int main(int argc, char* argv[]) {
    CLI::App app{"ARIMA-GARCH Time Series Modeling CLI", "ag"};
    app.require_subcommand(0, 1);  // Allow 0 or 1 subcommand (0 for --help)
    app.set_version_flag("--version,-v", "0.1.0");

    // Fit subcommand
    auto* fit = app.add_subcommand("fit", "Fit ARIMA-GARCH model to time series data");
    std::string fit_data_file;
    std::string fit_arima_order;
    std::string fit_garch_order;
    std::string fit_output_file;

    fit->add_option("-d,--data,-i,--input", fit_data_file,
                    "Input data file (CSV format, first column)")
        ->required();
    fit->add_option("-a,--arima", fit_arima_order, "ARIMA order as p,d,q (e.g., 1,1,1)")
        ->required();
    fit->add_option("-g,--garch", fit_garch_order, "GARCH order as p,q (e.g., 1,1)")->required();
    fit->add_option("-o,--output,--out", fit_output_file, "Output model file (JSON format)");

    fit->callback([&]() {
        return handleFit(fit_data_file, fit_arima_order, fit_garch_order, fit_output_file);
    });

    // Select subcommand
    auto* select = app.add_subcommand("select", "Automatic model selection from candidate grid");
    std::string select_data_file;
    int select_max_p = 2;
    int select_max_d = 1;
    int select_max_q = 2;
    int select_max_garch_p = 1;
    int select_max_garch_q = 1;
    std::string select_criterion = "BIC";
    std::string select_output_file;

    select
        ->add_option("-d,--data,-i,--input", select_data_file,
                     "Input data file (CSV format, first column)")
        ->required();
    select->add_option("--max-p", select_max_p, "Maximum ARIMA AR order (default: 2)");
    select->add_option("--max-d", select_max_d, "Maximum ARIMA differencing order (default: 1)");
    select->add_option("--max-q", select_max_q, "Maximum ARIMA MA order (default: 2)");
    select->add_option("--max-garch-p", select_max_garch_p, "Maximum GARCH p order (default: 1)");
    select->add_option("--max-garch-q", select_max_garch_q, "Maximum GARCH q order (default: 1)");
    select->add_option("-c,--criterion", select_criterion,
                       "Selection criterion: BIC, AIC, AICc, or CV (default: BIC)");
    select->add_option("-o,--output,--out", select_output_file, "Output model file (JSON format)");

    select->callback([&]() {
        return handleSelect(select_data_file, select_max_p, select_max_d, select_max_q,
                            select_max_garch_p, select_max_garch_q, select_criterion,
                            select_output_file);
    });

    // Forecast subcommand
    auto* forecast = app.add_subcommand("forecast", "Generate forecasts from fitted model");
    std::string forecast_model_file;
    int forecast_horizon = 10;
    std::string forecast_output_file;

    forecast->add_option("-m,--model", forecast_model_file, "Input model file (JSON format)")
        ->required();
    forecast->add_option("-n,--horizon", forecast_horizon,
                         "Forecast horizon (number of steps ahead, default: 10)");
    forecast->add_option("-o,--output,--out", forecast_output_file,
                         "Output forecast file (CSV format)");

    forecast->callback([&]() {
        return handleForecast(forecast_model_file, forecast_horizon, forecast_output_file);
    });

    // Simulate subcommand
    auto* simulate = app.add_subcommand("sim", "Simulate synthetic time series data");
    std::string sim_arima_order;
    std::string sim_garch_order;
    int sim_length = 1000;
    unsigned int sim_seed = 42;
    std::string sim_output_file;

    simulate->add_option("-a,--arima", sim_arima_order, "ARIMA order as p,d,q (e.g., 1,1,1)")
        ->required();
    simulate->add_option("-g,--garch", sim_garch_order, "GARCH order as p,q (e.g., 1,1)")
        ->required();
    simulate->add_option("-n,--length", sim_length,
                         "Number of observations to simulate (default: 1000)");
    simulate->add_option("-s,--seed", sim_seed, "Random seed (default: 42)");
    simulate->add_option("-o,--output,--out", sim_output_file, "Output data file (CSV format)")
        ->required();

    simulate->callback([&]() {
        return handleSimulate(sim_arima_order, sim_garch_order, sim_length, sim_seed,
                              sim_output_file);
    });

    // Diagnostics subcommand
    auto* diagnostics = app.add_subcommand("diagnostics", "Run diagnostic tests on fitted model");
    std::string diag_model_file;
    std::string diag_data_file;
    std::string diag_output_file;

    diagnostics->add_option("-m,--model", diag_model_file, "Input model file (JSON format)")
        ->required();
    diagnostics->add_option("-d,--data,-i,--input", diag_data_file, "Input data file (CSV format)")
        ->required();
    diagnostics->add_option("-o,--output,--out", diag_output_file,
                            "Output diagnostics file (JSON format)");

    diagnostics->callback(
        [&]() { return handleDiagnostics(diag_model_file, diag_data_file, diag_output_file); });

    // Parse command line
    CLI11_PARSE(app, argc, argv);

    return 0;
}
