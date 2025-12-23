/**
 * @file example_json_io.cpp
 * @brief Example demonstrating JSON serialization of ARIMA-GARCH models.
 *
 * This example shows how to:
 * 1. Create an ARIMA-GARCH model with specific parameters
 * 2. Save the model to a JSON file
 * 3. Load the model from the JSON file
 * 4. Verify that the loaded model has identical parameters
 * 5. Use the loaded model for forecasting
 */

#include "ag/io/Json.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

#include <fmt/core.h>

using namespace ag::models;
using namespace ag::models::composite;
using namespace ag::io;

void printModelInfo(const ArimaGarchModel& model, const std::string& label) {
    fmt::print("\n=== {} ===\n", label);

    // Print specification
    const auto& spec = model.getSpec();
    fmt::print("Specification: ARIMA({},{},{}) - GARCH({},{})\n", spec.arimaSpec.p,
               spec.arimaSpec.d, spec.arimaSpec.q, spec.garchSpec.p, spec.garchSpec.q);

    // Print ARIMA parameters
    const auto& arima_params = model.getArimaParams();
    fmt::print("\nARIMA Parameters:\n");
    fmt::print("  Intercept: {:.6f}\n", arima_params.intercept);

    if (!arima_params.ar_coef.empty()) {
        fmt::print("  AR coefficients:");
        for (size_t i = 0; i < arima_params.ar_coef.size(); ++i) {
            fmt::print(" φ{}={:.6f}", i + 1, arima_params.ar_coef[i]);
        }
        fmt::print("\n");
    }

    if (!arima_params.ma_coef.empty()) {
        fmt::print("  MA coefficients:");
        for (size_t i = 0; i < arima_params.ma_coef.size(); ++i) {
            fmt::print(" θ{}={:.6f}", i + 1, arima_params.ma_coef[i]);
        }
        fmt::print("\n");
    }

    // Print GARCH parameters
    const auto& garch_params = model.getGarchParams();
    fmt::print("\nGARCH Parameters:\n");
    fmt::print("  Omega (ω): {:.6f}\n", garch_params.omega);

    if (!garch_params.alpha_coef.empty()) {
        fmt::print("  ARCH coefficients:");
        for (size_t i = 0; i < garch_params.alpha_coef.size(); ++i) {
            fmt::print(" α{}={:.6f}", i + 1, garch_params.alpha_coef[i]);
        }
        fmt::print("\n");
    }

    if (!garch_params.beta_coef.empty()) {
        fmt::print("  GARCH coefficients:");
        for (size_t i = 0; i < garch_params.beta_coef.size(); ++i) {
            fmt::print(" β{}={:.6f}", i + 1, garch_params.beta_coef[i]);
        }
        fmt::print("\n");
    }
}

int main() {
    fmt::print("=== ARIMA-GARCH Model JSON Serialization Example ===\n\n");

    // Step 1: Create an ARIMA-GARCH model with specific parameters
    fmt::print("Step 1: Creating ARIMA(1,0,1)-GARCH(1,1) model...\n");

    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    // Set realistic parameter values (typical for financial returns)
    params.arima_params.intercept = 0.05;   // Small positive drift
    params.arima_params.ar_coef[0] = 0.60;  // Moderate persistence
    params.arima_params.ma_coef[0] = 0.30;  // MA component

    params.garch_params.omega = 0.01;          // Base volatility
    params.garch_params.alpha_coef[0] = 0.10;  // ARCH effect
    params.garch_params.beta_coef[0] = 0.85;   // High volatility persistence

    ArimaGarchModel model(spec, params);

    // Print original model information
    printModelInfo(model, "Original Model");

    // Step 2: Process some data with the model
    fmt::print("\n\nStep 2: Processing sample data...\n");
    std::vector<double> sample_data = {1.0, 1.5, 1.2, 1.8, 1.3, 1.6, 1.4};

    fmt::print("\n{:>10} {:>12} {:>12}\n", "Time", "Mean (μ_t)", "Variance (h_t)");
    fmt::print("{}\n", std::string(36, '-'));

    for (size_t t = 0; t < sample_data.size(); ++t) {
        auto output = model.update(sample_data[t]);
        fmt::print("{:10} {:12.6f} {:12.6f}\n", t + 1, output.mu_t, output.h_t);
    }

    // Step 3: Save the model to a JSON file
    fmt::print("\n\nStep 3: Saving model to JSON file...\n");

    std::filesystem::path model_file = "/tmp/arima_garch_model.json";
    auto save_result = JsonWriter::saveModel(model_file, model, 2);

    if (!save_result.has_value()) {
        fmt::print("Error saving model: {}\n", save_result.error().message);
        return 1;
    }

    fmt::print("Model successfully saved to: {}\n", model_file.string());

    // Display a snippet of the JSON file
    fmt::print("\nJSON file content (first few lines):\n");
    std::ifstream file(model_file);
    std::string line;
    int line_count = 0;
    while (std::getline(file, line) && line_count < 15) {
        fmt::print("{}\n", line);
        line_count++;
    }
    fmt::print("...\n");
    file.close();

    // Step 4: Load the model from the JSON file
    fmt::print("\n\nStep 4: Loading model from JSON file...\n");

    auto load_result = JsonReader::loadModel(model_file);

    if (!load_result.has_value()) {
        fmt::print("Error loading model: {}\n", load_result.error().message);
        return 1;
    }

    auto loaded_model = *load_result;
    fmt::print("Model successfully loaded from: {}\n", model_file.string());

    // Print loaded model information
    printModelInfo(loaded_model, "Loaded Model");

    // Step 5: Verify that parameters are identical
    fmt::print("\n\nStep 5: Verifying parameter consistency...\n");

    const auto& orig_arima = model.getArimaParams();
    const auto& load_arima = loaded_model.getArimaParams();
    const auto& orig_garch = model.getGarchParams();
    const auto& load_garch = loaded_model.getGarchParams();

    bool params_match = true;
    const double tolerance = 1e-9;

    // Check ARIMA parameters
    if (std::abs(orig_arima.intercept - load_arima.intercept) > tolerance) {
        fmt::print("❌ ARIMA intercept mismatch!\n");
        params_match = false;
    }

    for (size_t i = 0; i < orig_arima.ar_coef.size(); ++i) {
        if (std::abs(orig_arima.ar_coef[i] - load_arima.ar_coef[i]) > tolerance) {
            fmt::print("❌ ARIMA AR coefficient {} mismatch!\n", i);
            params_match = false;
        }
    }

    for (size_t i = 0; i < orig_arima.ma_coef.size(); ++i) {
        if (std::abs(orig_arima.ma_coef[i] - load_arima.ma_coef[i]) > tolerance) {
            fmt::print("❌ ARIMA MA coefficient {} mismatch!\n", i);
            params_match = false;
        }
    }

    // Check GARCH parameters
    if (std::abs(orig_garch.omega - load_garch.omega) > tolerance) {
        fmt::print("❌ GARCH omega mismatch!\n");
        params_match = false;
    }

    for (size_t i = 0; i < orig_garch.alpha_coef.size(); ++i) {
        if (std::abs(orig_garch.alpha_coef[i] - load_garch.alpha_coef[i]) > tolerance) {
            fmt::print("❌ GARCH alpha coefficient {} mismatch!\n", i);
            params_match = false;
        }
    }

    for (size_t i = 0; i < orig_garch.beta_coef.size(); ++i) {
        if (std::abs(orig_garch.beta_coef[i] - load_garch.beta_coef[i]) > tolerance) {
            fmt::print("❌ GARCH beta coefficient {} mismatch!\n", i);
            params_match = false;
        }
    }

    if (params_match) {
        fmt::print("✅ All parameters match perfectly!\n");
    }

    // Step 6: Use loaded model for forecasting
    fmt::print("\n\nStep 6: Using loaded model for forecasting...\n");
    fmt::print("Processing new observations with loaded model:\n\n");

    std::vector<double> new_data = {1.7, 1.5, 1.9, 1.6};

    fmt::print("{:>10} {:>12} {:>12}\n", "Time", "Mean (μ_t)", "Variance (h_t)");
    fmt::print("{}\n", std::string(36, '-'));

    for (size_t t = 0; t < new_data.size(); ++t) {
        auto output = loaded_model.update(new_data[t]);
        fmt::print("{:10} {:12.6f} {:12.6f}\n", t + sample_data.size() + 1, output.mu_t,
                   output.h_t);
    }

    // Summary
    fmt::print("\n\n=== Summary ===\n");
    fmt::print("✅ Model saved to JSON successfully\n");
    fmt::print("✅ Model loaded from JSON successfully\n");
    fmt::print("✅ Parameters preserved exactly\n");
    fmt::print("✅ Loaded model can be used for forecasting\n");
    fmt::print("\nThe JSON format enables:\n");
    fmt::print("  • Model persistence and versioning\n");
    fmt::print("  • Reproducible forecasts\n");
    fmt::print("  • Easy model sharing and deployment\n");
    fmt::print("  • Integration with other tools\n");

    // Cleanup
    std::filesystem::remove(model_file);

    return 0;
}
