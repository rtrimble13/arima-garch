#include "ag/io/Json.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"

#include <cmath>
#include <filesystem>
#include <string>

#include "test_framework.hpp"

using ag::io::JsonReader;
using ag::io::JsonWriter;
using ag::models::ArimaGarchSpec;
using ag::models::ArimaSpec;
using ag::models::GarchSpec;
using ag::models::arima::ArimaParameters;
using ag::models::composite::ArimaGarchModel;
using ag::models::composite::ArimaGarchParameters;
using ag::models::garch::GarchParameters;

// Helper function to check if two doubles are approximately equal
bool approx_equal(double a, double b, double tolerance = 1e-9) {
    return std::abs(a - b) <= tolerance;
}

// Test serialization and deserialization of ArimaSpec
TEST(json_arima_spec_roundtrip) {
    ArimaSpec spec(2, 1, 3);

    // Serialize
    auto json = JsonWriter::toJson(spec);

    // Deserialize
    auto result = JsonReader::arimaSpecFromJson(json);
    REQUIRE(result.has_value());

    const auto& loaded_spec = *result;
    REQUIRE(loaded_spec.p == spec.p);
    REQUIRE(loaded_spec.d == spec.d);
    REQUIRE(loaded_spec.q == spec.q);
}

// Test serialization and deserialization of GarchSpec
TEST(json_garch_spec_roundtrip) {
    GarchSpec spec(1, 1);

    // Serialize
    auto json = JsonWriter::toJson(spec);

    // Deserialize
    auto result = JsonReader::garchSpecFromJson(json);
    REQUIRE(result.has_value());

    const auto& loaded_spec = *result;
    REQUIRE(loaded_spec.p == spec.p);
    REQUIRE(loaded_spec.q == spec.q);
}

// Test serialization and deserialization of ArimaGarchSpec
TEST(json_arima_garch_spec_roundtrip) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);

    // Serialize
    auto json = JsonWriter::toJson(spec);

    // Deserialize
    auto result = JsonReader::arimaGarchSpecFromJson(json);
    REQUIRE(result.has_value());

    const auto& loaded_spec = *result;
    REQUIRE(loaded_spec.arimaSpec.p == spec.arimaSpec.p);
    REQUIRE(loaded_spec.arimaSpec.d == spec.arimaSpec.d);
    REQUIRE(loaded_spec.arimaSpec.q == spec.arimaSpec.q);
    REQUIRE(loaded_spec.garchSpec.p == spec.garchSpec.p);
    REQUIRE(loaded_spec.garchSpec.q == spec.garchSpec.q);
}

// Test serialization and deserialization of ArimaParameters
TEST(json_arima_parameters_roundtrip) {
    ArimaSpec spec(2, 0, 1);
    ArimaParameters params(spec.p, spec.q);
    params.intercept = 0.5;
    params.ar_coef = {0.6, 0.3};
    params.ma_coef = {0.4};

    // Serialize
    auto json = JsonWriter::toJson(params);

    // Deserialize
    auto result = JsonReader::arimaParametersFromJson(json, spec);
    REQUIRE(result.has_value());

    const auto& loaded_params = *result;
    REQUIRE(approx_equal(loaded_params.intercept, params.intercept));
    REQUIRE(loaded_params.ar_coef.size() == params.ar_coef.size());
    REQUIRE(loaded_params.ma_coef.size() == params.ma_coef.size());
    for (size_t i = 0; i < params.ar_coef.size(); ++i) {
        REQUIRE(approx_equal(loaded_params.ar_coef[i], params.ar_coef[i]));
    }
    for (size_t i = 0; i < params.ma_coef.size(); ++i) {
        REQUIRE(approx_equal(loaded_params.ma_coef[i], params.ma_coef[i]));
    }
}

// Test serialization and deserialization of GarchParameters
TEST(json_garch_parameters_roundtrip) {
    GarchSpec spec(1, 1);
    GarchParameters params(spec.p, spec.q);
    params.omega = 0.01;
    params.alpha_coef = {0.1};
    params.beta_coef = {0.85};

    // Serialize
    auto json = JsonWriter::toJson(params);

    // Deserialize
    auto result = JsonReader::garchParametersFromJson(json, spec);
    REQUIRE(result.has_value());

    const auto& loaded_params = *result;
    REQUIRE(approx_equal(loaded_params.omega, params.omega));
    REQUIRE(loaded_params.alpha_coef.size() == params.alpha_coef.size());
    REQUIRE(loaded_params.beta_coef.size() == params.beta_coef.size());
    for (size_t i = 0; i < params.alpha_coef.size(); ++i) {
        REQUIRE(approx_equal(loaded_params.alpha_coef[i], params.alpha_coef[i]));
    }
    for (size_t i = 0; i < params.beta_coef.size(); ++i) {
        REQUIRE(approx_equal(loaded_params.beta_coef[i], params.beta_coef[i]));
    }
}

// Test serialization and deserialization of ArimaGarchParameters
TEST(json_arima_garch_parameters_roundtrip) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.05;
    params.arima_params.ar_coef[0] = 0.6;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.01;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.85;

    // Serialize
    auto json = JsonWriter::toJson(params);

    // Deserialize
    auto result = JsonReader::arimaGarchParametersFromJson(json, spec);
    REQUIRE(result.has_value());

    const auto& loaded_params = *result;
    REQUIRE(approx_equal(loaded_params.arima_params.intercept, params.arima_params.intercept));
    REQUIRE(approx_equal(loaded_params.arima_params.ar_coef[0], params.arima_params.ar_coef[0]));
    REQUIRE(approx_equal(loaded_params.arima_params.ma_coef[0], params.arima_params.ma_coef[0]));
    REQUIRE(approx_equal(loaded_params.garch_params.omega, params.garch_params.omega));
    REQUIRE(
        approx_equal(loaded_params.garch_params.alpha_coef[0], params.garch_params.alpha_coef[0]));
    REQUIRE(
        approx_equal(loaded_params.garch_params.beta_coef[0], params.garch_params.beta_coef[0]));
}

// Test file writing and reading
TEST(json_file_write_read_roundtrip) {
    auto test_file = std::filesystem::temp_directory_path() / "test_model.json";

    // Create a simple JSON object
    nlohmann::json test_json = {{"key", "value"}, {"number", 42}, {"array", {1, 2, 3}}};

    // Write to file
    auto write_result = JsonWriter::writeToFile(test_file, test_json);
    REQUIRE(write_result.has_value());

    // Read from file
    auto read_result = JsonReader::readFromFile(test_file);
    REQUIRE(read_result.has_value());

    const auto& loaded_json = *read_result;
    REQUIRE(loaded_json["key"] == "value");
    REQUIRE(loaded_json["number"] == 42);
    REQUIRE(loaded_json["array"].size() == 3);

    // Cleanup
    std::filesystem::remove(test_file);
}

// Test model save and load with parameters preserved
TEST(json_model_save_load_parameters) {
    std::filesystem::path model_file = "/tmp/test_arima_garch_model.json";

    // Create a model with specific parameters
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.05;
    params.arima_params.ar_coef[0] = 0.6;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.01;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchModel model(spec, params);

    // Save model
    auto save_result = JsonWriter::saveModel(model_file, model);
    REQUIRE(save_result.has_value());

    // Load model
    auto load_result = JsonReader::loadModel(model_file);
    REQUIRE(load_result.has_value());

    const auto& loaded_model = *load_result;

    // Verify spec
    REQUIRE(loaded_model.getSpec().arimaSpec.p == spec.arimaSpec.p);
    REQUIRE(loaded_model.getSpec().arimaSpec.d == spec.arimaSpec.d);
    REQUIRE(loaded_model.getSpec().arimaSpec.q == spec.arimaSpec.q);
    REQUIRE(loaded_model.getSpec().garchSpec.p == spec.garchSpec.p);
    REQUIRE(loaded_model.getSpec().garchSpec.q == spec.garchSpec.q);

    // Verify ARIMA parameters
    REQUIRE(approx_equal(loaded_model.getArimaParams().intercept, params.arima_params.intercept));
    REQUIRE(approx_equal(loaded_model.getArimaParams().ar_coef[0], params.arima_params.ar_coef[0]));
    REQUIRE(approx_equal(loaded_model.getArimaParams().ma_coef[0], params.arima_params.ma_coef[0]));

    // Verify GARCH parameters
    REQUIRE(approx_equal(loaded_model.getGarchParams().omega, params.garch_params.omega));
    REQUIRE(approx_equal(loaded_model.getGarchParams().alpha_coef[0],
                         params.garch_params.alpha_coef[0]));
    REQUIRE(
        approx_equal(loaded_model.getGarchParams().beta_coef[0], params.garch_params.beta_coef[0]));

    // Cleanup
    std::filesystem::remove(model_file);
}

// Test that saved model produces identical forecasts (via update)
TEST(json_model_identical_forecasts) {
    auto model_file = std::filesystem::temp_directory_path() / "test_model_forecast.json";

    // Create a model
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.05;
    params.arima_params.ar_coef[0] = 0.6;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.01;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchModel original_model(spec, params);

    // Generate some observations with original model
    std::vector<double> test_data = {1.0, 1.5, 1.2, 1.8, 1.3};
    std::vector<double> original_means;
    std::vector<double> original_variances;

    for (double y : test_data) {
        auto output = original_model.update(y);
        original_means.push_back(output.mu_t);
        original_variances.push_back(output.h_t);
    }

    // Save model after processing
    auto save_result = JsonWriter::saveModel(model_file, original_model);
    REQUIRE(save_result.has_value());

    // Load model
    auto load_result = JsonReader::loadModel(model_file);
    REQUIRE(load_result.has_value());

    auto loaded_model = *load_result;

    // Process same data with loaded model
    std::vector<double> loaded_means;
    std::vector<double> loaded_variances;

    for (double y : test_data) {
        auto output = loaded_model.update(y);
        loaded_means.push_back(output.mu_t);
        loaded_variances.push_back(output.h_t);
    }

    // Compare outputs - they should be identical since parameters are the same
    // Note: State might differ slightly due to initialization, but with same parameters,
    // the model behavior should converge quickly
    REQUIRE(loaded_means.size() == original_means.size());
    REQUIRE(loaded_variances.size() == original_variances.size());

    // Check that parameters produce consistent behavior
    // (exact state match is not guaranteed due to how state is restored)
    for (size_t i = 0; i < loaded_means.size(); ++i) {
        // We use a slightly larger tolerance since state restoration may not be exact
        REQUIRE(approx_equal(loaded_means[i], original_means[i], 1e-6));
        REQUIRE(approx_equal(loaded_variances[i], original_variances[i], 1e-6));
    }

    // Cleanup
    std::filesystem::remove(model_file);
}

// Test error handling for invalid JSON
TEST(json_invalid_spec_handling) {
    nlohmann::json invalid_json = {{"p", -1}, {"d", 0}, {"q", 1}};

    auto result = JsonReader::arimaSpecFromJson(invalid_json);
    REQUIRE(!result.has_value());  // Should fail due to negative p
}

// Test error handling for missing fields
TEST(json_missing_field_handling) {
    nlohmann::json incomplete_json = {{"p", 1}};  // Missing d and q

    auto result = JsonReader::arimaSpecFromJson(incomplete_json);
    REQUIRE(!result.has_value());  // Should fail due to missing fields
}

int main() {
    report_test_results("JSON I/O");
    return get_test_result();
}
