#pragma once

#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/arima/ArimaModel.hpp"
#include "ag/models/arima/ArimaState.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/models/garch/GarchModel.hpp"
#include "ag/models/garch/GarchState.hpp"
#include "ag/util/Expected.hpp"

#include <filesystem>
#include <string>

#include <nlohmann/json.hpp>

namespace ag::io {

/**
 * @brief Error type for JSON operations.
 */
struct JsonError {
    std::string message;
};

/**
 * @brief Empty type to represent successful operations that don't return a value.
 */
struct Success {};

/**
 * @brief Metadata for a serialized model.
 */
struct ModelMetadata {
    std::string timestamp;   // ISO 8601 timestamp of when model was saved
    std::string version;     // Library version used to save the model
    std::string model_type;  // Type of model (e.g., "ArimaGarch")

    /**
     * @brief Construct metadata with current timestamp and library version.
     */
    ModelMetadata();

    /**
     * @brief Construct metadata with specific values.
     */
    ModelMetadata(const std::string& ts, const std::string& ver, const std::string& type);
};

/**
 * @brief JSON serialization utilities for ARIMA-GARCH models.
 *
 * Provides functions to serialize and deserialize model specifications,
 * parameters, states, and complete fitted models to/from JSON format.
 *
 * This enables:
 * - Saving fitted models to disk for later use
 * - Loading models to produce forecasts without refitting
 * - Model versioning and reproducibility
 * - Configuration management
 */
class JsonWriter {
public:
    /**
     * @brief Serialize an ARIMA specification to JSON.
     */
    static nlohmann::json toJson(const models::ArimaSpec& spec);

    /**
     * @brief Serialize a GARCH specification to JSON.
     */
    static nlohmann::json toJson(const models::GarchSpec& spec);

    /**
     * @brief Serialize an ARIMA-GARCH specification to JSON.
     */
    static nlohmann::json toJson(const models::ArimaGarchSpec& spec);

    /**
     * @brief Serialize ARIMA parameters to JSON.
     */
    static nlohmann::json toJson(const models::arima::ArimaParameters& params);

    /**
     * @brief Serialize GARCH parameters to JSON.
     */
    static nlohmann::json toJson(const models::garch::GarchParameters& params);

    /**
     * @brief Serialize ARIMA-GARCH parameters to JSON.
     */
    static nlohmann::json toJson(const models::composite::ArimaGarchParameters& params);

    /**
     * @brief Serialize ARIMA state (histories) to JSON.
     */
    static nlohmann::json toJson(const models::arima::ArimaState& state);

    /**
     * @brief Serialize GARCH state (histories) to JSON.
     */
    static nlohmann::json toJson(const models::garch::GarchState& state);

    /**
     * @brief Serialize model metadata to JSON.
     */
    static nlohmann::json toJson(const ModelMetadata& metadata);

    /**
     * @brief Serialize a complete ARIMA-GARCH model to JSON.
     *
     * Includes specification, parameters, state, and metadata.
     */
    static nlohmann::json toJson(const models::composite::ArimaGarchModel& model);

    /**
     * @brief Write a JSON object to a file.
     *
     * @param filepath Path to the output JSON file
     * @param json JSON object to write
     * @param indent Number of spaces for indentation (default: 2, -1 for compact)
     * @return Expected containing Success on success, or JsonError on failure
     */
    static expected<Success, JsonError> writeToFile(const std::filesystem::path& filepath,
                                                    const nlohmann::json& json, int indent = 2);

    /**
     * @brief Save an ARIMA-GARCH model to a JSON file.
     *
     * @param filepath Path to the output JSON file
     * @param model The model to save
     * @param indent Number of spaces for indentation (default: 2)
     * @return Expected containing Success on success, or JsonError on failure
     */
    static expected<Success, JsonError> saveModel(const std::filesystem::path& filepath,
                                                  const models::composite::ArimaGarchModel& model,
                                                  int indent = 2);
};

/**
 * @brief JSON deserialization utilities for ARIMA-GARCH models.
 */
class JsonReader {
public:
    /**
     * @brief Deserialize an ARIMA specification from JSON.
     */
    static expected<models::ArimaSpec, JsonError> arimaSpecFromJson(const nlohmann::json& json);

    /**
     * @brief Deserialize a GARCH specification from JSON.
     */
    static expected<models::GarchSpec, JsonError> garchSpecFromJson(const nlohmann::json& json);

    /**
     * @brief Deserialize an ARIMA-GARCH specification from JSON.
     */
    static expected<models::ArimaGarchSpec, JsonError>
    arimaGarchSpecFromJson(const nlohmann::json& json);

    /**
     * @brief Deserialize ARIMA parameters from JSON.
     */
    static expected<models::arima::ArimaParameters, JsonError>
    arimaParametersFromJson(const nlohmann::json& json, const models::ArimaSpec& spec);

    /**
     * @brief Deserialize GARCH parameters from JSON.
     */
    static expected<models::garch::GarchParameters, JsonError>
    garchParametersFromJson(const nlohmann::json& json, const models::GarchSpec& spec);

    /**
     * @brief Deserialize ARIMA-GARCH parameters from JSON.
     */
    static expected<models::composite::ArimaGarchParameters, JsonError>
    arimaGarchParametersFromJson(const nlohmann::json& json, const models::ArimaGarchSpec& spec);

    /**
     * @brief Deserialize ARIMA state from JSON.
     */
    static expected<models::arima::ArimaState, JsonError>
    arimaStateFromJson(const nlohmann::json& json, const models::ArimaSpec& spec);

    /**
     * @brief Deserialize GARCH state from JSON.
     */
    static expected<models::garch::GarchState, JsonError>
    garchStateFromJson(const nlohmann::json& json, const models::GarchSpec& spec);

    /**
     * @brief Deserialize model metadata from JSON.
     */
    static expected<ModelMetadata, JsonError> metadataFromJson(const nlohmann::json& json);

    /**
     * @brief Read a JSON object from a file.
     *
     * @param filepath Path to the input JSON file
     * @return Expected containing JSON object on success, or JsonError on failure
     */
    static expected<nlohmann::json, JsonError> readFromFile(const std::filesystem::path& filepath);

    /**
     * @brief Load an ARIMA-GARCH model from a JSON file.
     *
     * @param filepath Path to the input JSON file
     * @return Expected containing the loaded model on success, or JsonError on failure
     */
    static expected<models::composite::ArimaGarchModel, JsonError>
    loadModel(const std::filesystem::path& filepath);
};

}  // namespace ag::io
