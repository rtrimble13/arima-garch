#include "ag/io/Json.hpp"

#include "ag/Version.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace ag::io {

// Helper function to get current ISO 8601 timestamp
static std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    // Use thread-safe time conversion
    std::tm tm_buf;
#ifdef _WIN32
    gmtime_s(&tm_buf, &time_t);
    std::tm* tm_ptr = &tm_buf;
#else
    std::tm* tm_ptr = gmtime_r(&time_t, &tm_buf);
#endif

    std::stringstream ss;
    ss << std::put_time(tm_ptr, "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

// ModelMetadata implementation
ModelMetadata::ModelMetadata()
    : timestamp(getCurrentTimestamp()), version(ag::kVersion), model_type("ArimaGarch") {}

ModelMetadata::ModelMetadata(const std::string& ts, const std::string& ver, const std::string& type)
    : timestamp(ts), version(ver), model_type(type) {}

// JsonWriter implementations

nlohmann::json JsonWriter::toJson(const models::ArimaSpec& spec) {
    return nlohmann::json{{"p", spec.p}, {"d", spec.d}, {"q", spec.q}};
}

nlohmann::json JsonWriter::toJson(const models::GarchSpec& spec) {
    return nlohmann::json{{"p", spec.p}, {"q", spec.q}};
}

nlohmann::json JsonWriter::toJson(const models::ArimaGarchSpec& spec) {
    return nlohmann::json{{"arima", toJson(spec.arimaSpec)}, {"garch", toJson(spec.garchSpec)}};
}

nlohmann::json JsonWriter::toJson(const models::arima::ArimaParameters& params) {
    return nlohmann::json{
        {"intercept", params.intercept}, {"ar_coef", params.ar_coef}, {"ma_coef", params.ma_coef}};
}

nlohmann::json JsonWriter::toJson(const models::garch::GarchParameters& params) {
    return nlohmann::json{{"omega", params.omega},
                          {"alpha_coef", params.alpha_coef},
                          {"beta_coef", params.beta_coef}};
}

nlohmann::json JsonWriter::toJson(const models::composite::ArimaGarchParameters& params) {
    return nlohmann::json{{"arima", toJson(params.arima_params)},
                          {"garch", toJson(params.garch_params)}};
}

nlohmann::json JsonWriter::toJson(const models::arima::ArimaState& state) {
    return nlohmann::json{{"observation_history", state.getObservationHistory()},
                          {"residual_history", state.getResidualHistory()},
                          {"differenced_series", state.getDifferencedSeries()},
                          {"initialized", state.isInitialized()}};
}

nlohmann::json JsonWriter::toJson(const models::garch::GarchState& state) {
    return nlohmann::json{{"variance_history", state.getVarianceHistory()},
                          {"squared_residual_history", state.getSquaredResidualHistory()},
                          {"initial_variance", state.getInitialVariance()},
                          {"initialized", state.isInitialized()}};
}

nlohmann::json JsonWriter::toJson(const ModelMetadata& metadata) {
    return nlohmann::json{{"timestamp", metadata.timestamp},
                          {"version", metadata.version},
                          {"model_type", metadata.model_type}};
}

nlohmann::json JsonWriter::toJson(const models::composite::ArimaGarchModel& model) {
    nlohmann::json j;
    j["metadata"] = toJson(ModelMetadata());
    j["spec"] = toJson(model.getSpec());
    j["parameters"] = nlohmann::json{{"arima", toJson(model.getArimaParams())},
                                     {"garch", toJson(model.getGarchParams())}};
    j["state"] = nlohmann::json{{"arima", toJson(model.getArimaState())},
                                {"garch", toJson(model.getGarchState())}};

    // Differencing anchors (d > 0). Needed so a reloaded model integrates
    // forecasts from the saved terminal levels rather than from zero.
    const auto& differencer = model.getDifferencer();
    j["state"]["differencer"] =
        nlohmann::json{{"order", differencer.order()},
                       {"anchors", differencer.anchors()},
                       {"observation_count", differencer.observationCount()}};
    return j;
}

expected<Success, JsonError> JsonWriter::writeToFile(const std::filesystem::path& filepath,
                                                     const nlohmann::json& json, int indent) {
    try {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            return unexpected<JsonError>({"Failed to open file for writing: " + filepath.string()});
        }

        file << json.dump(indent);
        file.close();

        if (file.fail()) {
            return unexpected<JsonError>({"Failed to write to file: " + filepath.string()});
        }

        return Success{};
    } catch (const std::exception& e) {
        return unexpected<JsonError>({"Exception while writing JSON: " + std::string(e.what())});
    }
}

expected<Success, JsonError> JsonWriter::saveModel(const std::filesystem::path& filepath,
                                                   const models::composite::ArimaGarchModel& model,
                                                   int indent) {
    try {
        nlohmann::json j = toJson(model);
        return writeToFile(filepath, j, indent);
    } catch (const std::exception& e) {
        return unexpected<JsonError>({"Exception while saving model: " + std::string(e.what())});
    }
}

// JsonReader implementations

expected<models::ArimaSpec, JsonError> JsonReader::arimaSpecFromJson(const nlohmann::json& json) {
    try {
        int p = json.at("p").get<int>();
        int d = json.at("d").get<int>();
        int q = json.at("q").get<int>();
        return models::ArimaSpec(p, d, q);
    } catch (const nlohmann::json::exception& e) {
        return unexpected<JsonError>({"JSON parse error for ArimaSpec: " + std::string(e.what())});
    } catch (const std::exception& e) {
        return unexpected<JsonError>({"Error creating ArimaSpec: " + std::string(e.what())});
    }
}

expected<models::GarchSpec, JsonError> JsonReader::garchSpecFromJson(const nlohmann::json& json) {
    try {
        int p = json.at("p").get<int>();
        int q = json.at("q").get<int>();
        return models::GarchSpec(p, q);
    } catch (const nlohmann::json::exception& e) {
        return unexpected<JsonError>({"JSON parse error for GarchSpec: " + std::string(e.what())});
    } catch (const std::exception& e) {
        return unexpected<JsonError>({"Error creating GarchSpec: " + std::string(e.what())});
    }
}

expected<models::ArimaGarchSpec, JsonError>
JsonReader::arimaGarchSpecFromJson(const nlohmann::json& json) {
    try {
        auto arima_result = arimaSpecFromJson(json.at("arima"));
        if (!arima_result.has_value()) {
            return unexpected<JsonError>(arima_result.error());
        }

        auto garch_result = garchSpecFromJson(json.at("garch"));
        if (!garch_result.has_value()) {
            return unexpected<JsonError>(garch_result.error());
        }

        return models::ArimaGarchSpec(*arima_result, *garch_result);
    } catch (const nlohmann::json::exception& e) {
        return unexpected<JsonError>(
            {"JSON parse error for ArimaGarchSpec: " + std::string(e.what())});
    } catch (const std::exception& e) {
        return unexpected<JsonError>({"Error creating ArimaGarchSpec: " + std::string(e.what())});
    }
}

expected<models::arima::ArimaParameters, JsonError>
JsonReader::arimaParametersFromJson(const nlohmann::json& json, const models::ArimaSpec& spec) {
    try {
        models::arima::ArimaParameters params(spec.p, spec.q);
        params.intercept = json.at("intercept").get<double>();
        params.ar_coef = json.at("ar_coef").get<std::vector<double>>();
        params.ma_coef = json.at("ma_coef").get<std::vector<double>>();

        // Validate sizes
        if (static_cast<int>(params.ar_coef.size()) != spec.p) {
            return unexpected<JsonError>({"AR coefficient count mismatch: expected " +
                                          std::to_string(spec.p) + ", got " +
                                          std::to_string(params.ar_coef.size())});
        }
        if (static_cast<int>(params.ma_coef.size()) != spec.q) {
            return unexpected<JsonError>({"MA coefficient count mismatch: expected " +
                                          std::to_string(spec.q) + ", got " +
                                          std::to_string(params.ma_coef.size())});
        }

        return params;
    } catch (const nlohmann::json::exception& e) {
        return unexpected<JsonError>(
            {"JSON parse error for ArimaParameters: " + std::string(e.what())});
    }
}

expected<models::garch::GarchParameters, JsonError>
JsonReader::garchParametersFromJson(const nlohmann::json& json, const models::GarchSpec& spec) {
    try {
        models::garch::GarchParameters params(spec.p, spec.q);
        params.omega = json.at("omega").get<double>();
        params.alpha_coef = json.at("alpha_coef").get<std::vector<double>>();
        params.beta_coef = json.at("beta_coef").get<std::vector<double>>();

        // Validate sizes
        if (static_cast<int>(params.alpha_coef.size()) != spec.q) {
            return unexpected<JsonError>({"ARCH coefficient count mismatch: expected " +
                                          std::to_string(spec.q) + ", got " +
                                          std::to_string(params.alpha_coef.size())});
        }
        if (static_cast<int>(params.beta_coef.size()) != spec.p) {
            return unexpected<JsonError>({"GARCH coefficient count mismatch: expected " +
                                          std::to_string(spec.p) + ", got " +
                                          std::to_string(params.beta_coef.size())});
        }

        return params;
    } catch (const nlohmann::json::exception& e) {
        return unexpected<JsonError>(
            {"JSON parse error for GarchParameters: " + std::string(e.what())});
    }
}

expected<models::composite::ArimaGarchParameters, JsonError>
JsonReader::arimaGarchParametersFromJson(const nlohmann::json& json,
                                         const models::ArimaGarchSpec& spec) {
    try {
        auto arima_result = arimaParametersFromJson(json.at("arima"), spec.arimaSpec);
        if (!arima_result.has_value()) {
            return unexpected<JsonError>(arima_result.error());
        }

        auto garch_result = garchParametersFromJson(json.at("garch"), spec.garchSpec);
        if (!garch_result.has_value()) {
            return unexpected<JsonError>(garch_result.error());
        }

        models::composite::ArimaGarchParameters params(spec);
        params.arima_params = *arima_result;
        params.garch_params = *garch_result;
        return params;
    } catch (const nlohmann::json::exception& e) {
        return unexpected<JsonError>(
            {"JSON parse error for ArimaGarchParameters: " + std::string(e.what())});
    }
}

expected<models::arima::ArimaState, JsonError>
JsonReader::arimaStateFromJson(const nlohmann::json& json, const models::ArimaSpec& spec) {
    try {
        models::arima::ArimaState state(spec.p, spec.d, spec.q);

        // Restore the bounded sliding windows directly so the loaded state is
        // byte-for-byte the terminal state that was saved (replaying through
        // update() from a dummy-initialized state would not reproduce it).
        auto obs_history = json.at("observation_history").get<std::vector<double>>();
        auto residual_history = json.at("residual_history").get<std::vector<double>>();
        std::vector<double> differenced_series;
        if (json.contains("differenced_series")) {
            differenced_series = json.at("differenced_series").get<std::vector<double>>();
        }

        state.restore(obs_history, residual_history, differenced_series);

        return state;
    } catch (const nlohmann::json::exception& e) {
        return unexpected<JsonError>({"JSON parse error for ArimaState: " + std::string(e.what())});
    } catch (const std::exception& e) {
        return unexpected<JsonError>({"Error creating ArimaState: " + std::string(e.what())});
    }
}

expected<models::garch::GarchState, JsonError>
JsonReader::garchStateFromJson(const nlohmann::json& json, const models::GarchSpec& spec) {
    try {
        models::garch::GarchState state(spec.p, spec.q);

        // Restore the saved histories and initial variance directly (see the
        // ArimaState helper above for why replaying update() is insufficient).
        double initial_variance = json.at("initial_variance").get<double>();
        auto variance_history = json.at("variance_history").get<std::vector<double>>();
        auto squared_residual_history =
            json.at("squared_residual_history").get<std::vector<double>>();

        state.restore(variance_history, squared_residual_history, initial_variance);

        return state;
    } catch (const nlohmann::json::exception& e) {
        return unexpected<JsonError>({"JSON parse error for GarchState: " + std::string(e.what())});
    } catch (const std::exception& e) {
        return unexpected<JsonError>({"Error creating GarchState: " + std::string(e.what())});
    }
}

expected<ModelMetadata, JsonError> JsonReader::metadataFromJson(const nlohmann::json& json) {
    try {
        std::string timestamp = json.at("timestamp").get<std::string>();
        std::string version = json.at("version").get<std::string>();
        std::string model_type = json.at("model_type").get<std::string>();
        return ModelMetadata(timestamp, version, model_type);
    } catch (const nlohmann::json::exception& e) {
        return unexpected<JsonError>(
            {"JSON parse error for ModelMetadata: " + std::string(e.what())});
    }
}

expected<nlohmann::json, JsonError>
JsonReader::readFromFile(const std::filesystem::path& filepath) {
    try {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            return unexpected<JsonError>({"Failed to open file for reading: " + filepath.string()});
        }

        nlohmann::json j;
        file >> j;

        if (file.fail() && !file.eof()) {
            return unexpected<JsonError>({"Failed to parse JSON from file: " + filepath.string()});
        }

        return j;
    } catch (const nlohmann::json::exception& e) {
        return unexpected<JsonError>({"JSON parse error: " + std::string(e.what())});
    } catch (const std::exception& e) {
        return unexpected<JsonError>(
            {"Exception while reading JSON file: " + std::string(e.what())});
    }
}

expected<models::composite::ArimaGarchModel, JsonError>
JsonReader::loadModel(const std::filesystem::path& filepath) {
    try {
        auto json_result = readFromFile(filepath);
        if (!json_result.has_value()) {
            return unexpected<JsonError>(json_result.error());
        }

        const auto& j = *json_result;

        // Parse spec
        auto spec_result = arimaGarchSpecFromJson(j.at("spec"));
        if (!spec_result.has_value()) {
            return unexpected<JsonError>(spec_result.error());
        }
        const auto& spec = *spec_result;

        // Parse parameters
        auto params_result = arimaGarchParametersFromJson(j.at("parameters"), spec);
        if (!params_result.has_value()) {
            return unexpected<JsonError>(params_result.error());
        }

        // Create model
        models::composite::ArimaGarchModel model(spec, *params_result);

        // Restore the saved state so forecasts continue from the model's
        // terminal state rather than from a fresh zero-history state. Older
        // files without a "state" block keep the constructor's fresh state.
        if (j.contains("state")) {
            const auto& state_json = j.at("state");

            auto arima_state_result = arimaStateFromJson(state_json.at("arima"), spec.arimaSpec);
            if (!arima_state_result.has_value()) {
                return unexpected<JsonError>(arima_state_result.error());
            }

            auto garch_state_result = garchStateFromJson(state_json.at("garch"), spec.garchSpec);
            if (!garch_state_result.has_value()) {
                return unexpected<JsonError>(garch_state_result.error());
            }

            model.restoreState(*arima_state_result, *garch_state_result);

            // Restore the differencing anchors so d > 0 models integrate
            // forecasts from the saved terminal levels. Files written before
            // this block existed simply keep the constructor's fresh (zero)
            // anchors.
            if (state_json.contains("differencer")) {
                const auto& diff_json = state_json.at("differencer");
                int order = diff_json.at("order").get<int>();
                if (order != spec.arimaSpec.d) {
                    return unexpected<JsonError>({"Differencer order mismatch: expected " +
                                                  std::to_string(spec.arimaSpec.d) + ", got " +
                                                  std::to_string(order)});
                }
                auto anchors = diff_json.at("anchors").get<std::vector<double>>();
                auto observation_count = diff_json.at("observation_count").get<std::size_t>();

                ag::util::StreamingDifferencer differencer(order);
                differencer.restore(anchors, observation_count);
                model.restoreDifferencer(differencer);
            }
        }

        return model;
    } catch (const nlohmann::json::exception& e) {
        return unexpected<JsonError>(
            {"JSON parse error while loading model: " + std::string(e.what())});
    } catch (const std::exception& e) {
        return unexpected<JsonError>({"Exception while loading model: " + std::string(e.what())});
    }
}

}  // namespace ag::io
