#pragma once

#include "ag/diagnostics/DiagnosticReport.hpp"
#include "ag/forecasting/Forecaster.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/report/FitSummary.hpp"
#include "ag/selection/ModelSelector.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"
#include "ag/util/Expected.hpp"

#include <memory>
#include <vector>

namespace ag::api {

/**
 * @brief Error type for Engine operations.
 */
struct EngineError {
    std::string message;
};

/**
 * @brief Result of fitting an ARIMA-GARCH model.
 *
 * Contains the fitted model and a comprehensive summary of the fit including
 * parameters, convergence information, information criteria, and diagnostics.
 */
struct FitResult {
    std::shared_ptr<models::composite::ArimaGarchModel> model;
    report::FitSummary summary;

    FitResult(std::shared_ptr<models::composite::ArimaGarchModel> m, const report::FitSummary& s)
        : model(std::move(m)), summary(s) {}
};

/**
 * @brief Result of automatic model selection.
 *
 * Contains the selected model specification, fitted model, and selection statistics.
 */
struct SelectionResult {
    models::ArimaGarchSpec selected_spec;
    std::shared_ptr<models::composite::ArimaGarchModel> model;
    report::FitSummary summary;
    std::size_t candidates_evaluated;
    std::size_t candidates_failed;
    std::vector<selection::CandidateRanking> ranking;

    SelectionResult(const models::ArimaGarchSpec& spec,
                    std::shared_ptr<models::composite::ArimaGarchModel> m,
                    const report::FitSummary& s, std::size_t evaluated, std::size_t failed)
        : selected_spec(spec), model(std::move(m)), summary(s), candidates_evaluated(evaluated),
          candidates_failed(failed) {}
};

/**
 * @brief Engine facade for ARIMA-GARCH modeling operations.
 *
 * Engine provides a high-level, stable API for all major operations:
 * - fit(): Fit a model to time series data
 * - auto_select(): Automatically select the best model from candidates
 * - forecast(): Generate forecasts from a fitted model
 * - simulate(): Generate synthetic time series from a specification
 *
 * This is the primary entrypoint for CLI and high-level application code.
 * The Engine handles all details of parameter initialization, optimization,
 * model building, and diagnostic computation.
 *
 * Usage pattern:
 * ```cpp
 * Engine engine;
 *
 * // Fit a specific model
 * auto fit_result = engine.fit(data, spec);
 * if (!fit_result) {
 *     std::cerr << "Fit failed: " << fit_result.error().message << "\n";
 *     return 1;
 * }
 *
 * // Generate forecasts
 * auto forecast_result = engine.forecast(*fit_result.value().model, 10);
 * ```
 */
class Engine {
public:
    /**
     * @brief Construct an Engine with default settings.
     */
    Engine();

    /**
     * @brief Fit an ARIMA-GARCH model to time series data.
     *
     * This method performs the complete model fitting workflow:
     * 1. Initialize parameters from the data
     * 2. Build likelihood function for the specification
     * 3. Run optimization (Nelder-Mead with random restarts)
     * 4. Build the fitted model with optimized parameters
     * 5. Compute diagnostic tests on residuals
     * 6. Generate a comprehensive FitSummary
     *
     * The optimization uses:
     * - Nelder-Mead algorithm (derivative-free)
     * - Random restarts for improved global convergence
     * - Automatic constraint checking (positivity, stationarity)
     * - Robust error handling
     *
     * @param data Time series data (must have at least 10 observations)
     * @param spec ARIMA-GARCH model specification
     * @param compute_diagnostics Whether to compute diagnostic tests (default: true)
     * @return FitResult on success, EngineError on failure
     * @throws None - all errors returned via expected
     */
    [[nodiscard]] expected<FitResult, EngineError> fit(const std::vector<double>& data,
                                                       const models::ArimaGarchSpec& spec,
                                                       bool compute_diagnostics = true);

    /**
     * @brief Automatically select and fit the best model from candidates.
     *
     * This method performs model selection followed by fitting:
     * 1. Evaluate all candidate specifications
     * 2. Select the best model according to the selection criterion
     * 3. Fit the best model with diagnostic computation
     * 4. Return the fitted model and selection statistics
     *
     * The selection process is robust to individual fit failures and will
     * continue evaluating remaining candidates.
     *
     * @param data Time series data (must have at least 10 observations)
     * @param candidates Vector of candidate ARIMA-GARCH specifications
     * @param criterion Selection criterion (default: BIC)
     * @param build_ranking If true, include ranking of all models in result
     * @return SelectionResult on success, EngineError if all candidates fail
     * @throws None - all errors returned via expected
     */
    [[nodiscard]] expected<SelectionResult, EngineError>
    auto_select(const std::vector<double>& data,
                const std::vector<models::ArimaGarchSpec>& candidates,
                selection::SelectionCriterion criterion = selection::SelectionCriterion::BIC,
                bool build_ranking = false);

    /**
     * @brief Generate forecasts from a fitted model.
     *
     * Produces h-step ahead forecasts for both conditional mean and variance
     * using the model's current state (most recent observations).
     *
     * @param model Fitted ARIMA-GARCH model (must be initialized with data)
     * @param horizon Number of steps ahead to forecast (must be > 0)
     * @return ForecastResult on success, EngineError on failure
     * @throws None - all errors returned via expected
     */
    [[nodiscard]] expected<forecasting::ForecastResult, EngineError>
    forecast(const models::composite::ArimaGarchModel& model, int horizon);

    /**
     * @brief Simulate synthetic time series from an ARIMA-GARCH model.
     *
     * Generates synthetic data using the specified model parameters and
     * random innovations drawn from a standard normal distribution.
     *
     * @param spec ARIMA-GARCH model specification
     * @param params Model parameters (coefficients)
     * @param length Number of observations to simulate (must be > 0)
     * @param seed Random seed for reproducibility
     * @return SimulationResult on success, EngineError on failure
     * @throws None - all errors returned via expected
     */
    [[nodiscard]] expected<simulation::SimulationResult, EngineError>
    simulate(const models::ArimaGarchSpec& spec,
             const models::composite::ArimaGarchParameters& params, int length, unsigned int seed);

private:
    // Configuration parameters for optimization
    static constexpr double OPTIMIZER_FTOL = 1e-6;
    static constexpr double OPTIMIZER_XTOL = 1e-6;
    static constexpr int OPTIMIZER_MAX_ITER = 2000;
    static constexpr int NUM_RESTARTS = 3;
    static constexpr double PERTURBATION_SCALE = 0.15;

    /**
     * @brief Helper to pack ARIMA-GARCH parameters into a vector.
     * @param arima_params ARIMA parameters
     * @param garch_params GARCH parameters
     * @return Vector of packed parameters
     */
    std::vector<double> packParameters(const models::arima::ArimaParameters& arima_params,
                                       const models::garch::GarchParameters& garch_params) const;

    /**
     * @brief Helper to unpack vector into ARIMA-GARCH parameters.
     * @param params Parameter vector
     * @param spec Model specification
     * @param out_arima Output ARIMA parameters
     * @param out_garch Output GARCH parameters
     */
    void unpackParameters(const std::vector<double>& params, const models::ArimaGarchSpec& spec,
                          models::arima::ArimaParameters& out_arima,
                          models::garch::GarchParameters& out_garch) const;
};

}  // namespace ag::api
