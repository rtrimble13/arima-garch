#pragma once

#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace ag::selection {

/**
 * @brief Configuration for rolling origin cross-validation.
 *
 * Specifies parameters for time series cross-validation using a rolling origin
 * (also known as rolling forecast origin or walk-forward validation).
 *
 * The rolling origin approach:
 * 1. Starts with an initial training window of size `min_train_size`
 * 2. Fits the model and makes a 1-step-ahead forecast
 * 3. Rolls the window forward by 1 observation
 * 4. Repeats until the end of the data
 *
 * This produces a sequence of 1-step-ahead forecast errors that are used
 * to compute the Mean Squared Error (MSE) score.
 */
struct CrossValidationConfig {
    /**
     * @brief Minimum size of the training window.
     *
     * Must be large enough to reliably fit the model. A common rule of thumb
     * is at least 50-100 observations, though this depends on model complexity.
     */
    std::size_t min_train_size;

    /**
     * @brief Number of forecast steps to evaluate (currently fixed at 1).
     *
     * For initial implementation, only 1-step-ahead forecasts are supported.
     * Future extensions could support multi-step forecasts.
     */
    int horizon;

    /**
     * @brief Construct a CrossValidationConfig with default horizon of 1.
     * @param min_train Minimum training window size
     */
    explicit CrossValidationConfig(std::size_t min_train) : min_train_size(min_train), horizon(1) {}
};

/**
 * @brief Result of a cross-validation evaluation.
 *
 * Contains the MSE score and metadata about the CV process.
 */
struct CrossValidationResult {
    /**
     * @brief Mean Squared Error of 1-step-ahead forecasts.
     *
     * Lower values indicate better out-of-sample forecast performance.
     */
    double mse;

    /**
     * @brief Number of forecast windows evaluated.
     *
     * This is typically n_obs - min_train_size, where n_obs is the
     * total number of observations.
     */
    std::size_t n_windows;

    /**
     * @brief Construct a CrossValidationResult.
     * @param mse_score MSE of 1-step-ahead forecasts
     * @param windows Number of forecast windows
     */
    CrossValidationResult(double mse_score, std::size_t windows)
        : mse(mse_score), n_windows(windows) {}
};

/**
 * @brief Compute cross-validation score using rolling origin.
 *
 * This function evaluates a model specification using time series cross-validation
 * with a rolling origin approach. For each window:
 * 1. Fit the model to the training data
 * 2. Make a 1-step-ahead forecast
 * 3. Compare forecast to actual value
 * 4. Roll the window forward
 *
 * The Mean Squared Error (MSE) across all windows is returned as the score.
 *
 * @param data Pointer to time series data array
 * @param n_obs Number of observations in the data
 * @param spec Model specification to evaluate
 * @param config Cross-validation configuration
 * @return CrossValidationResult with MSE score, or empty if CV fails
 *
 * @note This function can be computationally expensive as it requires fitting
 *       the model multiple times (once per window). The number of windows is
 *       approximately n_obs - min_train_size.
 *
 * @note If any individual model fit fails during CV, the function returns
 *       std::nullopt to indicate that CV could not be completed for this model.
 */
[[nodiscard]] std::optional<CrossValidationResult>
computeCrossValidationScore(const double* data, std::size_t n_obs,
                            const ag::models::ArimaGarchSpec& spec,
                            const CrossValidationConfig& config);

}  // namespace ag::selection
