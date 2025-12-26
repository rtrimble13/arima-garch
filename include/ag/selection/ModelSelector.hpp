#pragma once

#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/report/FitSummary.hpp"

#include <optional>
#include <vector>

namespace ag::selection {

/**
 * @brief Information criterion for model selection.
 *
 * Specifies which information criterion to use when comparing candidate models.
 * Each criterion balances goodness of fit with model complexity differently:
 *
 * - BIC (Bayesian Information Criterion): Penalizes complexity more heavily,
 *   especially for larger sample sizes. Often preferred for model selection.
 * - AIC (Akaike Information Criterion): Less penalty for complexity than BIC.
 *   May favor more complex models.
 * - AICc (Corrected AIC): Corrected version of AIC for small sample sizes.
 *   Use when n/k < 40.
 * - CV (Cross-Validation): Uses rolling origin cross-validation with MSE of
 *   1-step-ahead forecasts. Selects based on out-of-sample forecast performance.
 */
enum class SelectionCriterion {
    BIC,   // Bayesian Information Criterion (default)
    AIC,   // Akaike Information Criterion
    AICc,  // Corrected Akaike Information Criterion
    CV,    // Cross-Validation (rolling origin, 1-step-ahead MSE)
};

/**
 * @brief Information about a single evaluated candidate model.
 *
 * Contains the specification and selection score for a candidate that was
 * successfully fitted during the selection process.
 */
struct CandidateRanking {
    int p, d, q;           // ARIMA orders
    int garch_p, garch_q;  // GARCH orders
    double score;          // IC or CV score (lower is better)
    bool converged;

    CandidateRanking(const ag::models::ArimaGarchSpec& s, double sc, bool conv)
        : p(s.arimaSpec.p), d(s.arimaSpec.d), q(s.arimaSpec.q), garch_p(s.garchSpec.p),
          garch_q(s.garchSpec.q), score(sc), converged(conv) {}
};

/**
 * @brief Result of model selection containing the best model and metadata.
 *
 * SelectionResult encapsulates the outcome of model selection, including:
 * - The best model specification
 * - The IC score of the best model
 * - The fitted parameters of the best model
 * - A full FitSummary with convergence info and diagnostics (optional)
 * - Statistics about the selection process
 * - Ranking of all evaluated candidates (optional)
 */
struct SelectionResult {
    /**
     * @brief The specification of the best model.
     */
    ag::models::ArimaGarchSpec best_spec;

    /**
     * @brief The information criterion score of the best model.
     *
     * Lower values indicate better models. For IC-based selection (BIC, AIC, AICc),
     * this is the IC score. For CV-based selection, this is the MSE score.
     */
    double best_score;

    /**
     * @brief Estimated parameters of the best model.
     */
    ag::models::composite::ArimaGarchParameters best_parameters;

    /**
     * @brief Complete fit summary for the best model (optional).
     *
     * Includes convergence information, diagnostic tests, and all IC scores.
     */
    std::optional<ag::report::FitSummary> best_fit_summary;

    /**
     * @brief Number of candidate models evaluated.
     */
    std::size_t candidates_evaluated;

    /**
     * @brief Number of candidates that failed to fit.
     */
    std::size_t candidates_failed;

    /**
     * @brief Ranking of all successfully fitted candidates (sorted by score, best first).
     *
     * Empty if ranking was not requested during selection.
     */
    std::vector<CandidateRanking> ranking;

    /**
     * @brief Construct a SelectionResult with required fields.
     * @param spec The best model specification
     * @param score The IC score of the best model
     * @param params The fitted parameters of the best model
     */
    SelectionResult(const ag::models::ArimaGarchSpec& spec, double score,
                    const ag::models::composite::ArimaGarchParameters& params)
        : best_spec(spec), best_score(score), best_parameters(params), candidates_evaluated(0),
          candidates_failed(0) {}
};

/**
 * @brief Model selector using information criteria for candidate comparison.
 *
 * ModelSelector fits a list of candidate ARIMA-GARCH specifications to time series
 * data and selects the best model according to a specified information criterion
 * (BIC by default).
 *
 * For each candidate:
 * 1. Parameters are initialized
 * 2. Model is fitted via maximum likelihood estimation
 * 3. Information criterion (IC) score is computed
 * 4. Best model is tracked (lowest IC score)
 *
 * The selection process is robust to individual model fitting failures: if a
 * candidate fails to fit (e.g., due to convergence issues or numerical problems),
 * it is skipped and the search continues.
 *
 * Usage pattern:
 * ```cpp
 * // Generate candidates
 * CandidateGridConfig config(2, 1, 2, 1, 1);
 * CandidateGrid grid(config);
 * auto candidates = grid.generate();
 *
 * // Select best model
 * ModelSelector selector(SelectionCriterion::BIC);
 * auto result = selector.select(data, candidates);
 *
 * // Use result
 * std::cout << "Best model: ARIMA(" << result.best_spec.arimaSpec.p << ","
 *           << result.best_spec.arimaSpec.d << ","
 *           << result.best_spec.arimaSpec.q << ")-GARCH("
 *           << result.best_spec.garchSpec.p << ","
 *           << result.best_spec.garchSpec.q << ")\n";
 * std::cout << "BIC: " << result.best_score << "\n";
 * ```
 */
class ModelSelector {
public:
    /**
     * @brief Construct a ModelSelector with the specified criterion.
     * @param criterion Information criterion to use for model comparison (default: BIC)
     */
    explicit ModelSelector(SelectionCriterion criterion = SelectionCriterion::BIC);

    /**
     * @brief Select the best model from a list of candidates.
     *
     * This method fits each candidate specification to the provided data and
     * selects the model with the lowest score (IC or CV MSE).
     *
     * The fitting process uses:
     * - Parameter initialization from the data
     * - Nelder-Mead optimization with random restarts
     * - Standard convergence criteria
     *
     * For IC-based selection (BIC, AIC, AICc), models are scored using the
     * specified information criterion on the full dataset.
     *
     * For CV-based selection, models are scored using rolling origin
     * cross-validation with 1-step-ahead MSE. This is more computationally
     * expensive but provides a better measure of out-of-sample forecast performance.
     *
     * Candidates that fail to fit (e.g., convergence failure, numerical issues)
     * are automatically skipped. If all candidates fail, the method returns
     * an empty optional.
     *
     * @param data Pointer to time series data array
     * @param n_obs Number of observations in the data
     * @param candidates Vector of candidate specifications to evaluate
     * @param compute_diagnostics If true, compute diagnostic tests for best model (IC only)
     * @param build_ranking If true, populate the ranking field with all fitted models
     * @return SelectionResult with best model, or empty if all candidates failed
     *
     * @throws std::invalid_argument if data is nullptr, n_obs is 0, or candidates is empty
     *
     * @note CV-based selection is computationally expensive: it requires fitting
     *       each candidate multiple times (once per rolling window). Consider using
     *       a smaller candidate set or IC-based selection for large datasets.
     */
    [[nodiscard]] std::optional<SelectionResult>
    select(const double* data, std::size_t n_obs,
           const std::vector<ag::models::ArimaGarchSpec>& candidates,
           bool compute_diagnostics = false, bool build_ranking = false);

    /**
     * @brief Get the selection criterion being used.
     * @return The information criterion for model selection
     */
    [[nodiscard]] SelectionCriterion getCriterion() const noexcept { return criterion_; }

    /**
     * @brief Set the selection criterion.
     * @param criterion New information criterion to use
     */
    void setCriterion(SelectionCriterion criterion) noexcept { criterion_ = criterion; }

private:
    SelectionCriterion criterion_;  // Information criterion for selection

    /**
     * @brief Fit a single candidate model and compute its IC score.
     *
     * @param data Pointer to time series data
     * @param n_obs Number of observations
     * @param spec Candidate specification to fit
     * @param out_summary Output parameter for fit summary (populated on success)
     * @return IC score if fitting succeeds, empty optional if fitting fails
     */
    [[nodiscard]] std::optional<double> fitAndScore(const double* data, std::size_t n_obs,
                                                    const ag::models::ArimaGarchSpec& spec,
                                                    ag::report::FitSummary& out_summary);

    /**
     * @brief Extract the appropriate IC score from a FitSummary based on criterion.
     * @param summary Fit summary containing IC scores
     * @return The IC score corresponding to the current criterion
     */
    [[nodiscard]] double extractScore(const ag::report::FitSummary& summary) const noexcept;
};

}  // namespace ag::selection
