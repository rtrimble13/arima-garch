#pragma once

#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/simulation/Innovations.hpp"

#include <optional>
#include <vector>

namespace ag::simulation {

/**
 * @brief Innovation distribution type for simulation.
 */
enum class InnovationDistribution {
    Normal,   // Standard normal N(0,1)
    StudentT  // Standardized Student-t with specified degrees of freedom
};

/**
 * @brief Result of an ARIMA-GARCH simulation.
 *
 * Contains the simulated return paths and conditional variances
 * generated from an ARIMA-GARCH model.
 */
struct SimulationResult {
    std::vector<double> returns;       // Simulated return series
    std::vector<double> volatilities;  // Conditional standard deviations (sqrt of variances)

    /**
     * @brief Construct a SimulationResult with specified size.
     * @param size Number of simulated observations
     */
    explicit SimulationResult(int size) : returns(size, 0.0), volatilities(size, 0.0) {}
};

/**
 * @brief Simulator for ARIMA-GARCH models.
 *
 * ArimaGarchSimulator generates synthetic time series paths from a fitted
 * ARIMA-GARCH model with specified parameters. The simulator uses a two-step
 * recursive process:
 *
 * 1. Generate standardized innovations z_t ~ N(0,1) or Student-t
 * 2. For each time step:
 *    a. Compute conditional mean μ_t from ARIMA recursion
 *    b. Compute conditional variance h_t from GARCH recursion
 *    c. Generate return: y_t = μ_t + sqrt(h_t) * z_t
 *    d. Update states with y_t and residual ε_t = y_t - μ_t
 *
 * The simulator is deterministic given a seed, ensuring reproducible results.
 */
class ArimaGarchSimulator {
public:
    /**
     * @brief Construct a simulator with model specification and parameters.
     *
     * @param spec ARIMA-GARCH model specification
     * @param params Model parameters (fitted coefficients)
     */
    ArimaGarchSimulator(const ag::models::ArimaGarchSpec& spec,
                        const ag::models::composite::ArimaGarchParameters& params);

    /**
     * @brief Simulate a return path from the ARIMA-GARCH model.
     *
     * Generates a synthetic time series of specified length using the model
     * parameters and a seeded random number generator for innovations.
     *
     * By default, uses standard normal innovations N(0,1). Optionally,
     * Student-t innovations can be used by specifying distribution type
     * and degrees of freedom.
     *
     * The method:
     * 1. Initializes an Innovations generator with the given seed
     * 2. Creates an ArimaGarchModel with the stored parameters
     * 3. For each time step:
     *    - Draws an innovation z_t from the specified distribution
     *    - Computes conditional mean μ_t and variance h_t
     *    - Generates return y_t = μ_t + sqrt(h_t) * z_t
     *    - Updates model state
     * 4. Returns the simulated path and volatilities
     *
     * @param length Number of observations to simulate (must be > 0)
     * @param seed Random seed for reproducible results
     * @param dist_type Distribution type for innovations (default: Normal)
     * @param df Degrees of freedom for Student-t (required if dist_type=StudentT, must be > 2)
     * @return SimulationResult containing returns and volatilities
     * @throws std::invalid_argument if length <= 0 or df invalid for Student-t
     */
    [[nodiscard]] SimulationResult
    simulate(int length, unsigned int seed,
             InnovationDistribution dist_type = InnovationDistribution::Normal,
             std::optional<double> df = std::nullopt) const;

private:
    ag::models::ArimaGarchSpec spec_;                     // Model specification
    ag::models::composite::ArimaGarchParameters params_;  // Model parameters
};

}  // namespace ag::simulation
