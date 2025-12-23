#pragma once

#include <random>

namespace ag::simulation {

/**
 * @brief Random number generator wrapper for producing innovations.
 *
 * Innovations provides a deterministic, seeded RNG for generating
 * standardized innovations (errors) from various distributions.
 * The class is designed to ensure reproducible simulation results
 * across runs when initialized with the same seed.
 *
 * Currently supports:
 * - Normal (Gaussian) distribution with mean 0 and variance 1
 * - Student-t distribution (stub for future implementation)
 */
class Innovations {
public:
    /**
     * @brief Construct an Innovations generator with a specified seed.
     *
     * @param seed Random seed for reproducible results. Same seed on the same
     *             platform/compiler combination will produce identical sequences.
     */
    explicit Innovations(unsigned int seed);

    /**
     * @brief Generate a standard normal innovation (N(0,1)).
     *
     * Produces a random draw from the standard normal distribution
     * with mean 0 and variance 1.
     *
     * @return A standard normal random variable
     */
    [[nodiscard]] double drawNormal();

    /**
     * @brief Generate a Student-t innovation (stub).
     *
     * This is a placeholder for future Student-t distribution support.
     * Currently throws std::runtime_error.
     *
     * @param df Degrees of freedom (unused in current implementation)
     * @return A Student-t random variable
     * @throws std::runtime_error Always, as this is not yet implemented
     */
    [[nodiscard]] double drawStudentT(double df);

    /**
     * @brief Reset the RNG state with a new seed.
     *
     * This allows reusing the same Innovations object to generate
     * a new reproducible sequence.
     *
     * @param seed New random seed
     */
    void reseed(unsigned int seed);

private:
    std::mt19937 rng_;                         // Mersenne Twister RNG engine
    std::normal_distribution<double> normal_;  // Standard normal distribution N(0,1)
};

}  // namespace ag::simulation
