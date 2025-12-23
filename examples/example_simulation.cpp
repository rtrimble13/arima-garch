#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"
#include "ag/stats/Descriptive.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

using ag::models::ArimaGarchSpec;
using ag::models::composite::ArimaGarchParameters;
using ag::simulation::ArimaGarchSimulator;

int main() {
    std::cout << "=== ARIMA-GARCH Simulation Example ===" << std::endl << std::endl;

    // Define an ARIMA(1,0,1)-GARCH(1,1) model
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    // Set ARIMA parameters for a mean-reverting process
    params.arima_params.intercept = 0.05;  // Small positive drift
    params.arima_params.ar_coef[0] = 0.6;  // Moderate persistence
    params.arima_params.ma_coef[0] = 0.3;  // Some MA effect

    // Set GARCH parameters for realistic volatility clustering
    params.garch_params.omega = 0.01;         // Base volatility
    params.garch_params.alpha_coef[0] = 0.1;  // ARCH effect (news impact)
    params.garch_params.beta_coef[0] = 0.85;  // GARCH effect (persistence)

    std::cout << "Model specification: ARIMA(" << spec.arimaSpec.p << "," << spec.arimaSpec.d << ","
              << spec.arimaSpec.q << ")-GARCH(" << spec.garchSpec.p << "," << spec.garchSpec.q
              << ")" << std::endl;
    std::cout << std::endl;

    std::cout << "ARIMA parameters:" << std::endl;
    std::cout << "  Intercept: " << params.arima_params.intercept << std::endl;
    std::cout << "  AR[1]: " << params.arima_params.ar_coef[0] << std::endl;
    std::cout << "  MA[1]: " << params.arima_params.ma_coef[0] << std::endl;
    std::cout << std::endl;

    std::cout << "GARCH parameters:" << std::endl;
    std::cout << "  Omega: " << params.garch_params.omega << std::endl;
    std::cout << "  Alpha[1]: " << params.garch_params.alpha_coef[0] << std::endl;
    std::cout << "  Beta[1]: " << params.garch_params.beta_coef[0] << std::endl;
    std::cout << std::endl;

    // Create simulator
    ArimaGarchSimulator simulator(spec, params);

    // Simulate a path
    int simulation_length = 1000;
    unsigned int seed = 42;
    std::cout << "Simulating " << simulation_length << " observations with seed " << seed << "..."
              << std::endl;

    auto result = simulator.simulate(simulation_length, seed);

    std::cout << "Simulation complete!" << std::endl << std::endl;

    // Compute summary statistics
    double mean_ret = ag::stats::mean(result.returns);
    double std_ret = std::sqrt(ag::stats::variance(result.returns));
    double min_ret = *std::min_element(result.returns.begin(), result.returns.end());
    double max_ret = *std::max_element(result.returns.begin(), result.returns.end());
    double skew_ret = ag::stats::skewness(result.returns);
    double kurt_ret = ag::stats::kurtosis(result.returns);

    std::cout << "Summary statistics of simulated returns:" << std::endl;
    std::cout << "  Mean: " << mean_ret << std::endl;
    std::cout << "  Std Dev: " << std_ret << std::endl;
    std::cout << "  Min: " << min_ret << std::endl;
    std::cout << "  Max: " << max_ret << std::endl;
    std::cout << "  Skewness: " << skew_ret << std::endl;
    std::cout << "  Kurtosis: " << kurt_ret << std::endl;
    std::cout << std::endl;

    // Compute volatility statistics
    double mean_vol = ag::stats::mean(result.volatilities);
    double std_vol = std::sqrt(ag::stats::variance(result.volatilities));
    double min_vol = *std::min_element(result.volatilities.begin(), result.volatilities.end());
    double max_vol = *std::max_element(result.volatilities.begin(), result.volatilities.end());

    std::cout << "Summary statistics of conditional volatility:" << std::endl;
    std::cout << "  Mean: " << mean_vol << std::endl;
    std::cout << "  Std Dev: " << std_vol << std::endl;
    std::cout << "  Min: " << min_vol << std::endl;
    std::cout << "  Max: " << max_vol << std::endl;
    std::cout << std::endl;

    // Demonstrate reproducibility
    std::cout << "Demonstrating reproducibility..." << std::endl;
    auto result2 = simulator.simulate(simulation_length, seed);

    bool identical = true;
    for (size_t i = 0; i < result.returns.size(); ++i) {
        if (result.returns[i] != result2.returns[i]) {
            identical = false;
            break;
        }
    }

    if (identical) {
        std::cout << "✓ Same seed produces identical output" << std::endl;
    } else {
        std::cout << "✗ Reproducibility check failed" << std::endl;
    }
    std::cout << std::endl;

    // Show first 10 simulated values
    std::cout << "First 10 simulated returns (Normal):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "  t=" << i + 1 << ": return=" << result.returns[i]
                  << ", volatility=" << result.volatilities[i] << std::endl;
    }
    std::cout << std::endl;

    // Demonstrate Student-t innovations
    std::cout << "=== Simulation with Student-t Innovations ===" << std::endl << std::endl;

    double df = 5.0;  // Degrees of freedom
    std::cout << "Simulating with Student-t(" << df << ") innovations..." << std::endl;

    auto result_t = simulator.simulate(simulation_length, seed,
                                       ag::simulation::InnovationDistribution::StudentT, df);

    std::cout << "Simulation complete!" << std::endl << std::endl;

    // Compute statistics for Student-t simulation
    double mean_ret_t = ag::stats::mean(result_t.returns);
    double std_ret_t = std::sqrt(ag::stats::variance(result_t.returns));
    double skew_ret_t = ag::stats::skewness(result_t.returns);
    double kurt_ret_t = ag::stats::kurtosis(result_t.returns);

    std::cout << "Summary statistics (Student-t):" << std::endl;
    std::cout << "  Mean: " << mean_ret_t << std::endl;
    std::cout << "  Std Dev: " << std_ret_t << std::endl;
    std::cout << "  Skewness: " << skew_ret_t << std::endl;
    std::cout << "  Kurtosis: " << kurt_ret_t << " (expect higher than Normal)" << std::endl;
    std::cout << std::endl;

    std::cout << "First 10 simulated returns (Student-t):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "  t=" << i + 1 << ": return=" << result_t.returns[i]
                  << ", volatility=" << result_t.volatilities[i] << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Note: Student-t innovations typically produce heavier tails (higher kurtosis)"
              << std::endl;
    std::cout
        << "      compared to normal innovations, which is useful for modeling extreme events."
        << std::endl;

    return 0;
}
