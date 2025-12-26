/**
 * @file bench_likelihood.cpp
 * @brief Benchmark for ARIMA-GARCH likelihood computation.
 *
 * This benchmark measures the performance of likelihood evaluation on mid-size
 * synthetic time series data. It helps track performance regressions in the
 * likelihood computation path.
 */

#include "ag/estimation/Likelihood.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/arima/ArimaModel.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/models/garch/GarchModel.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"
#include "ag/util/Timer.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include <fmt/core.h>

using namespace ag;

/**
 * @brief Benchmark configuration
 */
struct BenchmarkConfig {
    std::size_t data_size;    // Number of observations
    int num_iterations;       // Number of benchmark iterations
    std::string description;  // Description of benchmark
};

/**
 * @brief Run a single likelihood benchmark
 */
void runLikelihoodBenchmark(const BenchmarkConfig& config, const models::ArimaGarchSpec& spec,
                            const std::vector<double>& data) {
    using namespace estimation;
    using namespace models;
    using namespace models::arima;
    using namespace models::garch;

    // Create likelihood evaluator
    ArimaGarchLikelihood likelihood(spec);

    // Set up reasonable parameters
    ArimaParameters arima_params(spec.arimaSpec.p, spec.arimaSpec.q);
    arima_params.intercept = 0.05;
    for (std::size_t i = 0; i < spec.arimaSpec.p; ++i) {
        arima_params.ar_coef[i] = 0.5 / (i + 1);
    }
    for (std::size_t i = 0; i < spec.arimaSpec.q; ++i) {
        arima_params.ma_coef[i] = 0.3 / (i + 1);
    }

    GarchParameters garch_params(spec.garchSpec.p, spec.garchSpec.q);
    garch_params.omega = 0.01;
    for (std::size_t i = 0; i < spec.garchSpec.p; ++i) {
        garch_params.alpha_coef[i] = 0.1 / (i + 1);
    }
    for (std::size_t i = 0; i < spec.garchSpec.q; ++i) {
        garch_params.beta_coef[i] = 0.85 / (i + 1);
    }

    // Warm-up run (not timed) - also validates that parameters are valid
    double warmup_nll = likelihood.computeNegativeLogLikelihood(data.data(), data.size(),
                                                                arima_params, garch_params);
    if (!std::isfinite(warmup_nll)) {
        fmt::print("  Warning: Invalid warmup NLL for {}: {}\n", config.description, warmup_nll);
        return;
    }

    // Benchmark iterations
    Timer timer;
    timer.start();

    for (int i = 0; i < config.num_iterations; ++i) {
        [[maybe_unused]] double nll = likelihood.computeNegativeLogLikelihood(
            data.data(), data.size(), arima_params, garch_params);
    }

    double total_time = timer.stop();
    double avg_time = total_time / config.num_iterations;
    double throughput = config.data_size / avg_time;  // observations per second

    // Print results
    fmt::print("  {:40s} | {:>8.2f} ms | {:>10.0f} obs/s\n", config.description, avg_time * 1000.0,
               throughput);
}

/**
 * @brief Generate synthetic data for benchmarking
 */
std::vector<double> generateSyntheticData(std::size_t size, unsigned int seed) {
    // Create a simple ARIMA(1,0,1)-GARCH(1,1) model for data generation
    models::ArimaGarchSpec spec(1, 0, 1, 1, 1);
    models::composite::ArimaGarchParameters params(spec);

    params.arima_params.intercept = 0.05;
    params.arima_params.ar_coef[0] = 0.6;
    params.arima_params.ma_coef[0] = 0.3;
    params.garch_params.omega = 0.01;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.85;

    simulation::ArimaGarchSimulator simulator(spec, params);
    auto result = simulator.simulate(size, seed);

    return result.returns;
}

int main() {
    fmt::print("\n");
    fmt::print("=================================================================\n");
    fmt::print("  ARIMA-GARCH Likelihood Computation Benchmark\n");
    fmt::print("=================================================================\n");
    fmt::print("\n");
    fmt::print("This benchmark measures likelihood evaluation performance on\n");
    fmt::print("mid-size synthetic time series data.\n");
    fmt::print("\n");

    // Generate benchmark data
    const std::size_t DATA_SIZE = 5000;  // Mid-size time series
    const int NUM_ITERATIONS = 100;      // Number of benchmark runs

    fmt::print("Generating synthetic data (n={})...\n", DATA_SIZE);
    std::vector<double> data = generateSyntheticData(DATA_SIZE, 42);
    fmt::print("Data generation complete.\n\n");

    // Benchmark header
    fmt::print("Running benchmarks ({} iterations each):\n\n", NUM_ITERATIONS);
    fmt::print("  {:40s} | {:>10s} | {:>14s}\n", "Model Specification", "Avg Time", "Throughput");
    fmt::print("  {:-<40s}-+-{:-<10s}-+-{:-<14s}\n", "", "", "");

    // Benchmark different model specifications
    std::vector<std::tuple<models::ArimaGarchSpec, std::string>> specs = {
        {models::ArimaGarchSpec(0, 0, 0, 1, 1), "ARIMA(0,0,0)-GARCH(1,1)"},
        {models::ArimaGarchSpec(1, 0, 0, 1, 1), "ARIMA(1,0,0)-GARCH(1,1)"},
        {models::ArimaGarchSpec(1, 0, 1, 1, 1), "ARIMA(1,0,1)-GARCH(1,1)"},
        {models::ArimaGarchSpec(2, 0, 1, 1, 1), "ARIMA(2,0,1)-GARCH(1,1)"},
        {models::ArimaGarchSpec(2, 0, 2, 1, 1), "ARIMA(2,0,2)-GARCH(1,1)"},
    };

    for (const auto& [spec, description] : specs) {
        BenchmarkConfig config{DATA_SIZE, NUM_ITERATIONS, description};
        runLikelihoodBenchmark(config, spec, data);
    }

    fmt::print("\n");
    fmt::print("Benchmark complete.\n");
    fmt::print("\n");
    fmt::print("Performance notes:\n");
    fmt::print("  - Higher throughput (obs/s) is better\n");
    fmt::print("  - Typical range: 10,000-100,000 obs/s depending on model complexity\n");
    fmt::print("  - More complex models (higher p, q) are slower\n");
    fmt::print("\n");

    return 0;
}
