/**
 * @file bench_optimizer.cpp
 * @brief Benchmark for ARIMA-GARCH model optimization.
 *
 * This benchmark measures the performance of model fitting (parameter estimation)
 * on standard optimization tasks. It helps track performance regressions in the
 * optimization path.
 */

#include "ag/estimation/Likelihood.hpp"
#include "ag/estimation/Optimizer.hpp"
#include "ag/estimation/ParameterInitialization.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/arima/ArimaModel.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/models/garch/GarchModel.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"
#include "ag/util/Timer.hpp"

#include <cmath>
#include <functional>
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
    int num_runs;             // Number of benchmark runs
    std::string description;  // Description of benchmark
};

/**
 * @brief Run a single optimizer benchmark
 */
void runOptimizerBenchmark(const BenchmarkConfig& config, const models::ArimaGarchSpec& spec,
                           const std::vector<double>& data) {
    using namespace estimation;
    using namespace models;
    using namespace models::arima;
    using namespace models::garch;

    // Create likelihood evaluator
    ArimaGarchLikelihood likelihood(spec);

    // Initialize parameters
    auto [arima_init, garch_init] = initializeArimaGarchParameters(data.data(), data.size(), spec);

    // Create objective function
    auto objective = [&](const std::vector<double>& params) -> double {
        // Unpack parameters
        int arima_param_count = 1 + spec.arimaSpec.p + spec.arimaSpec.q;  // intercept + AR + MA

        ArimaParameters arima_params(spec.arimaSpec.p, spec.arimaSpec.q);
        GarchParameters garch_params(spec.garchSpec.p, spec.garchSpec.q);

        // Extract ARIMA parameters
        arima_params.intercept = params[0];
        for (std::size_t i = 0; i < spec.arimaSpec.p; ++i) {
            arima_params.ar_coef[i] = params[1 + i];
        }
        for (std::size_t i = 0; i < spec.arimaSpec.q; ++i) {
            arima_params.ma_coef[i] = params[1 + spec.arimaSpec.p + i];
        }

        // Extract GARCH parameters
        garch_params.omega = params[arima_param_count];
        for (std::size_t i = 0; i < spec.garchSpec.p; ++i) {
            garch_params.alpha_coef[i] = params[arima_param_count + 1 + i];
        }
        for (std::size_t i = 0; i < spec.garchSpec.q; ++i) {
            garch_params.beta_coef[i] = params[arima_param_count + 1 + spec.garchSpec.p + i];
        }

        return likelihood.computeNegativeLogLikelihood(data.data(), data.size(), arima_params,
                                                       garch_params);
    };

    // Create optimizer with reasonable settings
    NelderMeadOptimizer optimizer(1e-6, 1e-6, 500);

    // Prepare initial parameters vector
    std::vector<double> initial_params;
    initial_params.push_back(arima_init.intercept);
    for (std::size_t i = 0; i < spec.arimaSpec.p; ++i) {
        initial_params.push_back(arima_init.ar_coef[i]);
    }
    for (std::size_t i = 0; i < spec.arimaSpec.q; ++i) {
        initial_params.push_back(arima_init.ma_coef[i]);
    }
    initial_params.push_back(garch_init.omega);
    for (std::size_t i = 0; i < spec.garchSpec.p; ++i) {
        initial_params.push_back(garch_init.alpha_coef[i]);
    }
    for (std::size_t i = 0; i < spec.garchSpec.q; ++i) {
        initial_params.push_back(garch_init.beta_coef[i]);
    }

    // Warm-up run (not timed)
    [[maybe_unused]] auto warmup_result = optimizer.minimize(objective, initial_params);

    // Benchmark runs
    Timer timer;
    timer.start();

    int total_iterations = 0;
    int converged_count = 0;

    for (int run = 0; run < config.num_runs; ++run) {
        auto result = optimizer.minimize(objective, initial_params);
        total_iterations += result.iterations;
        if (result.converged) {
            converged_count++;
        }
    }

    double total_time = timer.stop();
    double avg_time = total_time / config.num_runs;
    double avg_iterations = static_cast<double>(total_iterations) / config.num_runs;

    // Print results
    fmt::print("  {:40s} | {:>8.2f} s | {:>7.1f} | {:>5d}/{:<3d}\n", config.description, avg_time,
               avg_iterations, converged_count, config.num_runs);
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
    fmt::print("  ARIMA-GARCH Optimizer Benchmark\n");
    fmt::print("=================================================================\n");
    fmt::print("\n");
    fmt::print("This benchmark measures optimizer performance on standard model\n");
    fmt::print("fitting tasks with synthetic time series data.\n");
    fmt::print("\n");

    // Generate benchmark data
    const std::size_t DATA_SIZE = 1000;  // Standard size for optimization
    const int NUM_RUNS = 5;              // Number of benchmark runs

    fmt::print("Generating synthetic data (n={})...\n", DATA_SIZE);
    std::vector<double> data = generateSyntheticData(DATA_SIZE, 42);
    fmt::print("Data generation complete.\n\n");

    // Benchmark header
    fmt::print("Running benchmarks ({} runs each, max 500 iterations):\n\n", NUM_RUNS);
    fmt::print("  {:40s} | {:>10s} | {:>7s} | {:>10s}\n", "Model Specification", "Avg Time",
               "Avg Iter", "Converged");
    fmt::print("  {:-<40s}-+-{:-<10s}-+-{:-<7s}-+-{:-<10s}\n", "", "", "", "");

    // Benchmark different model specifications
    std::vector<std::tuple<models::ArimaGarchSpec, std::string>> specs = {
        {models::ArimaGarchSpec(0, 0, 0, 1, 1), "ARIMA(0,0,0)-GARCH(1,1)"},
        {models::ArimaGarchSpec(1, 0, 0, 1, 1), "ARIMA(1,0,0)-GARCH(1,1)"},
        {models::ArimaGarchSpec(1, 0, 1, 1, 1), "ARIMA(1,0,1)-GARCH(1,1)"},
        {models::ArimaGarchSpec(2, 0, 1, 1, 1), "ARIMA(2,0,1)-GARCH(1,1)"},
    };

    for (const auto& [spec, description] : specs) {
        BenchmarkConfig config{DATA_SIZE, NUM_RUNS, description};
        runOptimizerBenchmark(config, spec, data);
    }

    fmt::print("\n");
    fmt::print("Benchmark complete.\n");
    fmt::print("\n");
    fmt::print("Performance notes:\n");
    fmt::print("  - Lower average time is better\n");
    fmt::print("  - Typical optimization time: 1-10 seconds depending on model complexity\n");
    fmt::print("  - Convergence rate should be high (>80%%)\n");
    fmt::print("  - More complex models require more iterations\n");
    fmt::print("\n");

    return 0;
}
