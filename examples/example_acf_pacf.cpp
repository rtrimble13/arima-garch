// Example demonstrating ACF/PACF computation
// Compile: cmake --build build && ./build/examples/example_acf_pacf

#include "ag/stats/ACF.hpp"
#include "ag/stats/PACF.hpp"

#include <random>
#include <vector>

#include <fmt/core.h>

int main() {
    fmt::print("=== ACF/PACF Example ===\n\n");

    // Generate white noise for demonstration
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(100);
    for (auto& val : data) {
        val = dist(gen);
    }

    // Compute ACF and PACF up to lag 10
    auto acf_values = ag::stats::acf(data, 10);
    auto pacf_values = ag::stats::pacf(data, 10);

    fmt::print("ACF values (lag 0-10):\n");
    for (std::size_t i = 0; i < acf_values.size(); ++i) {
        fmt::print("  Lag {:2d}: {:7.4f}\n", i, acf_values[i]);
    }

    fmt::print("\nPACF values (lag 1-10):\n");
    for (std::size_t i = 0; i < pacf_values.size(); ++i) {
        fmt::print("  Lag {:2d}: {:7.4f}\n", i + 1, pacf_values[i]);
    }

    fmt::print("\nNote: For white noise, ACF and PACF should be close to 0 at all lags.\n");
    fmt::print("For real time series analysis, use ACF/PACF plots to identify ARIMA orders.\n");

    return 0;
}
