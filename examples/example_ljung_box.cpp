// Example demonstrating Ljung-Box test for residual diagnostics
// Compile: cmake --build build && ./build/examples/example_ljung_box

#include "ag/stats/LjungBox.hpp"

#include <random>
#include <vector>

#include <fmt/core.h>

int main() {
    fmt::print("=== Ljung-Box Test Example ===\n\n");

    // Example 1: Test white noise residuals (should NOT reject null hypothesis)
    fmt::print("Example 1: White Noise Residuals\n");
    fmt::print("---------------------------------\n");
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> white_noise(500);
    for (auto& val : white_noise) {
        val = dist(gen);
    }

    auto result1 = ag::stats::ljung_box_test(white_noise, 10);
    fmt::print("Sample size: {}\n", white_noise.size());
    fmt::print("Lags tested: {}\n", result1.lags);
    fmt::print("Q statistic: {:.4f}\n", result1.statistic);
    fmt::print("P-value:     {:.4f}\n", result1.p_value);
    fmt::print("Interpretation: ");
    if (result1.p_value > 0.05) {
        fmt::print("Fail to reject null hypothesis (residuals appear to be white noise)\n");
    } else {
        fmt::print("Reject null hypothesis (residuals show significant autocorrelation)\n");
    }

    // Example 2: Test autocorrelated residuals (should reject null hypothesis)
    fmt::print("\nExample 2: Autocorrelated Residuals (AR(1) with φ=0.8)\n");
    fmt::print("-------------------------------------------------------\n");

    std::vector<double> autocorrelated(500);
    autocorrelated[0] = dist(gen);
    const double phi = 0.8;
    for (std::size_t i = 1; i < autocorrelated.size(); ++i) {
        autocorrelated[i] = phi * autocorrelated[i - 1] + dist(gen);
    }

    auto result2 = ag::stats::ljung_box_test(autocorrelated, 10);
    fmt::print("Sample size: {}\n", autocorrelated.size());
    fmt::print("Lags tested: {}\n", result2.lags);
    fmt::print("Q statistic: {:.4f}\n", result2.statistic);
    fmt::print("P-value:     {:.4f}\n", result2.p_value);
    fmt::print("Interpretation: ");
    if (result2.p_value > 0.05) {
        fmt::print("Fail to reject null hypothesis (residuals appear to be white noise)\n");
    } else {
        fmt::print("Reject null hypothesis (residuals show significant autocorrelation)\n");
    }

    // Example 3: Custom degrees of freedom (accounting for estimated parameters)
    fmt::print("\nExample 3: Adjusted for Estimated Parameters\n");
    fmt::print("---------------------------------------------\n");

    // If we estimated 2 parameters (e.g., ARMA(1,1)), adjust DOF
    std::size_t lags = 10;
    std::size_t estimated_params = 2;
    std::size_t dof = lags - estimated_params;

    auto result3 = ag::stats::ljung_box_test(white_noise, lags, dof);
    fmt::print("Sample size: {}\n", white_noise.size());
    fmt::print("Lags tested: {}\n", result3.lags);
    fmt::print("Degrees of freedom: {} (adjusted for {} estimated parameters)\n", result3.dof,
               estimated_params);
    fmt::print("Q statistic: {:.4f}\n", result3.statistic);
    fmt::print("P-value:     {:.4f}\n", result3.p_value);

    fmt::print("\n=== Usage Notes ===\n");
    fmt::print("- Use Ljung-Box test to check if model residuals are white noise\n");
    fmt::print("- High p-value (> 0.05): Good model fit, residuals are uncorrelated\n");
    fmt::print("- Low p-value (< 0.05): Poor model fit, residuals show autocorrelation\n");
    fmt::print("- Adjust DOF by subtracting number of estimated parameters from number of lags\n");
    fmt::print("- Typical lag choices: 10, 20, or min(20, n/5) for sample size n\n");

    return 0;
}
