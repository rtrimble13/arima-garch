/**
 * @file example_constraints.cpp
 * @brief Example demonstrating ARIMA-GARCH parameter transformation
 *
 * This example shows how to use ArimaGarchTransform to convert between
 * unconstrained optimizer parameters and constrained GARCH parameters.
 */

#include "ag/estimation/Constraints.hpp"

#include <iostream>
#include <random>
#include <vector>

using ag::estimation::ArimaGarchTransform;
using ag::estimation::ParameterVector;

void print_params(const std::string& label, const ParameterVector& params) {
    std::cout << label << ": [";
    for (std::size_t i = 0; i < params.size(); ++i) {
        std::cout << params[i];
        if (i < params.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

int main() {
    std::cout << "=== ARIMA-GARCH Parameter Transform Example ===\n\n";

    // Example 1: Transform unconstrained parameters to constrained GARCH(1,1)
    std::cout << "Example 1: GARCH(1,1) transformation\n";
    std::cout << "-------------------------------------\n";

    ParameterVector theta(3, 0.0);
    theta[0] = -4.6;  // omega (unconstrained)
    theta[1] = -2.3;  // alpha (unconstrained)
    theta[2] = 2.1;   // beta (unconstrained)

    print_params("Unconstrained theta", theta);

    ParameterVector params = ArimaGarchTransform::toConstrained(theta, 1, 1);
    print_params("Constrained GARCH params", params);

    std::cout << "\nConstraint verification:\n";
    std::cout << "  omega > 0: " << (params[0] > 0 ? "✓" : "✗") << " (omega = " << params[0]
              << ")\n";
    std::cout << "  alpha >= 0: " << (params[1] >= 0 ? "✓" : "✗") << " (alpha = " << params[1]
              << ")\n";
    std::cout << "  beta >= 0: " << (params[2] >= 0 ? "✓" : "✗") << " (beta = " << params[2]
              << ")\n";
    std::cout << "  sum < 1: " << (params[1] + params[2] < 1.0 ? "✓" : "✗")
              << " (sum = " << params[1] + params[2] << ")\n";
    std::cout << "  All constraints: "
              << (ArimaGarchTransform::validateConstraints(params, 1, 1) ? "✓" : "✗") << "\n";

    // Example 2: Inverse transformation
    std::cout << "\n\nExample 2: Inverse transformation\n";
    std::cout << "-----------------------------------\n";

    ParameterVector valid_params(3, 0.0);
    valid_params[0] = 0.01;  // omega
    valid_params[1] = 0.1;   // alpha
    valid_params[2] = 0.85;  // beta

    print_params("Input constrained params", valid_params);

    ParameterVector recovered_theta = ArimaGarchTransform::toUnconstrained(valid_params, 1, 1);
    print_params("Recovered theta", recovered_theta);

    // Transform back to verify round-trip
    ParameterVector round_trip_params = ArimaGarchTransform::toConstrained(recovered_theta, 1, 1);
    print_params("Round-trip params", round_trip_params);

    std::cout << "\nRound-trip error:\n";
    std::cout << "  omega: " << std::abs(valid_params[0] - round_trip_params[0]) << "\n";
    std::cout << "  alpha: " << std::abs(valid_params[1] - round_trip_params[1]) << "\n";
    std::cout << "  beta: " << std::abs(valid_params[2] - round_trip_params[2]) << "\n";

    // Example 3: Random theta values
    std::cout << "\n\nExample 3: Random unconstrained parameters\n";
    std::cout << "-------------------------------------------\n";

    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<> dis(-5.0, 5.0);

    std::cout << "Testing 10 random theta vectors:\n";
    for (int i = 0; i < 10; ++i) {
        ParameterVector random_theta(3, 0.0);
        random_theta[0] = dis(gen);
        random_theta[1] = dis(gen);
        random_theta[2] = dis(gen);

        ParameterVector constrained = ArimaGarchTransform::toConstrained(random_theta, 1, 1);
        bool valid = ArimaGarchTransform::validateConstraints(constrained, 1, 1);

        std::cout << "  Trial " << (i + 1) << ": sum = " << (constrained[1] + constrained[2])
                  << " - " << (valid ? "VALID" : "INVALID") << "\n";
    }

    // Example 4: GARCH(2,2) model
    std::cout << "\n\nExample 4: GARCH(2,2) transformation\n";
    std::cout << "------------------------------------\n";

    ParameterVector theta_22(5, 0.0);
    theta_22[0] = -3.0;  // omega
    theta_22[1] = 0.5;   // alpha1
    theta_22[2] = -0.5;  // alpha2
    theta_22[3] = 1.0;   // beta1
    theta_22[4] = 0.8;   // beta2

    print_params("GARCH(2,2) theta", theta_22);

    ParameterVector params_22 = ArimaGarchTransform::toConstrained(theta_22, 2, 2);
    print_params("GARCH(2,2) constrained", params_22);

    std::cout << "\nGARCH(2,2) constraints:\n";
    std::cout << "  omega: " << params_22[0] << "\n";
    std::cout << "  alphas: [" << params_22[1] << ", " << params_22[2] << "]\n";
    std::cout << "  betas: [" << params_22[3] << ", " << params_22[4] << "]\n";
    std::cout << "  sum(alphas) + sum(betas): "
              << (params_22[1] + params_22[2] + params_22[3] + params_22[4]) << "\n";
    std::cout << "  Valid: "
              << (ArimaGarchTransform::validateConstraints(params_22, 2, 2) ? "✓" : "✗") << "\n";

    std::cout << "\n=== All examples completed successfully ===\n";

    return 0;
}
