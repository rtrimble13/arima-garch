#include "ag/models/GarchSpec.hpp"
#include "ag/models/garch/GarchModel.hpp"
#include "ag/models/garch/GarchState.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "test_framework.hpp"

using ag::models::GarchSpec;
using ag::models::garch::GarchModel;
using ag::models::garch::GarchParameters;
using ag::models::garch::GarchState;

// ============================================================================
// GarchState Tests
// ============================================================================

// Test GarchState construction
TEST(garch_state_construction) {
    GarchState state(1, 1);
    REQUIRE(!state.isInitialized());
}

// Test GarchState initialization with sample variance
TEST(garch_state_init_sample_variance) {
    std::vector<double> residuals = {0.5, -0.3, 0.8, -0.2, 0.4, -0.6, 0.1};
    GarchState state(2, 1);

    state.initialize(residuals.data(), residuals.size());

    REQUIRE(state.isInitialized());
    REQUIRE(state.getVarianceHistory().size() == 2);
    REQUIRE(state.getSquaredResidualHistory().size() == 1);
    REQUIRE(state.getInitialVariance() > 0.0);
}

// Test GarchState initialization with unconditional variance
TEST(garch_state_init_unconditional_variance) {
    std::vector<double> residuals = {0.5, -0.3, 0.8};
    GarchState state(1, 1);

    double unconditional_var = 2.0;
    state.initialize(residuals.data(), residuals.size(), unconditional_var);

    REQUIRE(state.isInitialized());
    REQUIRE_APPROX(state.getInitialVariance(), unconditional_var, 1e-10);

    // Check that variance history is initialized to unconditional variance
    const auto& var_history = state.getVarianceHistory();
    REQUIRE_APPROX(var_history[0], unconditional_var, 1e-10);
}

// Test GarchState update
TEST(garch_state_update) {
    std::vector<double> residuals = {0.5, -0.3, 0.8};
    GarchState state(2, 1);
    state.initialize(residuals.data(), residuals.size());

    double init_var = state.getInitialVariance();

    // Update with new variance and squared residual
    state.update(1.5, 0.25);

    const auto& var_history = state.getVarianceHistory();
    const auto& sq_res_history = state.getSquaredResidualHistory();

    // Check that variance history was shifted and updated
    REQUIRE_APPROX(var_history[0], init_var, 1e-10);
    REQUIRE_APPROX(var_history[1], 1.5, 1e-10);

    // Check squared residual history
    REQUIRE_APPROX(sq_res_history[0], 0.25, 1e-10);

    // Update again
    state.update(1.8, 0.36);
    REQUIRE_APPROX(var_history[0], 1.5, 1e-10);
    REQUIRE_APPROX(var_history[1], 1.8, 1e-10);
    REQUIRE_APPROX(sq_res_history[0], 0.36, 1e-10);
}

// ============================================================================
// GarchParameters Tests
// ============================================================================

// Test parameter positivity check
TEST(garch_params_positivity) {
    GarchParameters params(1, 1);

    // Valid positive parameters
    params.omega = 0.1;
    params.alpha_coef[0] = 0.1;
    params.beta_coef[0] = 0.8;
    REQUIRE(params.isPositive());

    // Invalid: omega <= 0
    params.omega = 0.0;
    REQUIRE(!params.isPositive());

    params.omega = -0.1;
    REQUIRE(!params.isPositive());

    // Invalid: negative alpha
    params.omega = 0.1;
    params.alpha_coef[0] = -0.1;
    REQUIRE(!params.isPositive());

    // Invalid: negative beta
    params.alpha_coef[0] = 0.1;
    params.beta_coef[0] = -0.1;
    REQUIRE(!params.isPositive());
}

// Test parameter stationarity check
TEST(garch_params_stationarity) {
    GarchParameters params(1, 1);
    params.omega = 0.1;

    // Stationary: sum < 1
    params.alpha_coef[0] = 0.1;
    params.beta_coef[0] = 0.8;
    REQUIRE(params.isStationary());

    // Non-stationary: sum = 1
    params.alpha_coef[0] = 0.2;
    params.beta_coef[0] = 0.8;
    REQUIRE(!params.isStationary());

    // Non-stationary: sum > 1
    params.alpha_coef[0] = 0.6;
    params.beta_coef[0] = 0.5;
    REQUIRE(!params.isStationary());
}

// Test unconditional variance computation
TEST(garch_params_unconditional_variance) {
    GarchParameters params(1, 1);
    params.omega = 0.1;
    params.alpha_coef[0] = 0.15;
    params.beta_coef[0] = 0.75;

    // Check stationarity
    REQUIRE(params.isStationary());

    // Unconditional variance: σ² = ω / (1 - α - β) = 0.1 / (1 - 0.15 - 0.75) = 0.1 / 0.1 = 1.0
    double expected_var = 0.1 / (1.0 - 0.15 - 0.75);
    REQUIRE_APPROX(params.unconditionalVariance(), expected_var, 1e-10);

    // Non-stationary case
    params.alpha_coef[0] = 0.5;
    params.beta_coef[0] = 0.5;
    REQUIRE(!params.isStationary());
    REQUIRE_APPROX(params.unconditionalVariance(), 0.0, 1e-10);
}

// ============================================================================
// GarchModel Tests - GARCH(1,1)
// ============================================================================

// Test GARCH(1,1) with known parameters - positivity
TEST(garch_model_11_positivity) {
    GarchSpec spec(1, 1);
    GarchModel model(spec);

    // Create residuals
    std::vector<double> residuals = {0.5, -0.3, 0.8, -0.2, 0.4, -0.6, 0.1, 0.3};

    // Set up GARCH(1,1) parameters: h_t = 0.1 + 0.1*ε²_{t-1} + 0.8*h_{t-1}
    GarchParameters params(1, 1);
    params.omega = 0.1;
    params.alpha_coef[0] = 0.1;
    params.beta_coef[0] = 0.8;

    auto variances = model.computeConditionalVariances(residuals.data(), residuals.size(), params);

    REQUIRE(variances.size() == residuals.size());

    // All variances must be positive
    for (double h_t : variances) {
        REQUIRE(h_t > 0.0);
    }
}

// Test GARCH(1,1) with manual verification
TEST(garch_model_11_manual_verification) {
    GarchSpec spec(1, 1);
    GarchModel model(spec);

    std::vector<double> residuals = {1.0, 2.0, 1.5};

    GarchParameters params(1, 1);
    params.omega = 0.5;
    params.alpha_coef[0] = 0.2;
    params.beta_coef[0] = 0.6;

    // Unconditional variance: σ² = 0.5 / (1 - 0.2 - 0.6) = 0.5 / 0.2 = 2.5
    double h_0 = 2.5;

    auto variances = model.computeConditionalVariances(residuals.data(), residuals.size(), params);

    REQUIRE(variances.size() == 3);

    // Manual computation:
    // h_1 = 0.5 + 0.2 * 0² + 0.6 * 2.5 = 0.5 + 0 + 1.5 = 2.0
    double h_1_expected = 0.5 + 0.2 * 0.0 + 0.6 * 2.5;
    REQUIRE_APPROX(variances[0], h_1_expected, 1e-10);

    // h_2 = 0.5 + 0.2 * 1.0² + 0.6 * 2.0 = 0.5 + 0.2 + 1.2 = 1.9
    double h_2_expected = 0.5 + 0.2 * (1.0 * 1.0) + 0.6 * variances[0];
    REQUIRE_APPROX(variances[1], h_2_expected, 1e-10);

    // h_3 = 0.5 + 0.2 * 2.0² + 0.6 * 1.9 = 0.5 + 0.8 + 1.14 = 2.44
    double h_3_expected = 0.5 + 0.2 * (2.0 * 2.0) + 0.6 * variances[1];
    REQUIRE_APPROX(variances[2], h_3_expected, 1e-10);
}

// Test GARCH(1,1) stability over long series
TEST(garch_model_11_stability) {
    GarchSpec spec(1, 1);
    GarchModel model(spec);

    // Generate a long series of residuals
    std::vector<double> residuals(1000);
    std::mt19937 gen(42);
    std::normal_distribution<> d(0.0, 1.0);
    for (auto& eps : residuals) {
        eps = d(gen);
    }

    GarchParameters params(1, 1);
    params.omega = 0.05;
    params.alpha_coef[0] = 0.1;
    params.beta_coef[0] = 0.85;

    auto variances = model.computeConditionalVariances(residuals.data(), residuals.size(), params);

    REQUIRE(variances.size() == residuals.size());

    // Check all variances are positive
    for (double h_t : variances) {
        REQUIRE(h_t > 0.0);
    }

    // Check that variances remain bounded (stability)
    double max_variance = *std::max_element(variances.begin(), variances.end());
    REQUIRE(max_variance < 100.0);  // Should not explode
}

// ============================================================================
// GarchModel Tests - GARCH(2,2)
// ============================================================================

// Test GARCH(2,2) with multiple lags
TEST(garch_model_22) {
    GarchSpec spec(2, 2);
    GarchModel model(spec);

    std::vector<double> residuals = {0.5, -0.8, 1.2, -0.3, 0.7, -1.0, 0.4};

    GarchParameters params(2, 2);
    params.omega = 0.1;
    params.alpha_coef[0] = 0.08;
    params.alpha_coef[1] = 0.06;
    params.beta_coef[0] = 0.7;
    params.beta_coef[1] = 0.1;

    // Check stationarity: 0.08 + 0.06 + 0.7 + 0.1 = 0.94 < 1 ✓
    REQUIRE(params.isStationary());

    auto variances = model.computeConditionalVariances(residuals.data(), residuals.size(), params);

    REQUIRE(variances.size() == residuals.size());

    // All variances must be positive
    for (double h_t : variances) {
        REQUIRE(h_t > 0.0);
    }
}

// ============================================================================
// GarchModel Tests - Edge Cases
// ============================================================================

// Test with very small residuals
TEST(garch_model_small_residuals) {
    GarchSpec spec(1, 1);
    GarchModel model(spec);

    std::vector<double> residuals = {0.001, -0.002, 0.0015, -0.0008};

    GarchParameters params(1, 1);
    params.omega = 0.01;
    params.alpha_coef[0] = 0.1;
    params.beta_coef[0] = 0.8;

    auto variances = model.computeConditionalVariances(residuals.data(), residuals.size(), params);

    REQUIRE(variances.size() == residuals.size());

    for (double h_t : variances) {
        REQUIRE(h_t > 0.0);
    }
}

// Test with large residuals
TEST(garch_model_large_residuals) {
    GarchSpec spec(1, 1);
    GarchModel model(spec);

    std::vector<double> residuals = {5.0, -8.0, 12.0, -3.0};

    GarchParameters params(1, 1);
    params.omega = 1.0;
    params.alpha_coef[0] = 0.15;
    params.beta_coef[0] = 0.7;

    auto variances = model.computeConditionalVariances(residuals.data(), residuals.size(), params);

    REQUIRE(variances.size() == residuals.size());

    for (double h_t : variances) {
        REQUIRE(h_t > 0.0);
    }
}

// Test variance convergence to unconditional variance (under stationary conditions)
TEST(garch_model_convergence) {
    GarchSpec spec(1, 1);
    GarchModel model(spec);

    // Use small constant residuals (close to zero)
    std::vector<double> residuals(500, 0.01);

    GarchParameters params(1, 1);
    params.omega = 0.1;
    params.alpha_coef[0] = 0.1;
    params.beta_coef[0] = 0.8;

    double unconditional_var = params.unconditionalVariance();

    auto variances = model.computeConditionalVariances(residuals.data(), residuals.size(), params);

    // After many iterations with small shocks, variance should approach unconditional variance
    // Check last few variances
    for (size_t i = variances.size() - 10; i < variances.size(); ++i) {
        // Should be close to unconditional variance (with some tolerance for small residuals)
        REQUIRE(std::abs(variances[i] - unconditional_var) < 0.5);
    }
}

// ============================================================================
// GarchModel Tests - Parameter Validation
// ============================================================================

// Test that negative omega is rejected
TEST(garch_model_invalid_omega) {
    GarchSpec spec(1, 1);
    GarchModel model(spec);

    std::vector<double> residuals = {0.5, -0.3, 0.8};

    GarchParameters params(1, 1);
    params.omega = -0.1;  // Invalid
    params.alpha_coef[0] = 0.1;
    params.beta_coef[0] = 0.8;

    bool caught_exception = false;
    try {
        [[maybe_unused]] auto variances =
            model.computeConditionalVariances(residuals.data(), residuals.size(), params);
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }

    REQUIRE(caught_exception);
}

// Test that negative alpha is rejected
TEST(garch_model_invalid_alpha) {
    GarchSpec spec(1, 1);
    GarchModel model(spec);

    std::vector<double> residuals = {0.5, -0.3, 0.8};

    GarchParameters params(1, 1);
    params.omega = 0.1;
    params.alpha_coef[0] = -0.1;  // Invalid
    params.beta_coef[0] = 0.8;

    bool caught_exception = false;
    try {
        [[maybe_unused]] auto variances =
            model.computeConditionalVariances(residuals.data(), residuals.size(), params);
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }

    REQUIRE(caught_exception);
}

// Test that negative beta is rejected
TEST(garch_model_invalid_beta) {
    GarchSpec spec(1, 1);
    GarchModel model(spec);

    std::vector<double> residuals = {0.5, -0.3, 0.8};

    GarchParameters params(1, 1);
    params.omega = 0.1;
    params.alpha_coef[0] = 0.1;
    params.beta_coef[0] = -0.8;  // Invalid

    bool caught_exception = false;
    try {
        [[maybe_unused]] auto variances =
            model.computeConditionalVariances(residuals.data(), residuals.size(), params);
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }

    REQUIRE(caught_exception);
}

int main() {
    report_test_results("GARCH Models");
    return get_test_result();
}
