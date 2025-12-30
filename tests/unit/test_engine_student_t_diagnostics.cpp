#include "ag/api/Engine.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"

#include <cmath>
#include <iostream>

#include "test_framework.hpp"

using ag::api::Engine;
using ag::models::ArimaGarchSpec;
using ag::models::composite::ArimaGarchParameters;
using ag::simulation::ArimaGarchSimulator;

// ============================================================================
// Test that Student-t innovation parameters are passed to diagnostics
// ============================================================================

TEST(engine_fit_student_t_uses_bootstrap) {
    // Generate synthetic data
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters true_params(spec);

    true_params.arima_params.intercept = 0.05;
    true_params.arima_params.ar_coef[0] = 0.6;
    true_params.arima_params.ma_coef[0] = 0.3;
    true_params.garch_params.omega = 0.01;
    true_params.garch_params.alpha_coef[0] = 0.1;
    true_params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchSimulator simulator(spec, true_params);
    auto sim_result = simulator.simulate(500, 42);

    // Fit using Engine WITH Student-t innovations (df = 6 < 30)
    Engine engine;
    auto fit_result = engine.fit(sim_result.returns, spec, true, true, 6.0);

    REQUIRE(fit_result.has_value());
    REQUIRE(fit_result.value().model != nullptr);
    REQUIRE(fit_result.value().summary.converged);
    REQUIRE(fit_result.value().summary.diagnostics.has_value());

    // Check that diagnostics contain Student-t information
    auto& diagnostics = fit_result.value().summary.diagnostics.value();

    // Verify bootstrap method is used (df < 30)
    REQUIRE(diagnostics.ljung_box_method == "bootstrap");
    REQUIRE(diagnostics.adf_method == "bootstrap");

    // Verify innovation distribution info is stored
    REQUIRE(diagnostics.innovation_distribution.has_value());
    REQUIRE(diagnostics.innovation_distribution.value() == "Student-t");
    REQUIRE(diagnostics.student_t_df.has_value());
    REQUIRE(std::abs(diagnostics.student_t_df.value() - 6.0) < 1e-10);
}

TEST(engine_fit_student_t_high_df_uses_asymptotic) {
    // Generate synthetic data
    ArimaGarchSpec spec(1, 0, 0, 1, 1);
    ArimaGarchParameters true_params(spec);

    true_params.arima_params.intercept = 0.02;
    true_params.arima_params.ar_coef[0] = 0.5;
    true_params.garch_params.omega = 0.02;
    true_params.garch_params.alpha_coef[0] = 0.15;
    true_params.garch_params.beta_coef[0] = 0.80;

    ArimaGarchSimulator simulator(spec, true_params);
    auto sim_result = simulator.simulate(300, 123);

    // Fit using Engine WITH Student-t but high df (df = 50 >= 30)
    Engine engine;
    auto fit_result = engine.fit(sim_result.returns, spec, true, true, 50.0);

    REQUIRE(fit_result.has_value());
    REQUIRE(fit_result.value().model != nullptr);
    REQUIRE(fit_result.value().summary.diagnostics.has_value());

    // Check that diagnostics contain Student-t information
    auto& diagnostics = fit_result.value().summary.diagnostics.value();

    // Verify asymptotic method is used (df >= 30)
    REQUIRE(diagnostics.ljung_box_method == "asymptotic");
    REQUIRE(diagnostics.adf_method == "asymptotic");

    // But innovation distribution info should still be stored
    REQUIRE(diagnostics.innovation_distribution.has_value());
    REQUIRE(diagnostics.innovation_distribution.value() == "Student-t");
    REQUIRE(diagnostics.student_t_df.has_value());
    REQUIRE(std::abs(diagnostics.student_t_df.value() - 50.0) < 1e-10);
}

TEST(engine_fit_normal_uses_asymptotic) {
    // Generate synthetic data
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters true_params(spec);

    true_params.arima_params.intercept = 0.05;
    true_params.arima_params.ar_coef[0] = 0.6;
    true_params.arima_params.ma_coef[0] = 0.3;
    true_params.garch_params.omega = 0.01;
    true_params.garch_params.alpha_coef[0] = 0.1;
    true_params.garch_params.beta_coef[0] = 0.85;

    ArimaGarchSimulator simulator(spec, true_params);
    auto sim_result = simulator.simulate(500, 42);

    // Fit using Engine WITHOUT Student-t (Normal innovations)
    Engine engine;
    auto fit_result = engine.fit(sim_result.returns, spec, true, false);

    REQUIRE(fit_result.has_value());
    REQUIRE(fit_result.value().model != nullptr);
    REQUIRE(fit_result.value().summary.diagnostics.has_value());

    // Check that diagnostics use asymptotic methods for Normal
    auto& diagnostics = fit_result.value().summary.diagnostics.value();

    // Verify asymptotic method is used for Normal innovations
    REQUIRE(diagnostics.ljung_box_method == "asymptotic");
    REQUIRE(diagnostics.adf_method == "asymptotic");

    // For Normal distribution, the implementation doesn't store distribution info
    // (it's only stored for non-Normal distributions like Student-t)
    // This is the existing design, so we don't check for it
}

TEST(engine_fit_summary_contains_innovation_distribution) {
    // Generate synthetic data
    ArimaGarchSpec spec(0, 0, 0, 1, 1);
    ArimaGarchParameters true_params(spec);

    true_params.arima_params.intercept = 0.0;
    true_params.garch_params.omega = 0.05;
    true_params.garch_params.alpha_coef[0] = 0.15;
    true_params.garch_params.beta_coef[0] = 0.80;

    ArimaGarchSimulator simulator(spec, true_params);
    auto sim_result = simulator.simulate(200, 456);

    // Fit with Student-t
    Engine engine;
    auto fit_result = engine.fit(sim_result.returns, spec, true, true, 8.0);

    REQUIRE(fit_result.has_value());

    // Check FitSummary contains innovation distribution info
    auto& summary = fit_result.value().summary;
    REQUIRE(summary.innovation_distribution == "Student-t");
    REQUIRE(std::abs(summary.student_t_df - 8.0) < 1e-10);

    // Also check diagnostics
    REQUIRE(summary.diagnostics.has_value());
    auto& diagnostics = summary.diagnostics.value();
    REQUIRE(diagnostics.innovation_distribution.has_value());
    REQUIRE(diagnostics.innovation_distribution.value() == "Student-t");
    REQUIRE(diagnostics.student_t_df.has_value());
    REQUIRE(std::abs(diagnostics.student_t_df.value() - 8.0) < 1e-10);
}

int main() {
    report_test_results("Engine Student-t Diagnostics Tests");
    return get_test_result();
}
