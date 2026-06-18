#include "ag/api/Engine.hpp"
#include "ag/diagnostics/Residuals.hpp"
#include "ag/forecasting/Forecaster.hpp"
#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"
#include "ag/util/Differencing.hpp"

#include <cmath>
#include <random>
#include <vector>

#include "test_framework.hpp"

using ag::models::ArimaGarchSpec;
using ag::models::composite::ArimaGarchModel;
using ag::models::composite::ArimaGarchParameters;
using ag::util::StreamingDifferencer;

// ============================================================================
// StreamingDifferencer unit tests
// ============================================================================

// d == 0 is the identity in both directions.
TEST(streaming_differencer_order_zero_identity) {
    StreamingDifferencer diff(0);
    double w = 0.0;
    REQUIRE(diff.difference(3.5, w));
    REQUIRE_APPROX(w, 3.5, 1e-12);
    REQUIRE_APPROX(diff.integrate(2.0), 2.0, 1e-12);
}

// First difference: priming consumes the first observation, then Δy is emitted.
TEST(streaming_differencer_first_difference) {
    StreamingDifferencer diff(1);
    double w = 0.0;

    REQUIRE(!diff.difference(10.0, w));  // priming
    REQUIRE(diff.difference(11.0, w));
    REQUIRE_APPROX(w, 1.0, 1e-12);
    REQUIRE(diff.difference(13.0, w));
    REQUIRE_APPROX(w, 2.0, 1e-12);
}

// Second differencing: two observations prime the pipeline, then Δ²y is
// emitted. For {1,2,4,8,16} the second differences are 1, 2, 4.
TEST(streaming_differencer_second_difference) {
    StreamingDifferencer diff(2);
    std::vector<double> levels = {1.0, 2.0, 4.0, 8.0, 16.0};
    std::vector<double> emitted;
    for (double level : levels) {
        double w = 0.0;
        if (diff.difference(level, w)) {
            emitted.push_back(w);
        }
    }
    REQUIRE(emitted.size() == 3);
    REQUIRE_APPROX(emitted[0], 1.0, 1e-12);
    REQUIRE_APPROX(emitted[1], 2.0, 1e-12);
    REQUIRE_APPROX(emitted[2], 4.0, 1e-12);
}

// Integrating from zero anchors is the d-fold cumulative sum.
TEST(streaming_differencer_integrate_is_cumulative_sum) {
    StreamingDifferencer inv1(1);
    REQUIRE_APPROX(inv1.integrate(2.0), 2.0, 1e-12);
    REQUIRE_APPROX(inv1.integrate(3.0), 5.0, 1e-12);
    REQUIRE_APPROX(inv1.integrate(-1.0), 4.0, 1e-12);

    // Second-order: integrate twice-cumulatively. Feeding 1,1,1 yields the
    // partial sums of partial sums: 1, 3, 6.
    StreamingDifferencer inv2(2);
    REQUIRE_APPROX(inv2.integrate(1.0), 1.0, 1e-12);
    REQUIRE_APPROX(inv2.integrate(1.0), 3.0, 1e-12);
    REQUIRE_APPROX(inv2.integrate(1.0), 6.0, 1e-12);
}

// ============================================================================
// d-aware model + forecaster: deterministic level forecasts
// ============================================================================

// ARIMA(1,1,0)+GARCH(1,1): hand-computed one-, two-, three-step level
// forecasts. The mean path is independent of the GARCH parameters.
TEST(arima_d1_level_forecast_matches_hand_calc) {
    ArimaGarchSpec spec(1, 1, 0, 1, 1);
    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef[0] = 0.5;
    params.garch_params.omega = 0.01;
    params.garch_params.alpha_coef[0] = 0.1;
    params.garch_params.beta_coef[0] = 0.8;

    ArimaGarchModel model(spec, params);
    std::vector<double> levels = {10.0, 11.0, 13.0, 16.0, 20.0};
    for (double y : levels) {
        model.update(y);
    }

    ag::forecasting::Forecaster forecaster(model);
    auto fc = forecaster.forecast(3);

    // w_T = 20 - 16 = 4
    // mu_w1 = 0.1 + 0.5*4 = 2.1   -> level1 = 20 + 2.1 = 22.1
    // mu_w2 = 0.1 + 0.5*2.1 = 1.15 -> level2 = 22.1 + 1.15 = 23.25
    // mu_w3 = 0.1 + 0.5*1.15 = 0.675 -> level3 = 23.25 + 0.675 = 23.925
    REQUIRE_APPROX(fc.mean_forecasts[0], 22.1, 1e-9);
    REQUIRE_APPROX(fc.mean_forecasts[1], 23.25, 1e-9);
    REQUIRE_APPROX(fc.mean_forecasts[2], 23.925, 1e-9);

    // Level forecasts must be near the last level, not near the differenced
    // scale (~0): a regression guard against the un-integrated bug.
    REQUIRE(fc.mean_forecasts[0] > 15.0);
}

// ============================================================================
// d-aware diagnostics: residual count matches the likelihood (size - d)
// ============================================================================

TEST(diagnostics_residual_count_matches_size_minus_d) {
    for (int d = 0; d <= 2; ++d) {
        ArimaGarchSpec spec(1, d, 0, 1, 1);
        ArimaGarchParameters params(spec);
        params.arima_params.intercept = 0.0;
        params.arima_params.ar_coef[0] = 0.3;
        params.garch_params.omega = 0.05;
        params.garch_params.alpha_coef[0] = 0.1;
        params.garch_params.beta_coef[0] = 0.8;

        std::vector<double> data(40);
        double level = 0.0;
        for (std::size_t i = 0; i < data.size(); ++i) {
            level += 0.1 * static_cast<double>(i % 3) - 0.05;
            data[i] = level + 0.01 * static_cast<double>(i);
        }

        auto residuals = ag::diagnostics::computeResiduals(spec, params, data.data(), data.size());
        REQUIRE(residuals.eps_t.size() == data.size() - static_cast<std::size_t>(d));
    }
}

// ============================================================================
// d-aware simulation: the d=1 path is the integral of the d=0 path
// ============================================================================

TEST(simulate_d1_is_integral_of_d0) {
    ArimaGarchSpec spec_d0(1, 0, 0, 1, 1);
    ArimaGarchSpec spec_d1(1, 1, 0, 1, 1);

    ArimaGarchParameters params0(spec_d0);
    params0.arima_params.intercept = 0.02;
    params0.arima_params.ar_coef[0] = 0.4;
    params0.garch_params.omega = 0.01;
    params0.garch_params.alpha_coef[0] = 0.1;
    params0.garch_params.beta_coef[0] = 0.85;

    ArimaGarchParameters params1(spec_d1);
    params1.arima_params = params0.arima_params;
    params1.garch_params = params0.garch_params;

    const int length = 50;
    const unsigned int seed = 7;

    ag::simulation::ArimaGarchSimulator sim0(spec_d0, params0);
    ag::simulation::ArimaGarchSimulator sim1(spec_d1, params1);
    auto r0 = sim0.simulate(length, seed);
    auto r1 = sim1.simulate(length, seed);

    // Identical params/seed => identical differenced draws; the d=1 returns are
    // the cumulative sum of the d=0 returns.
    REQUIRE_APPROX(r1.returns[0], r0.returns[0], 1e-9);
    for (int t = 1; t < length; ++t) {
        REQUIRE_APPROX(r1.returns[t] - r1.returns[t - 1], r0.returns[t], 1e-9);
    }
}

// End-to-end on the CLI default path (select defaults to --max-d 1): fit an
// ARIMA(1,1,0)+GARCH(1,1) with drift and confirm the forecast is on the level
// scale (near the last observed level), not collapsed to the differenced-scale
// intercept (~0) as the un-integrated bug produced.
TEST(engine_fit_forecast_d1_is_level_scale) {
    std::mt19937 gen(11);
    std::normal_distribution<double> nd(0.0, 1.0);

    std::vector<double> data(150);
    double w_prev = 0.0;
    double level = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        double w = 0.3 * w_prev + nd(gen) + 2.0;  // AR(1) differences with drift
        w_prev = w - 2.0;
        level += w;
        data[i] = level;
    }

    REQUIRE(data.back() > 100.0);  // strongly trending, far from zero

    ag::api::Engine engine;
    ArimaGarchSpec spec(1, 1, 0, 1, 1);
    auto fit = engine.fit(data, spec, false);
    REQUIRE(fit.has_value());
    REQUIRE(fit.value().summary.converged);

    auto fc = engine.forecast(*fit.value().model, 5);
    REQUIRE(fc.has_value());
    REQUIRE(fc.value().mean_forecasts.size() == 5);

    // One-step forecast continues from the last level (differenced mean is
    // small); the old bug would land near 0, far below data.back().
    double f0 = fc.value().mean_forecasts[0];
    REQUIRE(std::isfinite(f0));
    REQUIRE(std::abs(f0 - data.back()) < 25.0);
}

int main() {
    report_test_results("ARIMA Differencing (d > 0)");
    return get_test_result();
}
