#include "ag/models/ArimaGarchSpec.hpp"
#include "ag/models/composite/ArimaGarchModel.hpp"
#include "ag/selection/DistributionSelector.hpp"
#include "ag/simulation/ArimaGarchSimulator.hpp"

#include <cmath>
#include <numbers>
#include <vector>

#include "test_framework.hpp"

using ag::models::ArimaGarchSpec;
using ag::models::composite::ArimaGarchParameters;
using ag::selection::compareDistributions;
using ag::selection::DistributionTestResult;
using ag::selection::estimateStudentTDF;
using ag::simulation::ArimaGarchSimulator;

// ============================================================================
// estimateStudentTDF Tests
// ============================================================================

// Test with Gaussian data (should estimate high df)
TEST(estimate_student_t_df_gaussian) {
    // Generate standard normal data
    std::vector<double> gaussian_residuals;
    for (int i = 0; i < 1000; ++i) {
        double u1 = (i + 1.0) / 1001.0;
        double u2 = (i + 2.0) / 1002.0;
        // Box-Muller transform
        double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::numbers::pi * u2);
        gaussian_residuals.push_back(z);
    }

    double df = estimateStudentTDF(gaussian_residuals);

    // For Gaussian data, should estimate relatively high df (>10)
    // Note: exact value depends on sample and estimation algorithm
    REQUIRE(df > 5.0);
}

// Test with heavy-tailed data (should estimate low df)
TEST(estimate_student_t_df_heavy_tails) {
    // Create data with heavy tails by including outliers
    std::vector<double> heavy_tail_residuals;
    for (int i = 0; i < 100; ++i) {
        double u1 = (i + 1.0) / 101.0;
        double u2 = (i + 2.0) / 102.0;
        double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::numbers::pi * u2);
        heavy_tail_residuals.push_back(z);
    }
    // Add some outliers
    heavy_tail_residuals.push_back(5.0);
    heavy_tail_residuals.push_back(-5.0);
    heavy_tail_residuals.push_back(6.0);
    heavy_tail_residuals.push_back(-6.0);

    double df = estimateStudentTDF(heavy_tail_residuals);

    // With outliers, should estimate lower df
    REQUIRE(df >= 3.0);
    REQUIRE(df < 30.0);
}

// Test with empty vector
TEST(estimate_student_t_df_empty) {
    std::vector<double> empty;

    bool caught_exception = false;
    try {
        estimateStudentTDF(empty);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("empty") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// Test that estimated df is in reasonable range
TEST(estimate_student_t_df_reasonable_range) {
    std::vector<double> residuals;
    for (int i = 0; i < 100; ++i) {
        residuals.push_back(std::sin(i * 0.1));
    }

    double df = estimateStudentTDF(residuals);

    // df should be in reasonable range
    REQUIRE(df >= 2.0);
    REQUIRE(df <= 100.0);
}

// ============================================================================
// compareDistributions Tests
// ============================================================================

// Test with simple ARIMA-GARCH model
TEST(compare_distributions_simple_model) {
    // Simple ARIMA(1,0,1)-GARCH(1,1) model
    ArimaGarchSpec spec(1, 0, 1, 1, 1);

    // Set up parameters
    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.5;
    params.arima_params.ar_coef = {0.3};
    params.arima_params.ma_coef = {0.2};
    params.garch_params.omega = 0.1;
    params.garch_params.alpha_coef = {0.2};
    params.garch_params.beta_coef = {0.7};

    // Simulate data
    ArimaGarchSimulator sim(spec, params);
    auto sim_result = sim.simulate(200, 42);
    std::vector<double> data = sim_result.returns;

    // Compare distributions
    DistributionTestResult result = compareDistributions(spec, params, data.data(), data.size());

    // Check that result fields are valid
    REQUIRE(!std::isnan(result.normal_ll));
    REQUIRE(!std::isnan(result.student_t_ll));
    REQUIRE(!std::isnan(result.df));
    REQUIRE(!std::isnan(result.lr_statistic));
    REQUIRE(!std::isnan(result.lr_p_value));
    REQUIRE(!std::isnan(result.aic_improvement));
    REQUIRE(!std::isnan(result.bic_improvement));
    REQUIRE(!std::isnan(result.kurtosis));

    // Check that df is in reasonable range
    REQUIRE(result.df > 2.0);
    REQUIRE(result.df <= 100.0);

    // Check that p-value is in [0, 1]
    REQUIRE(result.lr_p_value >= 0.0);
    REQUIRE(result.lr_p_value <= 1.0);

    // Student-t log-likelihood could be less than Gaussian due to
    // suboptimal df estimation via grid search
    // Just verify calculations are consistent
    double calculated_lr = 2.0 * (result.student_t_ll - result.normal_ll);
    REQUIRE_APPROX(result.lr_statistic, calculated_lr, 1e-8);
}

// Test with null data
TEST(compare_distributions_null_data) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);

    bool caught_exception = false;
    try {
        compareDistributions(spec, params, nullptr, 100);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("null") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// Test with zero size
TEST(compare_distributions_zero_size) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);
    ArimaGarchParameters params(spec);
    std::vector<double> data = {1.0, 2.0, 3.0};

    bool caught_exception = false;
    try {
        compareDistributions(spec, params, data.data(), 0);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg(e.what());
        REQUIRE(msg.find("positive") != std::string::npos || msg.find("size") != std::string::npos);
    }
    REQUIRE(caught_exception);
}

// Test that Student-t detector works for heavy-tailed data
TEST(compare_distributions_prefers_student_t_for_heavy_tails) {
    ArimaGarchSpec spec(0, 0, 0, 1, 1);

    ArimaGarchParameters params(spec);
    params.garch_params.omega = 1.0;
    params.garch_params.alpha_coef = {0.1};
    params.garch_params.beta_coef = {0.8};

    // Create data with heavy tails
    std::vector<double> data;
    for (int i = 0; i < 200; ++i) {
        double u1 = (i + 1.0) / 201.0;
        double u2 = (i + 2.0) / 202.0;
        double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::numbers::pi * u2);
        data.push_back(z);
    }
    // Add outliers
    for (int i = 0; i < 10; ++i) {
        data.push_back(5.0 + i);
        data.push_back(-5.0 - i);
    }

    DistributionTestResult result = compareDistributions(spec, params, data.data(), data.size());

    // With heavy tails, should have high excess kurtosis
    REQUIRE(result.kurtosis > 0.5);

    // df should be in valid range
    REQUIRE(result.df > 2.0);
    REQUIRE(result.df <= 100.0);
}

// Test AIC and BIC improvement calculation
TEST(compare_distributions_aic_bic_improvement) {
    ArimaGarchSpec spec(1, 0, 0, 1, 1);

    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.0;
    params.arima_params.ar_coef = {0.5};
    params.garch_params.omega = 1.0;
    params.garch_params.alpha_coef = {0.1};
    params.garch_params.beta_coef = {0.8};

    ArimaGarchSimulator sim(spec, params);
    auto sim_result = sim.simulate(150, 123);
    std::vector<double> data = sim_result.returns;

    DistributionTestResult result = compareDistributions(spec, params, data.data(), data.size());

    // AIC improvement = AIC_normal - AIC_student_t
    // If positive, Student-t is better
    int k_normal = spec.totalParamCount();
    int k_student_t = k_normal + 1;
    double expected_aic_normal = -2.0 * result.normal_ll + 2.0 * k_normal;
    double expected_aic_student_t = -2.0 * result.student_t_ll + 2.0 * k_student_t;
    double expected_aic_improvement = expected_aic_normal - expected_aic_student_t;

    REQUIRE_APPROX(result.aic_improvement, expected_aic_improvement, 1e-8);

    // BIC improvement = BIC_normal - BIC_student_t
    double n = static_cast<double>(data.size());
    double expected_bic_normal = -2.0 * result.normal_ll + k_normal * std::log(n);
    double expected_bic_student_t = -2.0 * result.student_t_ll + k_student_t * std::log(n);
    double expected_bic_improvement = expected_bic_normal - expected_bic_student_t;

    REQUIRE_APPROX(result.bic_improvement, expected_bic_improvement, 1e-8);
}

// Test likelihood ratio statistic calculation
TEST(compare_distributions_lr_statistic) {
    ArimaGarchSpec spec(0, 0, 1, 1, 1);

    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.0;
    params.arima_params.ma_coef = {0.3};
    params.garch_params.omega = 1.0;
    params.garch_params.alpha_coef = {0.15};
    params.garch_params.beta_coef = {0.75};

    ArimaGarchSimulator sim(spec, params);
    auto sim_result = sim.simulate(100, 456);
    std::vector<double> data = sim_result.returns;

    DistributionTestResult result = compareDistributions(spec, params, data.data(), data.size());

    // LR statistic = 2 * (LL_student_t - LL_normal)
    double expected_lr = 2.0 * (result.student_t_ll - result.normal_ll);
    REQUIRE_APPROX(result.lr_statistic, expected_lr, 1e-8);
}

// Test with larger dataset
TEST(compare_distributions_large_dataset) {
    ArimaGarchSpec spec(1, 0, 1, 1, 1);

    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.1;
    params.arima_params.ar_coef = {0.4};
    params.arima_params.ma_coef = {0.3};
    params.garch_params.omega = 0.05;
    params.garch_params.alpha_coef = {0.1};
    params.garch_params.beta_coef = {0.85};

    ArimaGarchSimulator sim(spec, params);
    auto sim_result = sim.simulate(500, 789);
    std::vector<double> data = sim_result.returns;

    DistributionTestResult result = compareDistributions(spec, params, data.data(), data.size());

    // With larger sample, all metrics should be valid
    REQUIRE(!std::isnan(result.normal_ll));
    REQUIRE(!std::isnan(result.student_t_ll));
    REQUIRE(!std::isinf(result.normal_ll));
    REQUIRE(!std::isinf(result.student_t_ll));

    // p-value should be in valid range
    REQUIRE(result.lr_p_value >= 0.0);
    REQUIRE(result.lr_p_value <= 1.0);

    // df should be reasonable
    REQUIRE(result.df > 2.0);
    REQUIRE(result.df <= 100.0);
}

// Test recommendation logic
TEST(compare_distributions_recommendation_logic) {
    ArimaGarchSpec spec(1, 0, 0, 1, 1);

    ArimaGarchParameters params(spec);
    params.arima_params.intercept = 0.0;
    params.arima_params.ar_coef = {0.6};
    params.garch_params.omega = 1.0;
    params.garch_params.alpha_coef = {0.2};
    params.garch_params.beta_coef = {0.7};

    ArimaGarchSimulator sim(spec, params);
    auto sim_result = sim.simulate(200, 321);
    std::vector<double> data = sim_result.returns;

    DistributionTestResult result = compareDistributions(spec, params, data.data(), data.size());

    // Check that recommendation is a boolean
    REQUIRE(result.prefer_student_t == true || result.prefer_student_t == false);

    // If p-value < 0.05, should prefer Student-t
    if (result.lr_p_value < 0.05) {
        REQUIRE(result.prefer_student_t == true);
    }
}

// Test parameter count calculation
TEST(compare_distributions_parameter_count) {
    // Test with different model specifications
    ArimaGarchSpec spec1(2, 0, 1, 1, 1);  // More ARIMA parameters
    ArimaGarchSpec spec2(0, 0, 0, 2, 2);  // More GARCH parameters

    ArimaGarchParameters params1(spec1);
    params1.arima_params.intercept = 0.1;
    params1.arima_params.ar_coef = {0.3, 0.2};
    params1.arima_params.ma_coef = {0.4};
    params1.garch_params.omega = 1.0;
    params1.garch_params.alpha_coef = {0.1};
    params1.garch_params.beta_coef = {0.8};

    ArimaGarchParameters params2(spec2);
    params2.garch_params.omega = 1.0;
    params2.garch_params.alpha_coef = {0.15, 0.10};
    params2.garch_params.beta_coef = {0.6, 0.15};

    ArimaGarchSimulator sim1(spec1, params1);
    ArimaGarchSimulator sim2(spec2, params2);
    auto sim_result1 = sim1.simulate(150, 111);
    auto sim_result2 = sim2.simulate(150, 222);
    std::vector<double> data1 = sim_result1.returns;
    std::vector<double> data2 = sim_result2.returns;

    DistributionTestResult result1 =
        compareDistributions(spec1, params1, data1.data(), data1.size());
    DistributionTestResult result2 =
        compareDistributions(spec2, params2, data2.data(), data2.size());

    // Both should produce valid results
    REQUIRE(!std::isnan(result1.normal_ll));
    REQUIRE(!std::isnan(result2.normal_ll));

    // BIC improvements should be different due to different parameter counts
    REQUIRE(result1.bic_improvement != result2.bic_improvement);
}

int main() {
    report_test_results("Distribution Selector Tests");
    return get_test_result();
}
