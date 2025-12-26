# Unit tests
add_executable(ag_tests
    unit/test_sanity.cpp
)

target_link_libraries(ag_tests PRIVATE arimagarch)

# Add CTest integration
add_test(NAME sanity_test COMMAND ag_tests)

# Utility tests
add_executable(test_util_expected
    unit/test_util_expected.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_util_expected PRIVATE arimagarch)
add_test(NAME test_util_expected COMMAND test_util_expected)

add_executable(test_util_timer
    unit/test_util_timer.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_util_timer PRIVATE arimagarch)
add_test(NAME test_util_timer COMMAND test_util_timer)

add_executable(test_util_logging
    unit/test_util_logging.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_util_logging PRIVATE arimagarch)
add_test(NAME test_util_logging COMMAND test_util_logging)

# Data tests
add_executable(test_data_timeseries
    unit/test_data_timeseries.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_data_timeseries PRIVATE arimagarch)
add_test(NAME test_data_timeseries COMMAND test_data_timeseries)

# IO tests
add_executable(test_io_csv
    unit/test_io_csv.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_io_csv PRIVATE arimagarch)
add_test(NAME test_io_csv COMMAND test_io_csv)

add_executable(test_io_json
    unit/test_io_json.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_io_json PRIVATE arimagarch)
add_test(NAME test_io_json COMMAND test_io_json)

# Stats tests
add_executable(test_stats_descriptive
    unit/test_stats_descriptive.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_stats_descriptive PRIVATE arimagarch)
add_test(NAME test_stats_descriptive COMMAND test_stats_descriptive)

add_executable(test_stats_acf
    unit/test_stats_acf.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_stats_acf PRIVATE arimagarch)
add_test(NAME test_stats_acf COMMAND test_stats_acf)

add_executable(test_stats_pacf
    unit/test_stats_pacf.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_stats_pacf PRIVATE arimagarch)
add_test(NAME test_stats_pacf COMMAND test_stats_pacf)

add_executable(test_stats_ljungbox
    unit/test_stats_ljungbox.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_stats_ljungbox PRIVATE arimagarch)
add_test(NAME test_stats_ljungbox COMMAND test_stats_ljungbox)

add_executable(test_stats_jarquebera
    unit/test_stats_jarquebera.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_stats_jarquebera PRIVATE arimagarch)
add_test(NAME test_stats_jarquebera COMMAND test_stats_jarquebera)

add_executable(test_stats_adf
    unit/test_stats_adf.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_stats_adf PRIVATE arimagarch)
add_test(NAME test_stats_adf COMMAND test_stats_adf)

# Model tests
add_executable(test_models_specs
    unit/test_models_specs.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_models_specs PRIVATE arimagarch)
add_test(NAME test_models_specs COMMAND test_models_specs)

add_executable(test_models_arima
    unit/test_models_arima.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_models_arima PRIVATE arimagarch)
add_test(NAME test_models_arima COMMAND test_models_arima)

add_executable(test_models_garch
    unit/test_models_garch.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_models_garch PRIVATE arimagarch)
add_test(NAME test_models_garch COMMAND test_models_garch)

add_executable(test_models_composite
    unit/test_models_composite.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_models_composite PRIVATE arimagarch)
add_test(NAME test_models_composite COMMAND test_models_composite)

# Estimation tests
add_executable(test_estimation_parameters
    unit/test_estimation_parameters.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_estimation_parameters PRIVATE arimagarch)
add_test(NAME test_estimation_parameters COMMAND test_estimation_parameters)

add_executable(test_estimation_constraints
    unit/test_estimation_constraints.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_estimation_constraints PRIVATE arimagarch)
add_test(NAME test_estimation_constraints COMMAND test_estimation_constraints)

add_executable(test_estimation_likelihood
    unit/test_estimation_likelihood.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_estimation_likelihood PRIVATE arimagarch)
add_test(NAME test_estimation_likelihood COMMAND test_estimation_likelihood)

add_executable(test_estimation_numerical_derivatives
    unit/test_estimation_numerical_derivatives.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_estimation_numerical_derivatives PRIVATE arimagarch)
add_test(NAME test_estimation_numerical_derivatives COMMAND test_estimation_numerical_derivatives)

add_executable(test_estimation_optimizer
    unit/test_estimation_optimizer.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_estimation_optimizer PRIVATE arimagarch)
add_test(NAME test_estimation_optimizer COMMAND test_estimation_optimizer)

add_executable(test_estimation_initialization
    unit/test_estimation_initialization.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_estimation_initialization PRIVATE arimagarch)
add_test(NAME test_estimation_initialization COMMAND test_estimation_initialization)

# Forecasting tests
add_executable(test_forecasting
    unit/test_forecasting.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_forecasting PRIVATE arimagarch)
add_test(NAME test_forecasting COMMAND test_forecasting)

# Simulation tests
add_executable(test_simulation
    unit/test_simulation.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_simulation PRIVATE arimagarch)
add_test(NAME test_simulation COMMAND test_simulation)

# Diagnostics tests
add_executable(test_diagnostics_residuals
    unit/test_diagnostics_residuals.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_diagnostics_residuals PRIVATE arimagarch)
add_test(NAME test_diagnostics_residuals COMMAND test_diagnostics_residuals)

add_executable(test_diagnostics_report
    unit/test_diagnostics_report.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_diagnostics_report PRIVATE arimagarch)
add_test(NAME test_diagnostics_report COMMAND test_diagnostics_report)

# Report tests
add_executable(test_report_fit_summary
    unit/test_report_fit_summary.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_report_fit_summary PRIVATE arimagarch)
add_test(NAME test_report_fit_summary COMMAND test_report_fit_summary)

# Selection tests
add_executable(test_selection_candidate_grid
    unit/test_selection_candidate_grid.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_selection_candidate_grid PRIVATE arimagarch)
add_test(NAME test_selection_candidate_grid COMMAND test_selection_candidate_grid)

add_executable(test_selection_information_criteria
    unit/test_selection_information_criteria.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_selection_information_criteria PRIVATE arimagarch)
add_test(NAME test_selection_information_criteria COMMAND test_selection_information_criteria)

add_executable(test_selection_model_selector
    unit/test_selection_model_selector.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_selection_model_selector PRIVATE arimagarch)
add_test(NAME test_selection_model_selector COMMAND test_selection_model_selector)

add_executable(test_selection_cross_validation
    unit/test_selection_cross_validation.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_selection_cross_validation PRIVATE arimagarch)
add_test(NAME test_selection_cross_validation COMMAND test_selection_cross_validation)

# API tests
add_executable(test_api_engine
    unit/test_api_engine.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_api_engine PRIVATE arimagarch)
add_test(NAME test_api_engine COMMAND test_api_engine)

# Integration tests
add_executable(test_integration_e2e
    unit/test_integration_e2e.cpp
    unit/test_framework.cpp
)
target_link_libraries(test_integration_e2e PRIVATE arimagarch)
add_test(NAME test_integration_e2e COMMAND test_integration_e2e)

# Placeholder for tests (keeping for backward compatibility)
add_executable(test_placeholder
    test_placeholder.cpp
)

target_link_libraries(test_placeholder PRIVATE arimagarch)
