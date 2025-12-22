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

# Placeholder for tests (keeping for backward compatibility)
add_executable(test_placeholder
    test_placeholder.cpp
)

target_link_libraries(test_placeholder PRIVATE arimagarch)
