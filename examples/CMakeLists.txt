# Example programs demonstrating library usage

add_executable(example_basic
    example_basic.cpp
)

target_link_libraries(example_basic PRIVATE arimagarch)

add_executable(example_csv_io
    example_csv_io.cpp
)

target_link_libraries(example_csv_io PRIVATE arimagarch)

add_executable(example_json_io
    example_json_io.cpp
)

target_link_libraries(example_json_io PRIVATE arimagarch)

add_executable(example_acf_pacf
    example_acf_pacf.cpp
)

target_link_libraries(example_acf_pacf PRIVATE arimagarch)

add_executable(example_ljung_box
    example_ljung_box.cpp
)

target_link_libraries(example_ljung_box PRIVATE arimagarch)

add_executable(example_constraints
    example_constraints.cpp
)

target_link_libraries(example_constraints PRIVATE arimagarch)

add_executable(example_arima_residuals
    example_arima_residuals.cpp
)

target_link_libraries(example_arima_residuals PRIVATE arimagarch)

# Example: ARIMA-GARCH residuals diagnostics
add_executable(example_residuals
    example_residuals.cpp
)

target_link_libraries(example_residuals PRIVATE arimagarch)

# Example: Comprehensive diagnostic report with Ljung-Box, JB, and ADF tests
add_executable(example_diagnostic_report
    example_diagnostic_report.cpp
)

target_link_libraries(example_diagnostic_report PRIVATE arimagarch)

# Example: Parameter initialization and random restarts
add_executable(example_initialization
    example_initialization.cpp
)

target_link_libraries(example_initialization PRIVATE arimagarch)

# Example: ARIMA-GARCH simulation
add_executable(example_simulation
    example_simulation.cpp
)

target_link_libraries(example_simulation PRIVATE arimagarch)

# Example: FitSummary and text report generation
add_executable(example_fit_summary
    example_fit_summary.cpp
)

target_link_libraries(example_fit_summary PRIVATE arimagarch)

# Example: CandidateGrid for model specification generation
add_executable(example_candidate_grid
    example_candidate_grid.cpp
)

target_link_libraries(example_candidate_grid PRIVATE arimagarch)
