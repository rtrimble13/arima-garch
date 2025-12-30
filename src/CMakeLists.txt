# Library target
add_library(arimagarch STATIC
    arima_garch.cpp
    api/Engine.cpp
    data/TimeSeries.cpp
    diagnostics/DiagnosticReport.cpp
    diagnostics/Residuals.cpp
    estimation/Constraints.cpp
    estimation/Likelihood.cpp
    estimation/NumericalDerivatives.cpp
    estimation/Optimizer.cpp
    estimation/ParameterInitialization.cpp
    forecasting/Forecaster.cpp
    io/CsvReader.cpp
    io/CsvWriter.cpp
    io/Json.cpp
    report/FitSummary.cpp
    models/arima/ArimaModel.cpp
    models/arima/ArimaState.cpp
    models/garch/GarchModel.cpp
    models/garch/GarchState.cpp
    models/composite/ArimaGarchModel.cpp
    selection/CandidateGrid.cpp
    selection/CrossValidation.cpp
    selection/DistributionSelector.cpp
    selection/InformationCriteria.cpp
    selection/ModelSelector.cpp
    simulation/ArimaGarchSimulator.cpp
    simulation/Innovations.cpp
    stats/ACF.cpp
    stats/ADF.cpp
    stats/Descriptive.cpp
    stats/JarqueBera.cpp
    stats/LjungBox.cpp
    stats/PACF.cpp
    util/Logging.cpp
    util/Timer.cpp
)

target_include_directories(arimagarch PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(arimagarch PUBLIC fmt::fmt nlohmann_json::nlohmann_json)

# CLI executable target
add_executable(ag
    cli/main.cpp
)

target_link_libraries(ag PRIVATE arimagarch CLI11::CLI11)

# Install rules for the library
install(TARGETS arimagarch
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install public headers
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ag
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Install the CLI executable
install(TARGETS ag
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
