# Library target
add_library(arimagarch STATIC
    arima_garch.cpp
    api/Engine.cpp
    data/TimeSeries.cpp
    diagnostics/DiagnosticReport.cpp
    diagnostics/Residuals.cpp
    estimation/Constraints.cpp
    estimation/FitDriver.cpp
    estimation/Likelihood.cpp
    estimation/NumericalDerivatives.cpp
    estimation/Optimizer.cpp
    estimation/ParameterInitialization.cpp
    estimation/ParameterVector.cpp
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
    stats/Bootstrap.cpp
    stats/Descriptive.cpp
    stats/JarqueBera.cpp
    stats/LjungBox.cpp
    stats/PACF.cpp
    stats/SpecialFunctions.cpp
    util/Differencing.cpp
    util/LinearAlgebra.cpp
    util/Logging.cpp
    util/Timer.cpp
)

target_include_directories(arimagarch PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/generated
)

target_link_libraries(arimagarch PUBLIC fmt::fmt nlohmann_json::nlohmann_json)

# CLI executable target
add_executable(ag
    cli/main.cpp
    cli/CliUtils.cpp
    cli/handlers/DiagnosticsHandler.cpp
    cli/handlers/FitHandler.cpp
    cli/handlers/ForecastHandler.cpp
    cli/handlers/SelectHandler.cpp
    cli/handlers/SimulateFromModelHandler.cpp
    cli/handlers/SimulateHandler.cpp
)

target_include_directories(ag PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cli)
target_link_libraries(ag PRIVATE arimagarch CLI11::CLI11)

# Stage `ag` under examples/terminal/bin so the terminal example scripts find
# it at a stable, generator-independent path. Opt-in:
#   cmake --build <build-dir> --target ag-examples
add_custom_target(ag-examples
    DEPENDS ag
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${PROJECT_SOURCE_DIR}/examples/terminal/bin
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:ag>
        ${PROJECT_SOURCE_DIR}/examples/terminal/bin/ag
    COMMENT "Stage ag binary at examples/terminal/bin/ag for example scripts"
    VERBATIM
)

# Install rules for library and executable
install(TARGETS arimagarch ag
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install public headers
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ag
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Install the generated Version.hpp alongside the public headers.
install(FILES ${CMAKE_BINARY_DIR}/generated/ag/Version.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ag
)
