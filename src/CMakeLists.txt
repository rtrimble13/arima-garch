# Library target
add_library(arimagarch STATIC
    arima_garch.cpp
    data/TimeSeries.cpp
    io/CsvReader.cpp
    io/CsvWriter.cpp
    stats/Descriptive.cpp
    util/Logging.cpp
    util/Timer.cpp
)

target_include_directories(arimagarch PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(arimagarch PUBLIC fmt::fmt)

# CLI executable target
add_executable(ag
    main.cpp
)

target_link_libraries(ag PRIVATE arimagarch)
