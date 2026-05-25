cmake_minimum_required(VERSION 3.14)

# Resolve project version from git tag. See cmake/GetVersion.cmake
# and docs/versioning.md.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(GetVersion)
ag_get_version()
message(STATUS "arima-garch version: ${AG_VERSION_FULL}")

project(arima-garch VERSION ${AG_VERSION_NUMERIC} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generated Version.hpp consumed by the library and CLI.
configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/Version.hpp.in
  ${CMAKE_BINARY_DIR}/generated/ag/Version.hpp
  @ONLY)

# Include GNUInstallDirs for standard installation directories
include(GNUInstallDirs)

# Fetch external dependencies
include(FetchContent)

# fmt library for formatting and logging
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        10.2.1
)
FetchContent_MakeAvailable(fmt)

# nlohmann/json library for JSON serialization
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(json)

# CLI11 library for command-line parsing
FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.4.1
)
FetchContent_MakeAvailable(cli11)

# Enable testing
enable_testing()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(benchmarks)
