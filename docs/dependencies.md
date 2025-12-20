# Dependency Management Strategy

This document describes how external dependencies are managed in the arima-garch project.

## Chosen Strategy: CMake FetchContent

After evaluating several dependency management approaches, we have chosen **CMake FetchContent** as our primary dependency management strategy.

### Rationale

**Why FetchContent?**

1. **Simplicity**: FetchContent is built directly into CMake (3.14+), requiring no additional tools or package managers to install
2. **Cross-Platform**: Works seamlessly on Windows, Linux, and macOS without platform-specific configuration
3. **Fresh Machine Builds**: Only requires CMake and a C++20 compiler - no vcpkg, Conan, or other external tools needed
4. **Modern CMake Integration**: Integrates naturally with modern CMake best practices and target-based dependency management
5. **Source-Based**: Downloads and builds dependencies from source, ensuring compatibility with project settings (C++20, compiler flags, etc.)
6. **Reproducible Builds**: Dependencies are pinned to specific Git tags/commits, ensuring consistent builds across environments
7. **IDE Support**: Works well with IDEs that support CMake (Visual Studio, CLion, VS Code, etc.)

**Why Not vcpkg or Conan?**

While vcpkg and Conan are excellent package managers for larger projects with complex dependency graphs:
- **vcpkg**: Requires separate installation and setup; adds complexity for contributors; platform-specific gotchas
- **Conan**: Requires Python and separate toolchain; steeper learning curve; another tool to maintain

For a project of this scope with a small number of well-maintained dependencies, FetchContent provides the best balance of simplicity and functionality.

### Alternative Approaches Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **FetchContent** | Built-in, simple, cross-platform | Can be slower on first build | ✅ **Chosen** |
| **vcpkg** | Binary caching, large ecosystem | Extra tool, setup complexity | ❌ Not chosen |
| **Conan** | Mature, binary packages | Requires Python, learning curve | ❌ Not chosen |
| **Git submodules** | Simple, version-controlled | Manual management, outdated | ❌ Not chosen |
| **System packages** | Fast when available | Platform-specific, version conflicts | ❌ Not chosen |

## Planned Dependencies

The following external libraries will be integrated into the project:

### 1. fmt (Logging and Formatting)

- **Purpose**: Modern C++ string formatting and logging
- **Repository**: https://github.com/fmtlib/fmt
- **Version**: 10.x (latest stable)
- **License**: MIT
- **Rationale**: 
  - High-performance, type-safe alternative to iostreams and printf
  - Basis for C++20 `std::format` (provides compatibility layer)
  - Excellent for diagnostic logging and user-facing output in CLI

**Usage Example**:
```cpp
#include <fmt/core.h>
#include <fmt/format.h>

fmt::print("Fitting ARIMA({}, {}, {}) model...\n", p, d, q);
auto message = fmt::format("Convergence achieved after {} iterations", iterCount);
```

### 2. nlohmann/json (JSON Serialization)

- **Purpose**: Model serialization, configuration files, and report generation
- **Repository**: https://github.com/nlohmann/json
- **Version**: 3.11.x (latest stable)
- **License**: MIT
- **Rationale**:
  - Header-only library (no build time impact)
  - Intuitive API, STL-like syntax
  - Widely used and well-maintained
  - Perfect for saving/loading model parameters and generating reports

**Usage Example**:
```cpp
#include <nlohmann/json.hpp>

nlohmann::json modelConfig;
modelConfig["arima"]["p"] = p;
modelConfig["arima"]["d"] = d;
modelConfig["arima"]["q"] = q;
modelConfig["garch"]["P"] = P;
modelConfig["garch"]["Q"] = Q;

// Save to file
std::ofstream file("model_config.json");
file << modelConfig.dump(2);
```

### 3. CLI11 (Command-Line Parsing)

- **Purpose**: Parse command-line arguments for the `ag` executable
- **Repository**: https://github.com/CLIUtils/CLI11
- **Version**: 2.x (latest stable)
- **License**: BSD-3-Clause
- **Rationale**:
  - Header-only library
  - Modern C++11+ design
  - Intuitive API with excellent error messages
  - Support for subcommands (fit, forecast, simulate)
  - Built-in help generation

**Alternative Considered**: cxxopts (also good, but CLI11 has better subcommand support)

**Usage Example**:
```cpp
#include <CLI/CLI.hpp>

CLI::App app{"ARIMA-GARCH Time Series Modeling"};

auto* fit = app.add_subcommand("fit", "Fit a model to data");
std::string dataFile;
fit->add_option("--data", dataFile, "Input time series data")->required();

CLI11_PARSE(app, argc, argv);
```

### 4. Catch2 (Unit Testing Framework)

- **Purpose**: Unit testing and test-driven development
- **Repository**: https://github.com/catchorg/Catch2
- **Version**: 3.x (latest stable)
- **License**: BSL-1.0 (Boost Software License)
- **Rationale**:
  - Modern, header-focused design (v3 is library-based but still lightweight)
  - Expressive BDD-style syntax
  - Excellent error messages and test output
  - No external dependencies
  - Great for TDD workflow

**Alternative Considered**: GoogleTest (more verbose, heavier, but also excellent)

**Usage Example**:
```cpp
#include <catch2/catch_test_macros.hpp>

TEST_CASE("ARIMA model parameter validation", "[arima]") {
    REQUIRE_THROWS_AS(ARIMAModel(-1, 0, 1), std::invalid_argument);
    REQUIRE_THROWS_AS(ARIMAModel(1, -1, 1), std::invalid_argument);
}

TEST_CASE("ARIMA model fitting", "[arima][integration]") {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    ARIMAModel model(1, 0, 1);
    
    REQUIRE_NOTHROW(model.fit(data));
    REQUIRE(model.isFitted());
}
```

## Dependency Summary Table

| Library | Purpose | Type | License | Version |
|---------|---------|------|---------|---------|
| **fmt** | Formatting/Logging | Compiled | MIT | 10.x |
| **nlohmann/json** | JSON I/O | Header-only | MIT | 3.11.x |
| **CLI11** | CLI Parsing | Header-only | BSD-3 | 2.x |
| **Catch2** | Unit Testing | Compiled | BSL-1.0 | 3.x |

## Implementation

Dependencies are managed in the root `CMakeLists.txt` file using CMake's `FetchContent` module. Here's the general pattern:

```cmake
include(FetchContent)

# Declare and fetch fmt
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        10.2.1  # Specific version tag
)
FetchContent_MakeAvailable(fmt)

# Usage in targets
target_link_libraries(arimagarch PUBLIC fmt::fmt)
```

### Build Process

When a developer runs CMake for the first time:

1. CMake detects missing dependencies
2. FetchContent downloads source from GitHub (specific tagged versions)
3. Dependencies are built as part of the project build
4. Subsequent builds use cached dependencies (no re-download)

### Build Instructions for Fresh Machine

To build the project on a fresh machine, you only need:

**Prerequisites:**
- CMake 3.14 or higher
- C++20 compatible compiler (GCC 10+, Clang 10+, MSVC 2019+)
- Git (for FetchContent to clone dependencies)
- Internet connection (for initial dependency download)

**Build Steps:**
```bash
# Clone the repository
git clone https://github.com/rtrimble13/arima-garch.git
cd arima-garch

# Configure (downloads dependencies automatically)
cmake -S . -B build

# Build
cmake --build build

# Run tests
cd build && ctest

# Run the CLI
./build/src/ag --help
```

That's it! No separate package manager installation or dependency pre-installation required.

### Offline Builds

For environments without internet access:
1. Perform an initial build on a machine with internet
2. Archive the `build/_deps` directory
3. Restore it on the offline machine before building

Or use CMake's `FETCHCONTENT_FULLY_DISCONNECTED` mode with pre-populated source directories.

## Dependency Update Policy

### When to Update Dependencies

- **Security patches**: Update immediately
- **Bug fixes**: Update in next minor version
- **New features**: Evaluate and update as needed
- **Major versions**: Evaluate breaking changes carefully

### How to Update Dependencies

Update the `GIT_TAG` in `CMakeLists.txt` to the desired version:

```cmake
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        10.2.1  # Change this to new version
)
```

Then rebuild:
```bash
rm -rf build/_deps  # Clear cached dependencies
cmake -S . -B build
cmake --build build
```

## Troubleshooting

### Common Issues

**Issue**: "Could not resolve host: github.com"
- **Solution**: Check internet connection; ensure firewall/proxy allows Git access

**Issue**: Build is slow on first run
- **Solution**: Expected behavior - dependencies are being built from source. Subsequent builds are fast.

**Issue**: Dependency version conflicts
- **Solution**: FetchContent builds from source with project settings, avoiding system library conflicts

**Issue**: CMake version too old
- **Solution**: Upgrade to CMake 3.14+. On Ubuntu: `sudo snap install cmake --classic`

## Future Considerations

As the project grows, we may consider:

1. **Binary Caching**: If build times become problematic, investigate CMake's binary caching or ccache
2. **Precompiled Headers**: For large dependencies like Boost (if added)
3. **Conan/vcpkg Migration**: If dependency count grows significantly (10+ libraries)

For now, FetchContent provides the right balance of simplicity and functionality for this project.

## References

- [CMake FetchContent Documentation](https://cmake.org/cmake/help/latest/module/FetchContent.html)
- [Modern CMake Best Practices](https://cliutils.gitlab.io/modern-cmake/)
- [C++ Package Manager Comparison](https://www.reddit.com/r/cpp/comments/kqr1q7/package_managers_comparison/)
