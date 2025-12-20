# C++ Coding Style Guide

This document describes the coding standards for the arima-garch project.

## Language Standard

This project uses **C++20**. All code must be compatible with C++20 standards.

## Naming Conventions

### Files

- **Header files**: Use `.hpp` extension for C++ headers
- **Source files**: Use `.cpp` extension for implementation files
- **File names**: Use lowercase with underscores (snake_case)
  - Example: `arima_garch.hpp`, `time_series_utils.cpp`

### Namespaces

- All library code must be in the `ag` namespace (short for arima-garch)
- Nested namespaces are allowed for logical grouping
- Use namespace aliases for long namespace names when appropriate
- Never use `using namespace` in header files
- Avoid `using namespace` in source files; prefer explicit qualification or targeted `using` declarations

```cpp
// Good
namespace ag {
namespace models {
    class ARIMAModel { /* ... */ };
} // namespace models
} // namespace ag

// Also acceptable (C++17 nested namespace syntax)
namespace ag::models {
    class ARIMAModel { /* ... */ };
}
```

### Classes and Structs

- Use PascalCase (UpperCamelCase) for class and struct names
- Use descriptive, unabbreviated names
- Examples: `ARIMAModel`, `GARCHModel`, `TimeSeries`, `ForecastResult`

### Functions and Methods

- Use camelCase (lowerCamelCase) for function and method names
- Use verb phrases for functions that perform actions
- Use noun phrases for functions that return properties
- Examples: `fitModel()`, `forecast()`, `calculateVariance()`, `getParameters()`

### Variables

- Use camelCase (lowerCamelCase) for local variables and function parameters
- Use descriptive names that convey meaning
- Avoid single-letter names except for common loop counters (`i`, `j`, `k`) or mathematical variables in formulas
- Examples: `modelParams`, `timeSeriesData`, `forecastHorizon`

### Member Variables

- Use camelCase with a trailing underscore for private member variables
- Example: `data_`, `parameters_`, `isInitialized_`
- Public member variables (in structs) do not need the trailing underscore

### Constants and Enums

- Use UPPER_SNAKE_CASE for constants and enum values
- Use `constexpr` for compile-time constants
- Use `inline constexpr` for constants in headers (C++17+)
- Examples:
  ```cpp
  constexpr int MAX_ITERATIONS = 1000;
  constexpr double DEFAULT_TOLERANCE = 1e-6;
  
  enum class OptimizationMethod {
      GRADIENT_DESCENT,
      NEWTON_RAPHSON,
      BFGS
  };
  ```

### Template Parameters

- Use PascalCase for type template parameters
- Use UPPER_SNAKE_CASE for non-type template parameters
- Example:
  ```cpp
  template<typename ValueType, size_t MAX_SIZE>
  class FixedSizeVector { /* ... */ };
  ```

## Header File Organization

Headers should be organized in the following order:

1. **Copyright/license comment** (if applicable)
2. **Header guard** (`#pragma once` preferred)
3. **System includes** (C++ standard library headers)
4. **Third-party library includes** (Boost, etc.)
5. **Project includes** (local headers)
6. **Forward declarations** (if needed)
7. **Namespace opening**
8. **Type definitions and aliases**
9. **Constants**
10. **Class declarations**
11. **Function declarations**
12. **Inline function definitions** (if small and performance-critical)
13. **Namespace closing**

Example:

```cpp
#pragma once

#include <vector>
#include <string>

#include <boost/numeric/ublas/matrix.hpp>

#include "ag/common_types.hpp"

namespace ag {

// Forward declarations
class GARCHModel;

// Type aliases
using TimeSeries = std::vector<double>;
using Matrix = boost::numeric::ublas::matrix<double>;

// Constants
inline constexpr double DEFAULT_ALPHA = 0.05;

// Class declaration
class ARIMAModel {
public:
    ARIMAModel(int p, int d, int q);
    
    void fit(const TimeSeries& data);
    TimeSeries forecast(int horizon) const;
    
private:
    int p_;
    int d_;
    int q_;
    std::vector<double> parameters_;
};

} // namespace ag
```

## Source File Organization

Source files should be organized in the following order:

1. **Corresponding header** (for `.cpp` files)
2. **System includes**
3. **Third-party library includes**
4. **Other project includes**
5. **Anonymous namespace** (for file-local helpers)
6. **Namespace opening**
7. **Implementation code**
8. **Namespace closing**

Example:

```cpp
#include "ag/arima_model.hpp"

#include <algorithm>
#include <cmath>

#include <boost/math/distributions.hpp>

#include "ag/optimization.hpp"

namespace {

// File-local helper functions
double calculateLogLikelihood(const std::vector<double>& residuals) {
    // Implementation
}

} // anonymous namespace

namespace ag {

ARIMAModel::ARIMAModel(int p, int d, int q)
    : p_(p), d_(d), q_(q) {
    // Implementation
}

void ARIMAModel::fit(const TimeSeries& data) {
    // Implementation
}

} // namespace ag
```

## Error Handling

### Exceptions

- Use exceptions for exceptional conditions, not for normal control flow
- Derive custom exceptions from standard exception classes
- Provide meaningful error messages
- Use `std::runtime_error` for runtime errors
- Use `std::invalid_argument` for invalid function arguments
- Use `std::logic_error` for logic errors (programming errors)

```cpp
class ModelFitError : public std::runtime_error {
public:
    explicit ModelFitError(const std::string& message)
        : std::runtime_error(message) {}
};

void ARIMAModel::fit(const TimeSeries& data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot fit model to empty data");
    }
    
    if (data.size() < static_cast<size_t>(p_ + d_ + q_)) {
        throw ModelFitError("Insufficient data points for model order");
    }
    
    // Fitting logic...
}
```

### Return Values and Error Codes

- Prefer exceptions for error handling in library code
- For CLI applications, catch exceptions at the top level and return appropriate exit codes
- Use `std::optional` for functions that may not return a value
- Use `std::expected` (C++23) or similar types for operations that may fail without being exceptional

```cpp
// Good: Using std::optional for possibly absent values
std::optional<double> ARIMAModel::getParameter(size_t index) const {
    if (index >= parameters_.size()) {
        return std::nullopt;
    }
    return parameters_[index];
}
```

### Assertions

- Use `assert()` for debugging checks that should never fail in correct code
- Use static_assert for compile-time checks
- Remove or disable assertions in release builds for performance

```cpp
#include <cassert>

void processData(const std::vector<double>& data) {
    assert(!data.empty() && "Data vector should not be empty");
    assert(data.size() <= MAX_SIZE && "Data exceeds maximum size");
    // Processing...
}
```

## Code Formatting

### Indentation and Spacing

- Use **4 spaces** for indentation (no tabs)
- Maximum line length: **100 characters** (prefer 80 when practical)
- Place opening braces on the same line as the statement (K&R style)
- Use a single space before opening braces
- No spaces inside parentheses or brackets
- One space after commas in parameter lists

```cpp
// Good
if (condition) {
    doSomething();
}

for (int i = 0; i < n; ++i) {
    process(data[i]);
}

void function(int a, int b, int c) {
    // Implementation
}
```

### Include Guards

- Prefer `#pragma once` over traditional include guards
- If traditional guards are needed, use format: `PROJECT_PATH_FILENAME_HPP_`

```cpp
#pragma once

// Header content
```

### Comments

- Use `//` for single-line comments
- Use `/* */` for multi-line comments
- Write self-documenting code; use comments to explain "why", not "what"
- Add comments for complex algorithms or non-obvious code
- Use Doxygen-style comments for public API documentation

```cpp
/**
 * @brief Fits an ARIMA model to the provided time series data.
 * 
 * @param data The time series data to fit
 * @param maxIterations Maximum number of optimization iterations
 * @return true if fitting converged, false otherwise
 * 
 * @throws std::invalid_argument if data is empty or insufficient
 * @throws ModelFitError if optimization fails to converge
 */
bool ARIMAModel::fit(const TimeSeries& data, int maxIterations = 100);
```

### Modern C++ Features

Use modern C++ features appropriately:

- **Prefer `auto`** when type is obvious from context or overly verbose
- **Use range-based for loops** when iterating over containers
- **Use smart pointers** (`std::unique_ptr`, `std::shared_ptr`) instead of raw pointers for ownership
- **Use `nullptr`** instead of `NULL` or `0` for null pointers
- **Use `enum class`** instead of plain enums
- **Use `constexpr`** for compile-time constants and functions
- **Use structured bindings** (C++17) for tuple-like types
- **Use `[[nodiscard]]`** for functions where ignoring the return value is likely an error

```cpp
// Good examples
auto modelParams = model.getParameters();  // Type obvious from function name

for (const auto& value : timeSeries) {
    process(value);
}

auto model = std::make_unique<ARIMAModel>(1, 1, 1);

if (auto result = tryFit(data); result.has_value()) {
    // Use result.value()
}
```

## Best Practices

### General Guidelines

1. **Follow the Single Responsibility Principle**: Each class/function should have one clear purpose
2. **Prefer composition over inheritance**
3. **Make interfaces easy to use correctly and hard to use incorrectly**
4. **Use const correctness**: Mark methods and parameters as `const` when they don't modify state
5. **Minimize dependencies**: Reduce coupling between components
6. **Prefer standard library** over custom implementations
7. **Write unit tests** for all new functionality

### Performance Considerations

- Pass large objects by const reference: `const std::vector<double>&`
- Return large objects by value (rely on RVO/move semantics)
- Use move semantics when appropriate
- Prefer `reserve()` for vectors when final size is known
- Use `emplace_back()` instead of `push_back()` for in-place construction

```cpp
// Good
void processData(const std::vector<double>& data) {
    std::vector<double> results;
    results.reserve(data.size());
    
    for (const auto& value : data) {
        results.emplace_back(transform(value));
    }
    
    return results;  // RVO or move
}
```

### Thread Safety

- Document thread-safety guarantees for public APIs
- Use `std::mutex` or other synchronization primitives when needed
- Prefer immutable data structures when possible
- Use `std::atomic` for lock-free operations on simple types

## Tools and Automation

### Recommended Tools

- **clang-format**: For automatic code formatting (configuration TBD)
- **clang-tidy**: For static analysis and linting
- **CMake**: Build system
- **Google Test or Catch2**: For unit testing

### Editor Configuration

See the `.editorconfig` file in the project root for editor-agnostic formatting settings.

## References

- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [Modern C++ Best Practices](https://www.modernescpp.com/)
