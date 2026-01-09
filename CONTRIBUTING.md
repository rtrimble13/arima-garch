# Contributing to arima-garch

Thank you for your interest in contributing to the arima-garch project! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project follows standard open-source community guidelines. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

To contribute to this project, you'll need:

- **C++20 compatible compiler**:
  - GCC 10+ 
  - Clang 10+
  - MSVC 2019+
- **CMake** 3.14 or higher
- **Git** for version control

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/arima-garch.git
   cd arima-garch
   ```

3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/rtrimble13/arima-garch.git
   ```

4. **Create a development branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Building the Project

#### Quick Start (Recommended)

The project includes a Makefile wrapper that streamlines the CMake + Ninja workflow:

```bash
# Build in Release mode (default)
make

# Build in Debug mode
make debug

# Run tests
make test

# Clean build artifacts
make clean
```

Build artifacts are placed in `build/ninja-<buildtype>/` directories (e.g., `build/ninja-release/`).

#### Alternative: Direct CMake Usage

If you prefer to use CMake directly:

```bash
# Configure the build
cmake -S . -B build

# Build the project
cmake --build build

# Run tests
cd build && ctest
```

#### Build Targets

The project includes several build targets:

- `arimagarch` - Static library
- `ag` - Command-line interface
- `test_placeholder` - Unit tests
- `example_basic` - Example programs

You can build specific targets:

```bash
# With Make + Ninja
make  # builds all targets

# With direct CMake
cmake --build build --target ag
cmake --build build --target test_placeholder
```

### Running Tests

The project has comprehensive test suites. As you add features, ensure you add corresponding tests:

```bash
# Run all tests with Make
make test

# Or with direct CMake
cd build && ctest

# Run a specific test executable (adjust path based on build method)
./build/ninja-release/tests/test_placeholder  # if using Make
./build/tests/test_placeholder                # if using direct CMake
```

## Coding Standards

### Style Guide

All code must conform to the project's style guide documented in [docs/style.md](docs/style.md). Key points:

- **Language**: C++20
- **Naming**: 
  - Classes/Structs: PascalCase (`ARIMAModel`)
  - Functions/Methods: camelCase (`fitModel()`)
  - Variables: camelCase (`forecastHorizon`)
  - Private members: camelCase with trailing underscore (`data_`)
  - Constants: UPPER_SNAKE_CASE (`MAX_ITERATIONS`)
- **Formatting**:
  - 4 spaces for indentation (no tabs)
  - 100 character line limit
  - K&R brace style
  - Use `./scripts/format.sh` to automatically format all code
  - Check formatting with `./scripts/check-format.sh`
- **Headers**: Use `#pragma once`
- **Namespace**: All library code in `ag` namespace

### Code Quality

- Write clear, self-documenting code
- Add comments to explain complex logic or algorithms
- Follow modern C++ best practices
- Use standard library and approved dependencies (see [docs/dependencies.md](docs/dependencies.md))
- Avoid raw pointers; prefer smart pointers
- Use const correctness throughout

## Making Changes

### Workflow

1. **Sync with upstream** before starting work:
   ```bash
   git fetch upstream
   git merge upstream/main
   ```

2. **Make your changes** in focused, logical commits:
   - Keep commits small and focused on a single change
   - Write clear, descriptive commit messages
   - Follow conventional commit format when possible

3. **Test your changes**:
   - Build the project successfully
   - Run existing tests
   - Add new tests for new functionality
   - Test your changes manually when applicable
   - **Format your code**: Run `./scripts/format.sh` to ensure consistent formatting

4. **Update documentation**:
   - Update relevant documentation for your changes
   - Add examples if introducing new features
   - Update the README if needed

### Commit Messages

Write clear commit messages following this format:

```
<type>: <short summary>

<optional longer description>

<optional footer>
```

Types:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style/formatting (no functional change)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Build/tooling changes

Example:
```
feat: Add ARIMA model parameter estimation

Implement maximum likelihood estimation for ARIMA model parameters
using Nelder-Mead optimization. Includes validation for model orders
and data sufficiency checks.

Closes #42
```

## Submitting Changes

### Pull Request Process

1. **Push your changes** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub:
   - Go to the main repository
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill out the PR template

3. **PR Requirements**:
   - Clear title describing the change
   - Description explaining what and why
   - Reference any related issues
   - All tests passing
   - Code follows style guidelines
   - Documentation updated if needed

### Pull Request Checklist

Before submitting, ensure:

- [ ] Code builds without errors or warnings
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Code follows [style guide](docs/style.md) and passes formatting check (`./scripts/check-format.sh`)
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the changes
- [ ] Related issues are referenced

### Code Review

After submission:

- A maintainer will review your PR
- Address any feedback or requested changes
- Update your PR by pushing new commits to your branch
- Once approved, a maintainer will merge your PR

## What to Contribute

### Good First Issues

Look for issues labeled `good first issue` for beginner-friendly tasks.

### Areas for Contribution

- **Core Functionality**: ARIMA and GARCH model implementation
- **Algorithms**: Optimization, parameter estimation, forecasting
- **Testing**: Unit tests, integration tests, test data
- **Documentation**: API docs, examples, tutorials
- **Performance**: Optimization, benchmarking
- **Tools**: CLI features, data input/output

### Reporting Bugs

If you find a bug:

1. Check if it's already reported in the Issues
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs. actual behavior
   - Your environment (OS, compiler, CMake version)
   - Minimal code example if applicable

### Suggesting Features

For feature requests:

1. Check if it's already proposed
2. Open an issue describing:
   - The feature and its use case
   - Why it's valuable
   - Possible implementation approach
   - Alternatives considered

## Development Tips

### Useful Commands

```bash
# Clean build with Make + Ninja
make clean
make

# Reconfigure and build
make reconfigure

# Clean build with direct CMake
rm -rf build && cmake -S . -B build && cmake --build build

# Build with parallel jobs (Make)
make N=8  # Use custom Makefile parameter N to build with 8 parallel jobs

# Build with verbose output (CMake)
cmake --build build --verbose

# Run a specific test (adjust path based on build method)
./build/ninja-release/tests/test_name  # if using Make
./build/tests/test_name                # if using direct CMake

# Check for memory leaks (with valgrind, adjust path based on build method)
valgrind --leak-check=full ./build/ninja-release/src/ag  # if using Make
valgrind --leak-check=full ./build/src/ag                # if using direct CMake
```

### Editor Configuration

The repository includes an `.editorconfig` file. Use an editor or plugin that supports EditorConfig for consistent formatting.

### Code Formatting

The project uses clang-format to ensure consistent code formatting. Before submitting a pull request:

```bash
# Format all C++ files automatically
./scripts/format.sh

# Check if formatting is correct (without modifying files)
./scripts/check-format.sh
```

The CI pipeline will automatically check formatting and fail if code is not properly formatted.

### Recommended Tools

- **clang-format**: Automatic code formatting (required for contributions)
- **clang-tidy**: Static analysis
- **valgrind**: Memory leak detection
- **gdb/lldb**: Debugging

## Questions?

If you have questions:

- Check existing documentation
- Look through closed issues and PRs
- Open a new issue with the `question` label
- Reach out to maintainers

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

---

Thank you for contributing to arima-garch! Your efforts help make time series analysis more accessible to everyone.
