# Scripts

This directory contains utility scripts for the arima-garch project.

## Version Management

### update_version.sh

Updates the project version across all relevant files in the codebase.

**Usage:**
```bash
./scripts/update_version.sh <new_version>
```

**Example:**
```bash
./scripts/update_version.sh 0.2.0
```

**What it does:**
- Validates the version format (must be X.Y.Z semantic versioning)
- Extracts the current version from each file
- Updates the version in:
  - `CMakeLists.txt` (project VERSION)
  - `src/cli/main.cpp` (CLI version flag)
  - `python/ag_viz/__init__.py` (__version__)
- Reports old version, new version, and files updated

**Exit codes:**
- `0`: Version updated successfully
- `1`: Invalid arguments or version format

## Formatting Scripts

### format.sh

Automatically formats all C++ source files in the project using clang-format.

**Usage:**
```bash
./scripts/format.sh
```

**Requirements:**
- clang-format (version 10 or higher recommended)

**What it does:**
- Finds all C++ files (`.cpp`, `.hpp`, `.h`, `.cc`, `.cxx`, `.hxx`)
- Excludes build directories and hidden directories
- Formats files in-place according to `.clang-format` configuration

### check-format.sh

Checks if all C++ source files are properly formatted without modifying them.

**Usage:**
```bash
./scripts/check-format.sh
```

**Exit codes:**
- `0`: All files are properly formatted
- `1`: One or more files have formatting issues

**Requirements:**
- clang-format (version 10 or higher recommended)

**What it does:**
- Finds all C++ files (`.cpp`, `.hpp`, `.h`, `.cc`, `.cxx`, `.hxx`)
- Excludes build directories and hidden directories
- Checks each file against `.clang-format` configuration
- Reports any files with formatting issues

**CI Integration:**
This script is run automatically in the CI pipeline to ensure all code is properly formatted before merging.

## Installation

These scripts require clang-format to be installed:

**Ubuntu/Debian:**
```bash
sudo apt-get install clang-format
```

**macOS:**
```bash
brew install clang-format
```

**Arch Linux:**
```bash
sudo pacman -S clang
```
