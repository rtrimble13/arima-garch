# Scripts

This directory contains utility scripts for the arima-garch project.

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
