#!/bin/bash
# Update project version across all relevant files

set -e

# Get the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Check if version argument is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <new_version>"
  echo "Example: $0 0.2.0"
  exit 1
fi

NEW_VERSION="$1"

# Validate version format (basic semantic versioning check)
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: Version must be in format X.Y.Z (e.g., 0.2.0)"
  exit 1
fi

# Define the three files to update
CMAKE_FILE="${PROJECT_ROOT}/CMakeLists.txt"
CLI_FILE="${PROJECT_ROOT}/src/cli/main.cpp"
PYTHON_FILE="${PROJECT_ROOT}/python/ag_viz/__init__.py"

# Check that all files exist
if [ ! -f "$CMAKE_FILE" ]; then
  echo "Error: CMakeLists.txt not found at ${CMAKE_FILE}"
  exit 1
fi

if [ ! -f "$CLI_FILE" ]; then
  echo "Error: main.cpp not found at ${CLI_FILE}"
  exit 1
fi

if [ ! -f "$PYTHON_FILE" ]; then
  echo "Error: __init__.py not found at ${PYTHON_FILE}"
  exit 1
fi

# Extract old versions from each file
echo "Extracting current versions..."
echo ""

# Extract from CMakeLists.txt
OLD_CMAKE_VERSION=$(grep -E "^project\(arima-garch VERSION" "$CMAKE_FILE" | sed -E 's/.*VERSION ([0-9]+\.[0-9]+\.[0-9]+).*/\1/')
if [ -z "$OLD_CMAKE_VERSION" ]; then
  echo "Error: Could not extract version from CMakeLists.txt"
  exit 1
fi

# Extract from main.cpp
OLD_CLI_VERSION=$(grep -E 'app\.set_version_flag.*"[0-9]+\.[0-9]+\.[0-9]+"' "$CLI_FILE" | sed -E 's/.*"([0-9]+\.[0-9]+\.[0-9]+)".*/\1/')
if [ -z "$OLD_CLI_VERSION" ]; then
  echo "Error: Could not extract version from main.cpp"
  exit 1
fi

# Extract from __init__.py
OLD_PYTHON_VERSION=$(grep -E '^__version__ = "[0-9]+\.[0-9]+\.[0-9]+"' "$PYTHON_FILE" | sed -E 's/.*"([0-9]+\.[0-9]+\.[0-9]+)".*/\1/')
if [ -z "$OLD_PYTHON_VERSION" ]; then
  echo "Error: Could not extract version from __init__.py"
  exit 1
fi

# Check if versions are consistent
if [ "$OLD_CMAKE_VERSION" != "$OLD_CLI_VERSION" ] || [ "$OLD_CMAKE_VERSION" != "$OLD_PYTHON_VERSION" ]; then
  echo "Warning: Current versions are not consistent across files!"
  echo "  CMakeLists.txt: ${OLD_CMAKE_VERSION}"
  echo "  main.cpp:       ${OLD_CLI_VERSION}"
  echo "  __init__.py:    ${OLD_PYTHON_VERSION}"
  echo ""
fi

# Display old and new versions
echo "Old version: ${OLD_CMAKE_VERSION}"
echo "New version: ${NEW_VERSION}"
echo ""

# Confirm if versions are the same
if [ "$OLD_CMAKE_VERSION" = "$NEW_VERSION" ]; then
  echo "Warning: New version is the same as the current version"
  read -p "Continue anyway? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
  fi
fi

# Update CMakeLists.txt
echo "Updating ${CMAKE_FILE#${PROJECT_ROOT}/}..."
sed -i "s/^project(arima-garch VERSION ${OLD_CMAKE_VERSION}/project(arima-garch VERSION ${NEW_VERSION}/" "$CMAKE_FILE"

# Update main.cpp
echo "Updating ${CLI_FILE#${PROJECT_ROOT}/}..."
sed -i "s/app\.set_version_flag(\"--version,-v\", \"${OLD_CLI_VERSION}\")/app.set_version_flag(\"--version,-v\", \"${NEW_VERSION}\")/" "$CLI_FILE"

# Update __init__.py
echo "Updating ${PYTHON_FILE#${PROJECT_ROOT}/}..."
sed -i "s/^__version__ = \"${OLD_PYTHON_VERSION}\"/__version__ = \"${NEW_VERSION}\"/" "$PYTHON_FILE"

echo ""
echo "✅ Version updated successfully!"
echo ""
echo "Files updated:"
echo "  - ${CMAKE_FILE#${PROJECT_ROOT}/}"
echo "  - ${CLI_FILE#${PROJECT_ROOT}/}"
echo "  - ${PYTHON_FILE#${PROJECT_ROOT}/}"
