#!/bin/bash
# Check if all C++ source files are properly formatted with clang-format
# This script is intended for use in CI to verify code formatting

set -e

# Get the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null; then
  echo "Error: clang-format is not installed"
  echo "Please install clang-format (version 10 or higher recommended)"
  exit 1
fi

# Print clang-format version
echo "Using clang-format version:"
clang-format --version

echo ""
echo "Checking C++ file formatting in: ${PROJECT_ROOT}"
echo ""

# Track if any files are not formatted correctly
FORMAT_ISSUES=0

# Find all C++ source files and check their formatting
while IFS= read -r -d '' file; do
  # Check if file would be changed by clang-format
  if ! clang-format -style=file --dry-run --Werror "${file}" 2>/dev/null; then
    echo "❌ Formatting issues found in: ${file#${PROJECT_ROOT}/}"
    FORMAT_ISSUES=$((FORMAT_ISSUES + 1))
  fi
done < <(find "${PROJECT_ROOT}" \
  -type f \
  \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.cc" -o -name "*.cxx" -o -name "*.hxx" \) \
  -not -path "*/build/*" \
  -not -path "*/_codeql_build_dir/*" \
  -not -path "*/.*/*" \
  -print0)

# Check if we found any formatting issues
if [ ${FORMAT_ISSUES} -ne 0 ]; then
  echo ""
  echo "❌ Formatting check failed!"
  echo "Found ${FORMAT_ISSUES} file(s) with formatting issues."
  echo ""
  echo "To fix formatting, run:"
  echo "  ./scripts/format.sh"
  exit 1
fi

echo ""
echo "✅ All files are properly formatted!"
exit 0
