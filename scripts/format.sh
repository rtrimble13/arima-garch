#!/bin/bash
# Format all C++ source files in the project using clang-format

set -e

# Get the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null; then
  echo "Error: clang-format is not installed"
  echo "Please install clang-format (version 10 or higher recommended)"
  echo ""
  echo "Installation instructions:"
  echo "  Ubuntu/Debian: sudo apt-get install clang-format"
  echo "  macOS: brew install clang-format"
  echo "  Arch Linux: sudo pacman -S clang"
  exit 1
fi

# Print clang-format version
echo "Using clang-format version:"
clang-format --version

# Find all C++ source files and format them
echo ""
echo "Formatting C++ files in: ${PROJECT_ROOT}"
echo ""

# Find and format files
find "${PROJECT_ROOT}" \
  -type f \
  \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.cc" -o -name "*.cxx" -o -name "*.hxx" \) \
  -not -path "*/build/*" \
  -not -path "*/.*/*" \
  -print0 | while IFS= read -r -d '' file; do
    echo "Formatting: ${file#${PROJECT_ROOT}/}"
    clang-format -i -style=file "${file}"
  done

echo ""
echo "Formatting complete!"
