# shellcheck shell=bash
# Shared helpers for the terminal example scripts.
# Source this file from any example with:   source "$(dirname "$0")/../_common.sh"

set -euo pipefail

# Path layout — derived from the location of the *calling* script (BASH_SOURCE[1]),
# so every example resolves the same DATA_DIR / VAR_DIR regardless of cwd.
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
EX_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
PROJ_ROOT="$(cd "$EX_ROOT/../.." && pwd)"
DATA_DIR="$EX_ROOT/data"
VAR_DIR="$EX_ROOT/var"

mkdir -p "$VAR_DIR"

# Locate the `ag` binary. Search order:
#   1. $AG_BIN env var (explicit override)
#   2. examples/terminal/bin/ag        (staged by `cmake --build <dir> --target ag-examples`)
#   3. build/ninja-release/src/ag      (default Make + Ninja preset)
#   4. build/ninja-relwithdebinfo/src/ag
#   5. build/ninja-debug/src/ag
#   6. build/src/ag                    (direct CMake, no preset)
#   7. `command -v ag`                 (installed system-wide)
locate_ag() {
    if [[ -n "${AG_BIN:-}" ]]; then
        if [[ -x "$AG_BIN" ]]; then echo "$AG_BIN"; return; fi
        echo "AG_BIN is set to '$AG_BIN' but that path is not executable." >&2
        return 1
    fi
    local candidates=(
        "$EX_ROOT/bin/ag"
        "$PROJ_ROOT/build/ninja-release/src/ag"
        "$PROJ_ROOT/build/ninja-relwithdebinfo/src/ag"
        "$PROJ_ROOT/build/ninja-debug/src/ag"
        "$PROJ_ROOT/build/src/ag"
    )
    for c in "${candidates[@]}"; do
        if [[ -x "$c" ]]; then echo "$c"; return; fi
    done
    if command -v ag >/dev/null 2>&1; then
        command -v ag
        return
    fi
    cat >&2 <<'EOF'
ERROR: could not find the `ag` binary.

Build it with one of:
    make                                            # Make + Ninja (preferred)
    cmake -S . -B build && cmake --build build      # direct CMake

Then re-run, or stage a stable copy with:
    cmake --build build/ninja-release --target ag-examples

…or set AG_BIN to an explicit path before running the script.
EOF
    return 1
}

AG_BIN="$(locate_ag)"
export AG_BIN

# Per-script output directory: var/<calling-script-name-without-.sh>/
SCRIPT_STEM="$(basename "${BASH_SOURCE[1]}" .sh)"
OUT_DIR="$VAR_DIR/$SCRIPT_STEM"
mkdir -p "$OUT_DIR"

# Pretty header.
banner() {
    local title="$1"
    printf '\n==== %s ====\n' "$title"
    printf 'ag binary : %s\n' "$AG_BIN"
    printf 'output dir: %s\n\n' "$OUT_DIR"
}
