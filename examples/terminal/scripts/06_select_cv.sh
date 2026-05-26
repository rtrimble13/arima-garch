#!/usr/bin/env bash
# 06_select_cv.sh — Selection by rolling-origin cross-validation (1-step MSE).
# CV is much slower than BIC; we keep the grid tiny on purpose.
source "$(dirname "$0")/../_common.sh"
banner "06 — Selection (Cross-Validation)"

echo "Note: CV is O(grid_size × CV_folds × fit_cost). Keeping the grid tiny."
echo

time "$AG_BIN" select \
    --data        "$DATA_DIR/returns.csv" \
    --max-p       1 \
    --max-d       0 \
    --max-q       1 \
    --max-garch-p 1 \
    --max-garch-q 1 \
    --criterion   CV \
    --out         "$OUT_DIR/best_cv.json" \
    | tee "$OUT_DIR/select.console.txt"
