#!/usr/bin/env bash
# 02_fit_student_t.sh — Fit with Student-t innovations to capture heavy tails.
source "$(dirname "$0")/../_common.sh"
banner "02 — Fit with Student-t innovations"

"$AG_BIN" fit \
    --data   "$DATA_DIR/returns.csv" \
    --arima  1,0,1 \
    --garch  1,1 \
    --t-dist 5.0 \
    --out    "$OUT_DIR/model_student_t.json" \
    | tee "$OUT_DIR/fit.console.txt"
