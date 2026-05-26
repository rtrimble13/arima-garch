#!/usr/bin/env bash
# 03_fit_csv_options.sh — Demonstrate CSV input variants: --no-header and
# multi-column files (first column is used).
source "$(dirname "$0")/../_common.sh"
banner "03 — CSV input variants"

echo "--- Header-less CSV (--no-header) ---"
"$AG_BIN" fit \
    --data      "$DATA_DIR/returns_noheader.csv" \
    --no-header \
    --arima     1,0,1 \
    --garch     1,1 \
    --out       "$OUT_DIR/model_noheader.json" \
    | tee "$OUT_DIR/fit.noheader.txt" \
    | grep -E "Loaded|✅|AIC|BIC" | head -10

echo
echo "--- Multi-column CSV (returns,date,volume) — first column wins ---"
"$AG_BIN" fit \
    --data  "$DATA_DIR/returns_multicol.csv" \
    --arima 1,0,1 \
    --garch 1,1 \
    --out   "$OUT_DIR/model_multicol.json" \
    | tee "$OUT_DIR/fit.multicol.txt" \
    | grep -E "Loaded|✅|AIC|BIC" | head -10
