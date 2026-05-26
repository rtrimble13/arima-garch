#!/usr/bin/env bash
# 01_fit_basic.sh — Baseline fit: ARIMA(1,0,1)-GARCH(1,1) with Gaussian innovations.
source "$(dirname "$0")/../_common.sh"
banner "01 — Fit basic (Gaussian)"

# Fit and save the model JSON for use by later examples.
"$AG_BIN" fit \
    --data  "$DATA_DIR/returns.csv" \
    --arima 1,0,1 \
    --garch 1,1 \
    --out   "$OUT_DIR/model.json" \
    | tee "$OUT_DIR/fit.console.txt"
