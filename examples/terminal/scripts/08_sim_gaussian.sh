#!/usr/bin/env bash
# 08_sim_gaussian.sh — Simulate synthetic data from default-parameter ARIMA-GARCH.
source "$(dirname "$0")/../_common.sh"
banner "08 — Simulate synthetic series (Gaussian)"

"$AG_BIN" sim \
    --arima  1,0,1 \
    --garch  1,1 \
    --length 500 \
    --seed   42 \
    --out    "$OUT_DIR/synthetic.csv"

echo
echo "--- First 6 rows ---"
head -n 6 "$OUT_DIR/synthetic.csv"
echo "..."
wc -l "$OUT_DIR/synthetic.csv"
