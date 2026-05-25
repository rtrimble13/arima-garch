#!/usr/bin/env bash
# 10_simulate_paths.sh — Many simulation paths from a fitted model (Monte Carlo).
source "$(dirname "$0")/../_common.sh"
banner "10 — Simulate paths from saved model (--stats)"

MODEL="$VAR_DIR/01_fit_basic/model.json"
if [[ ! -f "$MODEL" ]]; then
    echo "Model $MODEL not found — fitting one now..."
    "$AG_BIN" fit \
        --data  "$DATA_DIR/returns.csv" \
        --arima 1,0,1 \
        --garch 1,1 \
        --out   "$OUT_DIR/model.json" > /dev/null
    MODEL="$OUT_DIR/model.json"
fi

# 100 paths × 252 obs (one trading year per path) — Monte Carlo for VaR work.
"$AG_BIN" simulate \
    --model  "$MODEL" \
    --paths  100 \
    --length 252 \
    --seed   42 \
    --out    "$OUT_DIR/monte_carlo.csv" \
    --stats \
    | tee "$OUT_DIR/simulate.console.txt"

echo
echo "--- First 6 rows of $OUT_DIR/monte_carlo.csv ---"
head -n 6 "$OUT_DIR/monte_carlo.csv"
echo "..."
wc -l "$OUT_DIR/monte_carlo.csv"
