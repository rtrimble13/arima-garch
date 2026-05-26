#!/usr/bin/env bash
# 07_forecast.sh — Generate h-step-ahead forecasts from a saved model.
source "$(dirname "$0")/../_common.sh"
banner "07 — Forecast from saved model"

# Use the model fit in 01 (run 01_fit_basic.sh first, or this will fall back
# to fitting on the fly).
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

"$AG_BIN" forecast \
    --model   "$MODEL" \
    --horizon 30 \
    --out     "$OUT_DIR/forecasts.csv" \
    | tee "$OUT_DIR/forecast.console.txt"

echo
echo "--- First 6 rows of $OUT_DIR/forecasts.csv ---"
head -n 6 "$OUT_DIR/forecasts.csv"
