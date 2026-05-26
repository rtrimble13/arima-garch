#!/usr/bin/env bash
# 11_diagnostics.sh — Run diagnostic tests on a fitted model against the data.
source "$(dirname "$0")/../_common.sh"
banner "11 — Diagnostics"

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

"$AG_BIN" diagnostics \
    --model "$MODEL" \
    --data  "$DATA_DIR/returns.csv" \
    --out   "$OUT_DIR/diagnostics.json" \
    | tee "$OUT_DIR/diagnostics.console.txt"

echo
echo "--- $OUT_DIR/diagnostics.json (truncated) ---"
head -n 30 "$OUT_DIR/diagnostics.json"
