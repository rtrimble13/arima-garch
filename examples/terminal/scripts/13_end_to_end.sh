#!/usr/bin/env bash
# 13_end_to_end.sh — Full sim → fit → forecast → diagnostics workflow.
source "$(dirname "$0")/../_common.sh"
banner "13 — End-to-end workflow"

echo "### Step 1: simulate fresh synthetic returns (sim → ground-truth data)"
"$AG_BIN" sim \
    --arima  1,0,1 \
    --garch  1,1 \
    --length 1000 \
    --seed   2024 \
    --out    "$OUT_DIR/synthetic.csv"

# `ag sim` writes observation,return,volatility — strip to a single returns
# column so the fitter has the right CSV shape.
echo "returns" > "$OUT_DIR/synthetic_returns.csv"
tail -n +2 "$OUT_DIR/synthetic.csv" \
    | awk -F, '{print $2}' >> "$OUT_DIR/synthetic_returns.csv"

echo
echo "### Step 2: let select find the best spec (BIC)"
"$AG_BIN" select \
    --data        "$OUT_DIR/synthetic_returns.csv" \
    --max-p       2 --max-d 0 --max-q 2 \
    --max-garch-p 1 --max-garch-q 1 \
    --criterion   BIC \
    --out         "$OUT_DIR/best.json" \
    | grep -E "Best model|AIC|BIC|Candidates" | head -6

echo
echo "### Step 3: forecast 20 steps from the selected model"
"$AG_BIN" forecast \
    --model   "$OUT_DIR/best.json" \
    --horizon 20 \
    --out     "$OUT_DIR/forecast.csv" \
    | tail -n 25

echo
echo "### Step 4: diagnostics on the fitted model"
"$AG_BIN" diagnostics \
    --model "$OUT_DIR/best.json" \
    --data  "$OUT_DIR/synthetic_returns.csv" \
    --out   "$OUT_DIR/diagnostics.json" \
    | grep -E "Ljung-Box|Jarque-Bera|Dickey-Fuller|Statistic|P-value" | head -20

echo
echo "Artifacts written to: $OUT_DIR"
ls -1 "$OUT_DIR"
