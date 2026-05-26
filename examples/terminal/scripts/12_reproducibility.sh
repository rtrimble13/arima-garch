#!/usr/bin/env bash
# 12_reproducibility.sh — Confirm seed reproducibility: same seed ⇒ same first path.
source "$(dirname "$0")/../_common.sh"
banner "12 — Reproducibility check"

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

echo "--- Run #1, seed=12345 ---"
"$AG_BIN" simulate \
    --model  "$MODEL" \
    --paths  1 \
    --length 100 \
    --seed   12345 \
    --out    "$OUT_DIR/sim1.csv" > /dev/null

echo "--- Run #2, seed=12345 (should match) ---"
"$AG_BIN" simulate \
    --model  "$MODEL" \
    --paths  1 \
    --length 100 \
    --seed   12345 \
    --out    "$OUT_DIR/sim2.csv" > /dev/null

echo "--- Run #3, seed=99999 (should differ) ---"
"$AG_BIN" simulate \
    --model  "$MODEL" \
    --paths  1 \
    --length 100 \
    --seed   99999 \
    --out    "$OUT_DIR/sim3.csv" > /dev/null

echo
if diff -q "$OUT_DIR/sim1.csv" "$OUT_DIR/sim2.csv" > /dev/null; then
    echo "✅ sim1 == sim2 (same seed reproduces the path)"
else
    echo "❌ sim1 != sim2 — reproducibility broken!"
    exit 1
fi

if ! diff -q "$OUT_DIR/sim1.csv" "$OUT_DIR/sim3.csv" > /dev/null; then
    echo "✅ sim1 != sim3 (different seed → different realisation)"
else
    echo "❌ sim1 == sim3 — different seeds produced identical paths!"
    exit 1
fi
