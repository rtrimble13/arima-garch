#!/usr/bin/env bash
# 04_select_bic.sh — Automatic model selection using BIC over a small grid.
source "$(dirname "$0")/../_common.sh"
banner "04 — Automatic selection (BIC)"

"$AG_BIN" select \
    --data      "$DATA_DIR/returns.csv" \
    --max-p     2 \
    --max-d     0 \
    --max-q     2 \
    --max-garch-p 1 \
    --max-garch-q 1 \
    --criterion BIC \
    --out       "$OUT_DIR/best_model.json" \
    | tee "$OUT_DIR/select.console.txt"
