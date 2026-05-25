#!/usr/bin/env bash
# 05_select_aic_topk.sh — Selection with AIC criterion + print top-K ranking.
source "$(dirname "$0")/../_common.sh"
banner "05 — Selection (AIC + --top-k)"

"$AG_BIN" select \
    --data      "$DATA_DIR/returns.csv" \
    --max-p     2 \
    --max-d     0 \
    --max-q     2 \
    --max-garch-p 1 \
    --max-garch-q 1 \
    --criterion AIC \
    --top-k     5 \
    --out       "$OUT_DIR/best_aic.json" \
    | tee "$OUT_DIR/select.console.txt"
