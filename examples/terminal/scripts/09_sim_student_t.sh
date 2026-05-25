#!/usr/bin/env bash
# 09_sim_student_t.sh — Simulate with Student-t(df=4) innovations (heavy tails).
source "$(dirname "$0")/../_common.sh"
banner "09 — Simulate with Student-t innovations"

"$AG_BIN" sim \
    --arima  1,0,1 \
    --garch  1,1 \
    --length 500 \
    --seed   42 \
    --t-dist 4.0 \
    --out    "$OUT_DIR/synthetic_t.csv"

echo
echo "--- First 6 rows ---"
head -n 6 "$OUT_DIR/synthetic_t.csv"

# Crude min/max check: heavy-tailed series should occasionally throw an extreme.
python3 - "$OUT_DIR/synthetic_t.csv" <<'PY'
import csv, sys, math
with open(sys.argv[1]) as f:
    r = [float(row['return']) for row in csv.DictReader(f)]
mean = sum(r) / len(r)
sd = (sum((x - mean) ** 2 for x in r) / len(r)) ** 0.5
print(f"n={len(r)} mean={mean:.5f} sd={sd:.5f} min={min(r):.4f} max={max(r):.4f}")
print(f"max |z| = {max(abs(x - mean) for x in r) / sd:.2f}σ")
PY
