# 12 — Reproducibility check

## What this demonstrates

A small but critical property: `ag simulate` with the **same `--seed`** must
produce the **same path** on every run, and **different seeds** must produce
**different paths**. This guards against accidental non-determinism creeping
into the simulator (e.g. uninitialised state, thread-of-execution dependence).

## Run it

```bash
./12_reproducibility.sh
```

The script runs three simulations from the same saved model:

| Run | Seed | Expected |
|---|---|---|
| 1 | 12345 | reference |
| 2 | 12345 | identical to run 1 |
| 3 | 99999 | different from runs 1/2 |

It compares the CSV outputs with `diff -q` and asserts the expected equality
pattern. The script exits non-zero if any assertion fails.

## What to notice

- This is the **smoke test** behind every Monte Carlo run in
  [10_simulate_paths.md](10_simulate_paths.md): if reproducibility broke, MC
  results from yesterday would be uncomparable to today's.
- The same property holds for `ag sim` — try replacing the `simulate` calls
  with `sim` and the same assertions should still pass.

## Try next

- Verify the property survives across machines (commit a path produced on one
  box, regenerate it on another, `diff -q`).
- For research where you want **deliberate** variation, sweep `--seed` over a
  range and aggregate the resulting paths.
