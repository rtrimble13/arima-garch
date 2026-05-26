# 13 — End-to-end workflow

## What this demonstrates

A complete pipeline gluing the four core subcommands together:

```
sim ─► (strip volatility col) ─► select ─► forecast
                                        └► diagnostics
```

The example simulates ground-truth data, hides the true generating spec from
`ag select`, then asks selection to recover a reasonable model. With ARIMA
recovery on a known DGP you can sanity-check the whole stack.

## Run it

```bash
./13_end_to_end.sh
```

Each step's output is interleaved with section headers so you can follow the
flow. All artifacts land in `var/13_end_to_end/`:

| File | Produced by |
|---|---|
| `synthetic.csv` | `ag sim` (observation, return, volatility) |
| `synthetic_returns.csv` | One-column slice for the fitter |
| `best.json` | `ag select --criterion BIC` |
| `forecast.csv` | `ag forecast` |
| `diagnostics.json` | `ag diagnostics` |

## What to notice

- **`sim` writes 3 columns** (`observation, return, volatility`) but the
  fitter expects a single returns column — the inline `awk` strip is a useful
  pattern when chaining the two.
- **Selection rarely picks back the exact true spec** unless the sample is
  huge — that's a property of finite-sample model selection, not a bug.
- The diagnostics on synthetic data should mostly **pass** (the model is
  correctly specified by construction). Compare with the diagnostics on real
  data in [11_diagnostics.md](11_diagnostics.md), which usually shows leftover
  ARCH/heavy tails.

## Try next

- Swap `--criterion BIC` for `CV` (slow!) to see if cross-validation picks a
  different winner.
- Run [12_reproducibility.sh](12_reproducibility.sh) on `best.json` to confirm
  the simulator path is deterministic.
- Wrap the whole thing in a loop over seeds and aggregate diagnostic p-values
  for a small Monte Carlo recovery study.
