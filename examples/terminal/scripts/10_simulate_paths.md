# 10 — Simulate paths from saved model (`--stats`)

## What this demonstrates

`ag simulate` is the **model-aware** sibling of `ag sim`. Instead of using
default coefficients, it loads parameters from a saved `model.json` and
generates `--paths N` independent realisations, each `--length L` long. With
`--stats`, it also prints aggregate moments across all paths plus the first
path's leading values (so you can confirm reproducibility).

This is the workhorse for **Monte Carlo VaR / Expected-Shortfall** workflows.

## Inputs

| File | Role |
|---|---|
| `var/01_fit_basic/model.json` | Fitted model from [01_fit_basic.sh](01_fit_basic.sh) |

If that model is missing the script fits one on the fly.

## Run it

```bash
./10_simulate_paths.sh        # 100 paths × 252 obs (one trading year each)
```

The output CSV columns are `path, observation, return, volatility`.

## Expected output (abridged)

```
=== Summary Statistics Across All Paths ===
Returns (aggregated over 100 paths):
  Mean:     -0.000310
  Std Dev:   0.025441
  Min:      -0.118437
  Max:       0.124902
  Skewness: -0.018721
  Kurtosis:  0.873452

First path statistics (for reproducibility check):
  First 5 returns: 0.012461 -0.005219 0.017802 -0.009041 ...
```

## What to notice

- **Sample skewness/kurtosis** approach the theoretical values of the
  innovation distribution as `N×L` grows. Gaussian → skew ≈ 0, ex-kurtosis ≈ 0.
- The `path` column makes it trivial to load the CSV into pandas and group
  by path for percentile fan charts.
- For deterministic results across machines, always pin `--seed`.

## Try next

- Drop `--stats` and load the CSV in Python/R for your own analysis.
- See [12_reproducibility.md](12_reproducibility.md) for a side-by-side seed
  check (same seed ⇒ identical first path).
- Bump `--paths 1000` for a richer Monte Carlo distribution (linear cost).
