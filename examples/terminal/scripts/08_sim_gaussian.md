# 08 — Simulate synthetic series (Gaussian)

## What this demonstrates

`ag sim` generates synthetic ARIMA-GARCH data without needing a fitted model.
It uses default parameters baked into the simulator and writes a tidy CSV with
columns `observation, return, volatility`.

## Inputs

None — `ag sim` is parameter-driven, not data-driven.

## Run it

```bash
./08_sim_gaussian.sh
```

This produces 500 simulated observations with seed `42` from a default
`ARIMA(1,0,1)-GARCH(1,1)` spec, Gaussian innovations.

## Expected output (abridged)

```csv
observation,return,volatility
1,-0.0152,0.0980
2, 0.0312,0.0984
3,-0.0067,0.0991
...
```

## What to notice

- The **volatility** column is the conditional standard deviation σₜ that
  produced each observation — handy for sanity-checking GARCH clustering.
- The seed makes runs reproducible: re-running with `--seed 42` gives the
  exact same series.
- `ag sim` uses fixed default coefficients; to simulate from *your* fitted
  parameters use [10_simulate_paths.sh](10_simulate_paths.sh) (`ag simulate`,
  the model-aware variant).

## Try next

- See [09_sim_student_t.md](09_sim_student_t.md) for heavy-tailed innovations.
- Feed the synthetic CSV back through `ag fit` to verify parameter recovery
  (the [13_end_to_end.md](13_end_to_end.md) walk-through does exactly this).
