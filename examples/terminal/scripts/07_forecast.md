# 07 — Forecast from saved model

## What this demonstrates

`ag forecast` takes a saved `model.json` and produces `h`-step-ahead mean and
variance forecasts. The CSV output (`step, mean, variance, std_dev`) is the
input format expected by `ag-viz forecast --plot` and most downstream tools.

## Inputs

| File | Role |
|---|---|
| `var/01_fit_basic/model.json` | Model produced by [01_fit_basic.sh](01_fit_basic.sh) |

If the model from `01_*` is missing the script fits one on the fly so this
example is runnable standalone.

## Run it

```bash
./07_forecast.sh           # uses 30-step horizon
```

## Expected output (abridged)

```
Step  Mean Forecast  Std Dev
----  -------------  -------
   1       0.000006  0.027362
   2       0.000013  0.027376
   3       0.000019  0.027390
   ...
  30       0.000180  0.027779
```

## What to notice

- **Mean forecasts** decay toward the unconditional mean (≈ intercept) — fast
  for low-AR-coefficient models, slow for near-unit-root ones.
- **Standard deviations** are produced by the GARCH process. For a stationary
  GARCH(1,1) they converge upward toward the unconditional vol
  `sqrt(ω / (1 − α − β))`.
- For confidence intervals around the mean path, multiply std-dev by your
  preferred z-score: `mean ± 1.96·std_dev` for 95 %.

## Try next

- Bump `--horizon 252` to project a full trading year and observe the
  convergence behaviour.
- Pipe the CSV into `ag-viz forecast --plot forecasts.csv` for a fan chart
  (requires the Python extras — see top-level [README](../../../README.md)).
