# 01 — Fit basic (Gaussian)

## What this demonstrates

The most common entry point: fit an `ARIMA(1,0,1)-GARCH(1,1)` model to a daily-
return series, save the fitted parameters to JSON, and read the diagnostic
summary `ag` prints to the console.

## Inputs

| File | Role |
|---|---|
| [../data/returns.csv](../data/returns.csv) | 1000 synthetic daily returns (CSV with header) |

## Run it

```bash
./01_fit_basic.sh
```

The script invokes:

```bash
ag fit --data returns.csv --arima 1,0,1 --garch 1,1 --out model.json
```

and tees the human-readable report to `var/01_fit_basic/fit.console.txt`. The
saved JSON model is reused by later examples (forecast, diagnostics, simulate).

## Expected output (abridged)

```
Fitting ARIMA(1,0,1)-GARCH(1,1) model...
✅ Model fitted successfully
   Innovation dist.:   Normal (Gaussian)
   AIC:                -4666.04
   BIC:                -4641.50

   Ljung-Box (residuals)        p-value: 0.0280  ✗
   Ljung-Box (squared)          p-value: 0.0000  ✗
   Jarque-Bera                  p-value: 0.3691  ✓
   ADF                          p-value: 0.0010  ✓
```

## What to notice

- The full report includes **ARIMA** parameters (intercept, AR, MA), **GARCH**
  parameters (ω, α, β), AIC/BIC, and a battery of diagnostic tests.
- `ag` also prints a **distribution comparison**: it refits with Student-t and
  tells you whether the heavier-tailed assumption would have been a better fit.
  See [02_fit_student_t.md](02_fit_student_t.md) for the refit.
- A failing Ljung-Box on squared residuals usually means the GARCH order is too
  low — try `(2,1)` or `(1,2)` (see [04_select_bic.md](04_select_bic.md) for
  automatic selection).

## Try next

- Increase the ARIMA order: `--arima 2,0,2` and watch how AIC/BIC respond.
- Hand the saved `model.json` to [06_forecast.sh](06_forecast.sh) or
  [10_simulate_paths.sh](10_simulate_paths.sh).
