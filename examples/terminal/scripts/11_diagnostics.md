# 11 — Diagnostics

## What this demonstrates

After fitting, you need to ask: *is this model actually good?* `ag diagnostics`
re-loads the saved model, re-applies it to the data, computes the standardised
residuals, and runs four classical mis-specification tests:

| Test | What it checks | Pass if p-value … |
|---|---|---|
| Ljung-Box on residuals | Remaining autocorrelation in the mean equation | > 0.05 |
| Ljung-Box on squared residuals | Remaining ARCH effects (volatility clustering) | > 0.05 |
| Jarque-Bera | Normality of innovations | > 0.05 (or refit with `--t-dist`) |
| Augmented Dickey-Fuller | Residual stationarity | < 0.05 (we *want* rejection of unit root) |

## Inputs

| File | Role |
|---|---|
| `var/01_fit_basic/model.json` | Model from [01_fit_basic.sh](01_fit_basic.sh) |
| [../data/returns.csv](../data/returns.csv) | Same series used for fitting |

## Run it

```bash
./11_diagnostics.sh
```

The console report is human-readable; the JSON output (`diagnostics.json`)
contains the raw statistics and p-values for downstream tooling.

## What to notice

- A failing Ljung-Box on **squared** residuals is the most common red flag:
  it means GARCH(`p,q`) isn't capturing all the volatility clustering — try
  `(2,1)` / `(1,2)` or let `ag select` find a better fit.
- Jarque-Bera failure on real financial data is normal (heavy tails) — that's
  the signal to refit with `ag fit --t-dist`.
- ADF failure would be alarming — it suggests the residuals aren't stationary,
  meaning the model has fundamentally mis-specified the mean equation.

## Try next

- After refitting with Student-t ([02_fit_student_t.sh](02_fit_student_t.sh)),
  re-run diagnostics against `model_student_t.json` and watch the Jarque-Bera
  p-value behave better.
- The JSON output is consumable by `ag-viz diagnostics --plot` (Python wrapper).
