# 06 — Selection (Cross-Validation)

## What this demonstrates

`--criterion CV` swaps the in-sample information criterion for **rolling-origin
cross-validation with 1-step-ahead MSE**. Each candidate is fitted on an
expanding window and scored by its out-of-sample 1-step forecast error,
making it a much more direct proxy for forecast accuracy than AIC/BIC.

The downside: cost. CV must refit every model on every fold, so it's slow.
This script intentionally uses the smallest grid (`p,q ∈ {0,1}`) so the demo
finishes in tens of seconds rather than minutes.

## Inputs

| File | Role |
|---|---|
| [../data/returns.csv](../data/returns.csv) | 1000 synthetic returns |

## Run it

```bash
./06_select_cv.sh
```

The `time` wrapper reports wall-clock so you can extrapolate to bigger grids.

## What to notice

- For 4 candidates × default folds the run takes ~20–30s on a modern laptop.
  Multiplying the search space by 10 multiplies the wall-clock by 10.
- CV picks the model with the lowest **out-of-sample** MSE, which is *not*
  guaranteed to match the BIC winner. If they disagree, prefer CV when your
  goal is forecasting and BIC when your goal is parsimonious explanation.
- For batch model selection, prototype with BIC, then validate the shortlist
  with CV.

## Try next

- Compare the winning spec here with the BIC winner in
  [04_select_bic.md](04_select_bic.md).
- See [docs/model_selection.md](../../../docs/model_selection.md) for the full
  selection-criteria comparison.
