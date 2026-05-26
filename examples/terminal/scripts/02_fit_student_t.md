# 02 — Fit with Student-t innovations

## What this demonstrates

Financial returns frequently have heavier tails than the Normal distribution
predicts. The `--t-dist <df>` flag asks `ag fit` to assume Student-t innovations
with the given degrees of freedom (`5.0` is a common starting point — lower df
⇒ heavier tails).

## Inputs

| File | Role |
|---|---|
| [../data/returns.csv](../data/returns.csv) | 1000 synthetic returns |

## Run it

```bash
./02_fit_student_t.sh
```

This runs:

```bash
ag fit --data returns.csv --arima 1,0,1 --garch 1,1 --t-dist 5.0 \
       --out model_student_t.json
```

## What to notice

- The console output now reports `Innovation dist.: Student-t (df=5.00)`.
- The **log-likelihood goes up** versus the Gaussian fit in
  [01_fit_basic.md](01_fit_basic.md), but BIC penalises the extra parameters.
  For the synthetic data here, Gaussian usually wins; on real equity returns
  Student-t typically wins by a wide margin.
- Use this whenever the Gaussian distribution comparison printed by
  `ag fit` (without `--t-dist`) recommends Student-t.

## Try next

- Lower the degrees of freedom (`--t-dist 3.0`) to model fatter tails.
- Refit the model selected by [04_select_bic.sh](04_select_bic.sh) with
  Student-t and compare AIC/BIC.
