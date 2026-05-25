# 04 — Automatic selection (BIC)

## What this demonstrates

Rather than guessing `(p,d,q,P,Q)` by hand, `ag select` enumerates a grid of
candidates, fits each one, and ranks them. With `--criterion BIC` (the default)
selection is based on the Bayesian Information Criterion — a fast, parsimonious
choice for medium-sized data.

## Inputs

| File | Role |
|---|---|
| [../data/returns.csv](../data/returns.csv) | 1000 synthetic returns |

## Run it

```bash
./04_select_bic.sh
```

This searches `p ∈ {0,1,2}`, `d = 0`, `q ∈ {0,1,2}`, GARCH `(1,1)` — i.e. 9
candidate ARIMA specs × 1 GARCH spec = 9 models — and saves the winner to
`best_model.json`.

## Expected output (abridged)

```
Generated 9 candidate models
Performing model selection using BIC...
✅ Model selection completed
Best model: ARIMA(0,0,0)-GARCH(1,1)
Candidates evaluated: 9
Candidates failed: 0
AIC: -6552.33
BIC: -6537.60

   Innovation Distribution Comparison
     ✓ Gaussian distribution is adequate for this data
```

## What to notice

- **BIC penalises complexity** more harshly than AIC (the penalty is
  `k·log(n)` vs `2k`), so it tends to pick the smallest model that explains the
  data — here, the pure GARCH(1,1) with a constant mean.
- The post-selection **distribution comparison** still runs against
  Student-t — if it recommends switching, refit with `ag fit --t-dist` as in
  [02_fit_student_t.md](02_fit_student_t.md).
- `--max-d > 0` lets the search consider differencing for non-stationary
  series. Pure return series like ours don't need it.

## Try next

- See [05_select_aic_topk.md](05_select_aic_topk.md) for an AIC-driven search
  with the full ranking table printed.
- See [06_select_cv.md](06_select_cv.md) for cross-validation-based selection
  (slower, better forecast accuracy proxy).
