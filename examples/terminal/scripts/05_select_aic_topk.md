# 05 — Selection (AIC + `--top-k`)

## What this demonstrates

Two refinements to model selection:

1. `--criterion AIC` — use Akaike's Information Criterion instead of BIC. AIC
   penalises complexity less aggressively (`2k` vs BIC's `k·log(n)`), so it
   tends to choose slightly larger models that fit better in-sample.
2. `--top-k 5` — instead of only printing the winner, list the **5 best
   candidates** sorted by the chosen criterion. Useful for spotting near-ties.

## Inputs

| File | Role |
|---|---|
| [../data/returns.csv](../data/returns.csv) | 1000 synthetic returns |

## Run it

```bash
./05_select_aic_topk.sh
```

## Expected output (abridged)

```
=== Model Ranking (Top 5) ===
Rank   Model                       AIC          Converged
-------------------------------------------------------------
1      ARIMA(0,0,0)-GARCH(1,1)     -6552.33     Yes
2      ARIMA(1,0,0)-GARCH(1,1)     -6550.30     Yes
3      ARIMA(0,0,1)-GARCH(1,1)     -6550.05     Yes
4      ARIMA(1,0,1)-GARCH(1,1)     -6549.10     Yes
5      ARIMA(2,0,0)-GARCH(1,1)     -6548.21     Yes
```

## What to notice

- The top entries cluster within a few AIC units of each other — when models
  are this close, parsimony and forecast performance should drive the choice,
  not raw criterion value.
- The criterion `AICc` (small-sample-corrected AIC) is also accepted; for
  large samples (n ≫ k) it coincides with AIC.
- The candidate-grid sweep skips models that fail to converge — they appear
  in the `Candidates failed:` count, not the ranking.

## Try next

- Re-run with `--criterion BIC` (the default — see
  [04_select_bic.md](04_select_bic.md)) and confirm the ranking shifts toward
  smaller models.
- For a more forecast-oriented criterion, use
  [06_select_cv.md](06_select_cv.md).
