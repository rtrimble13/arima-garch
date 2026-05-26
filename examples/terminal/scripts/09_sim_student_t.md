# 09 — Simulate with Student-t innovations

## What this demonstrates

The same `ag sim` command as [08_sim_gaussian.md](08_sim_gaussian.md) with one
extra flag: `--t-dist 4.0`. Innovations are now drawn from a Student-t(4)
distribution, which has materially heavier tails than the Normal — useful for
stress-testing risk models and generating worst-case scenarios.

## Inputs

None.

## Run it

```bash
./09_sim_student_t.sh
```

A tiny Python snippet at the end prints the empirical mean, std-dev, and the
largest absolute z-score in the simulated path so you can see how an extreme
event compares with a Gaussian world (where ~99.99 % of mass lies inside ±4 σ).

## What to notice

- For Student-t(4), the **kurtosis is theoretically infinite at df=4** (it's
  finite only for `df > 4`); in finite samples you'll routinely see 5–7 σ
  events that Gaussian innovations would essentially never produce.
- Pair this with `ag fit` to study how a misspecified Gaussian model fails on
  fat-tailed data.

## Try next

- Reduce `--t-dist` to `3.0` or `2.5` for truly pathological tails.
- Run [11_diagnostics.md](11_diagnostics.md) on a Gaussian-fitted model
  against this Student-t data and watch Jarque-Bera fail loudly.
