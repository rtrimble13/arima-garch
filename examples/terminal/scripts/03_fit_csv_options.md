# 03 — CSV input variants

## What this demonstrates

`ag` reads CSV time-series files in two flavours:

1. **Header-less** — pass `--no-header` and the very first row is treated as
   data, not a label.
2. **Multi-column** — the parser unconditionally uses the **first column** of
   every row, so `returns,date,volume` works without any pre-processing.

Both calls fit the same `ARIMA(1,0,1)-GARCH(1,1)` model so the AIC/BIC values
should match (they're the same underlying data).

## Inputs

| File | Role |
|---|---|
| [../data/returns_noheader.csv](../data/returns_noheader.csv) | Same 1000 returns, **no** header row |
| [../data/returns_multicol.csv](../data/returns_multicol.csv) | `returns,date,volume` — three columns |

## Run it

```bash
./03_fit_csv_options.sh
```

## What to notice

- Without `--no-header`, the first row would be parsed as a number and either
  fail or skew the model — always tell `ag` whether your file has a header.
- The multi-column read silently drops everything after column 0. Put the
  series you want fit first, or pre-slice the CSV. (If you have a date column
  and want it preserved in plots, use the `ag-viz` Python wrapper — the C++
  CLI is index-based.)
- The two outputs (`model_noheader.json` and `model_multicol.json`) have
  identical parameter values up to numerical precision.

## Try next

- Point `ag` at your own returns/log-returns/log-price-differences CSV. As long
  as it's one column of numbers, it will fit.
- See [docs/file_formats.md](../../../docs/file_formats.md) for the full CSV
  spec.
