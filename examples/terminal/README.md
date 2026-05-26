# `ag` command-line examples

A runnable, walkthrough-style introduction to the `ag` CLI. Every numbered
script under [`scripts/`](scripts/) demonstrates one feature, has a matching
`.md` walkthrough next to it, and writes its output to [`var/`](var/)
(gitignored).

> Looking for the C++ library API? See the programs under
> [`examples/`](../) — `example_basic.cpp`, `example_simulation.cpp`,
> `example_json_io.cpp`, `example_model_selector.cpp`, etc.

## Layout

```
examples/terminal/
├── _common.sh        — shared helper sourced by every script
├── data/             — synthetic returns CSVs in three input flavours
├── scripts/          — 01_*.sh through 13_*.sh + run_all.sh + .md walkthroughs
└── var/              — generated output, one sub-directory per script (gitignored)
```

## Prerequisites

You need the `ag` binary on disk. Build it with one of:

```bash
# Make + Ninja wrapper (recommended — uses CMake presets)
make
# → build/ninja-release/src/ag

# Or direct CMake
cmake -S . -B build
cmake --build build --target ag
# → build/src/ag
```

The example scripts find `ag` automatically by checking, in order:

1. `$AG_BIN` (explicit override)
2. `examples/terminal/bin/ag` (staged by the `ag-examples` CMake target — see below)
3. `build/ninja-release/src/ag` (default Make + Ninja preset)
4. `build/ninja-relwithdebinfo/src/ag`
5. `build/ninja-debug/src/ag`
6. `build/src/ag` (direct CMake)
7. `command -v ag` (system-wide install)

### Optional: stage `ag` at a stable path

If you want the binary at a known location that survives clean builds or that
your shell scripts can reference without env vars:

```bash
cmake --build build/ninja-release --target ag-examples
# → examples/terminal/bin/ag
```

`ag-examples` is a custom CMake target in [`src/CMakeLists.txt`](../../src/CMakeLists.txt)
that depends on `ag` and copies it under `examples/terminal/bin/`. The staged
file is gitignored.

## Running

Run a single example:

```bash
cd examples/terminal/scripts
./01_fit_basic.sh
```

Or smoke-test the whole suite:

```bash
./run_all.sh
# Skip the slow cross-validation example:
SKIP_CV=1 ./run_all.sh
```

Each script writes everything it generates to `var/<script-name>/`. Re-running
overwrites that subdirectory.

## Example index

| # | Script | Subcommand(s) | Feature |
|---|---|---|---|
| 01 | [scripts/01_fit_basic.sh](scripts/01_fit_basic.sh) ([md](scripts/01_fit_basic.md)) | `fit`        | Baseline ARIMA-GARCH fit with Gaussian innovations |
| 02 | [scripts/02_fit_student_t.sh](scripts/02_fit_student_t.sh) ([md](scripts/02_fit_student_t.md)) | `fit --t-dist` | Heavy-tailed Student-t innovations |
| 03 | [scripts/03_fit_csv_options.sh](scripts/03_fit_csv_options.sh) ([md](scripts/03_fit_csv_options.md)) | `fit --no-header` | Header-less and multi-column CSV input |
| 04 | [scripts/04_select_bic.sh](scripts/04_select_bic.sh) ([md](scripts/04_select_bic.md)) | `select`     | Automatic model selection via BIC |
| 05 | [scripts/05_select_aic_topk.sh](scripts/05_select_aic_topk.sh) ([md](scripts/05_select_aic_topk.md)) | `select -c AIC --top-k` | AIC selection + top-K ranking |
| 06 | [scripts/06_select_cv.sh](scripts/06_select_cv.sh) ([md](scripts/06_select_cv.md)) | `select -c CV` | Cross-validation selection (slow) |
| 07 | [scripts/07_forecast.sh](scripts/07_forecast.sh) ([md](scripts/07_forecast.md)) | `forecast`   | h-step-ahead mean & variance forecasts |
| 08 | [scripts/08_sim_gaussian.sh](scripts/08_sim_gaussian.sh) ([md](scripts/08_sim_gaussian.md)) | `sim`        | Synthetic data with default-parameter Gaussian innovations |
| 09 | [scripts/09_sim_student_t.sh](scripts/09_sim_student_t.sh) ([md](scripts/09_sim_student_t.md)) | `sim --t-dist` | Synthetic data with heavy-tailed Student-t innovations |
| 10 | [scripts/10_simulate_paths.sh](scripts/10_simulate_paths.sh) ([md](scripts/10_simulate_paths.md)) | `simulate --stats` | Many Monte Carlo paths from a saved model |
| 11 | [scripts/11_diagnostics.sh](scripts/11_diagnostics.sh) ([md](scripts/11_diagnostics.md)) | `diagnostics` | Ljung-Box, Jarque-Bera, ADF tests on residuals |
| 12 | [scripts/12_reproducibility.sh](scripts/12_reproducibility.sh) ([md](scripts/12_reproducibility.md)) | `simulate -s` | Seed determinism check (same seed ⇒ same path) |
| 13 | [scripts/13_end_to_end.sh](scripts/13_end_to_end.sh) ([md](scripts/13_end_to_end.md)) | all          | Full `sim → select → forecast → diagnostics` pipeline |

## Data

| File | Description |
|---|---|
| [data/returns.csv](data/returns.csv) | 1000 synthetic daily returns, generated from an `ARIMA(1,0,1)-GARCH(1,1)` process with Gaussian innovations (seed 42); has a `returns` header row |
| [data/returns_noheader.csv](data/returns_noheader.csv) | Same series with the header stripped (for `--no-header`) |
| [data/returns_multicol.csv](data/returns_multicol.csv) | Same series in a `returns,date,volume` three-column file (the CLI uses the first column) |

The data is fully synthetic — no real market history is bundled — so the
examples are runnable anywhere with no external downloads.

## Visualization

These examples generate raw CSV/JSON. For publication-quality plots (forecast
fan charts, residual QQ-plots, ACF, simulation percentile bands) feed the
outputs into the `ag-viz` Python wrapper — see the
[Visualization section of the top-level README](../../README.md#visualization).
