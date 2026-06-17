# Project Review — arima-garch — 2026-06-17

## Verdict & summary

The project is, on the whole, in good shape: a clean layered C++20 architecture
(`data → models → estimation → forecasting/selection/diagnostics → api → cli`),
an `expected<T,E>` error-handling discipline at the IO/API boundaries, consistent
input validation, and broad unit-test coverage (29 suites). The Python `ag-viz`
wrapper is genuinely solid.

The headline concern is a **single high-impact correctness defect that runs
through several modules**: ARIMA differencing (`d > 0`) is applied during fitting
but **never inverted** during forecasting, diagnostics, or state warm-up, so every
result for an integrated model is silently on the wrong scale. Because `select`
defaults to `--max-d 1`, this is on the default user path, and the entire `d > 0`
path is untested. A closely related defect is that `loadModel` discards the saved
model state, so the CLI `fit → forecast` workflow forecasts from zero history.
Together these two undermine the project's core forecasting promise and should be
fixed first.

Beyond those, the statistics layer has a cluster of real but bounded defects (ADF
auto lag selection is effectively a constant; ADF critical-value adjustment is a
no-op that contradicts its comment; the distribution LR test can select Student-t
when it fits worse), plus the usual maintainability wins (triplicated chi-square
code, a bespoke matrix inverse that duplicates `ag::util`). None of these are
five-alarm fires, but the d-differencing bug is.

All findings below are filed as GitHub issues (labels `project-review`, tier,
lens, and `severity:*` for defects).

## How this review was scoped

- **Languages/frameworks:** C++20 (CMake + Ninja, CLI11, nlohmann/json, fmt,
  Catch2) for the library/CLI; Python (Click, matplotlib, pandas) for `ag-viz`.
  Git history was available and used for churn analysis.
- **Read in full and traced:** the estimation core (`Likelihood`, `Optimizer`,
  `FitDriver`), the model core (`ArimaModel`/`ArimaState`, `GarchModel`/`GarchState`,
  composite `ArimaGarchModel`), `Forecaster`, the IO trust boundary (`CsvReader`,
  `Json`), `api/Engine`, `cli/main`, and the Python `cli.py`/`utils.py`. The
  d-differencing data flow was traced across fit → forecast → diagnostics →
  Engine warm-up and confirmed by three independent passes.
- **Sampled:** the stats/selection/diagnostics modules (`ADF`, `ACF`/`PACF`,
  `Bootstrap`, `JarqueBera`, `LjungBox`, `DistributionSelector`,
  `InformationCriteria`, `CrossValidation`, `Residuals`) — read closely around
  suspected defects, sampled elsewhere; the CLI handlers, the build system, and
  the remaining Python modules (`plotting.py`, `markdown_reports.py`, `io.py`).
- **Skipped:** the test suites themselves (read for intent/coverage signals only),
  example programs, benchmarks, vendored/fetched dependencies, and docs prose.
- **Note:** the four-way subagent fan-out for stats/forecasting completed; the
  estimation, IO/CLI, and Python passes were done by direct reading after transient
  capacity errors, so those areas reflect first-hand reads rather than delegated
  summaries.

## Findings

Ordered by priority tier (P0 → P3), then severity within tier.

### [P0] ARIMA differencing (d>0) is never inverted — [#128](https://github.com/rtrimble13/arima-garch/issues/128)
- **Lens:** Hidden bug
- **Priority:** P0 · **Impact:** High · **Effort:** High · **Severity:** Critical · **Confidence:** High
- **Evidence:** `src/models/arima/ArimaModel.cpp:36-40` (fit differences) vs `src/models/composite/ArimaGarchModel.cpp:52-86`, `src/forecasting/Forecaster.cpp:40-87`, `src/api/Engine.cpp:61-63`, `src/diagnostics/Residuals.cpp:32-71`
- **Why it matters:** Every forecast/diagnostic for an integrated model applies differenced-scale coefficients to raw levels and never re-integrates; `select` defaults to `--max-d 1` and the d>0 path is untested.
- **Recommendation:** Make the composite model d-aware (difference in `update`, integrate forecasts back to levels via the last d anchors); add d=1 end-to-end tests.

### [P0] loadModel discards saved model state — [#129](https://github.com/rtrimble13/arima-garch/issues/129)
- **Lens:** Hidden bug
- **Priority:** P0 · **Impact:** High · **Effort:** Medium · **Severity:** High · **Confidence:** High
- **Evidence:** `src/io/Json.cpp:380-390` (state not restored), `src/io/Json.cpp:258-317` (dead deserialization helpers), `src/forecasting/Forecaster.cpp:32-37`
- **Why it matters:** `ag fit -o model.json` then `ag forecast -m model.json` forecasts from zero history ≈ intercept/omega only, contradicting the README's reproducibility claim.
- **Recommendation:** Restore state in `loadModel` (wire up the existing helpers + a `restoreState`), remove the dead code.

### [P1] ADF auto lag selection is broken — [#130](https://github.com/rtrimble13/arima-garch/issues/130)
- **Lens:** Hidden bug
- **Priority:** P1 · **Impact:** High · **Effort:** Medium · **Severity:** High · **Confidence:** High
- **Evidence:** `src/stats/ADF.cpp:222-275` (builds `X`, then scores IC off `Var(Δy)`)
- **Why it matters:** IC reduces to always preferring lag 0, so `adf_test_auto` runs an under-specified regression and biases the unit-root decision.
- **Recommendation:** Fit the OLS regression per lag (reuse `ag::util`) and use the real RSS.

### [P1] Likelihood does not guard NaN/Inf — [#131](https://github.com/rtrimble13/arima-garch/issues/131)
- **Lens:** Robustness
- **Priority:** P1 · **Impact:** Medium · **Effort:** Low · **Severity:** Medium · **Confidence:** Medium
- **Evidence:** `src/estimation/Likelihood.cpp:50,70`; recursions lack `isfinite` checks (`ArimaModel.cpp:47-57`, `GarchModel.cpp:128-145`)
- **Why it matters:** A NaN `h_t` passes the `<=0` test, NaN NLL is accepted by Nelder–Mead as "not worse", and the fit can report `converged` with garbage parameters.
- **Recommendation:** Return a large finite penalty on non-finite NLL (FitDriver already maps to `CONSTRAINT_PENALTY`); validate input finiteness once.

### [P1] Triplicated gamma/chi-square special functions — [#132](https://github.com/rtrimble13/arima-garch/issues/132)
- **Lens:** Refactoring
- **Priority:** P1 · **Impact:** Medium · **Effort:** Low · **Severity:** — · **Confidence:** High
- **Evidence:** `src/stats/JarqueBera.cpp:15-121`, `src/stats/LjungBox.cpp:15-121`, `src/selection/DistributionSelector.cpp:19-117`
- **Why it matters:** Three copies that have already drifted (`M_PI` vs `std::numbers::pi`); a `1 - chi_square_cdf` double-negation round-trip exists.
- **Recommendation:** Extract one shared `ag::stats` special-functions header; drop `chi_square_cdf` in favor of `ccdf`.

### [P2] ADF adjust_critical_value is a no-op / misleading comment — [#133](https://github.com/rtrimble13/arima-garch/issues/133)
- **Lens:** Hidden bug
- **Priority:** P2 · **Impact:** Medium · **Effort:** Medium · **Severity:** Medium · **Confidence:** High
- **Evidence:** `src/stats/ADF.cpp:41-57`
- **Why it matters:** Critical values are fixed at the n=100 row for 25<n<500 with a discontinuity at 500; comment claims interpolation that doesn't happen.
- **Recommendation:** Implement the MacKinnon n-dependent surface, or document fixed-n and remove the dead branches.

### [P2] compareDistributions LR can be negative → spurious Student-t — [#134](https://github.com/rtrimble13/arima-garch/issues/134)
- **Lens:** Hidden bug
- **Priority:** P2 · **Impact:** Medium · **Effort:** Low · **Severity:** Medium · **Confidence:** Medium
- **Evidence:** `src/selection/DistributionSelector.cpp:225-226,242`
- **Why it matters:** A negative LR yields `lr_p_value = 0` and selects Student-t exactly when it fits worse than Normal.
- **Recommendation:** Clamp `lr_stat ≥ 0`; guard the `prefer_student_t` decision.

### [P2] estimateStudentTDF returns bracket midpoint, not argmin — [#135](https://github.com/rtrimble13/arima-garch/issues/135)
- **Lens:** Robustness
- **Priority:** P2 · **Impact:** Medium · **Effort:** Low · **Severity:** Medium · **Confidence:** Medium
- **Evidence:** `src/selection/DistributionSelector.cpp:158-185`
- **Why it matters:** Returned df is arbitrary on flat (near-Gaussian) surfaces and wrong at bracket boundaries, feeding the LR test and AIC/BIC.
- **Recommendation:** Return the tracked argmin; short-circuit on flat surfaces.

### [P2] CsvReader leading-blank-line breaks header detection — [#136](https://github.com/rtrimble13/arima-garch/issues/136)
- **Lens:** Hidden bug
- **Priority:** P2 · **Impact:** Medium · **Effort:** Low · **Severity:** Medium · **Confidence:** High
- **Evidence:** `src/io/CsvReader.cpp:122-147` (header check `line_number == 1`), error line math at `:205,215,253`
- **Why it matters:** A leading blank line makes the header parse as data; error line numbers drift when blank lines are interspersed.
- **Recommendation:** Detect the header on the first non-empty line; track physical line numbers for errors.

### [P2] Bootstrap duplicates OLS/matrix inverse already in ag::util — [#137](https://github.com/rtrimble13/arima-garch/issues/137)
- **Lens:** Refactoring
- **Priority:** P2 · **Impact:** Medium · **Effort:** Medium · **Severity:** — · **Confidence:** High
- **Evidence:** `src/stats/Bootstrap.cpp:277-443` vs `src/stats/ADF.cpp:125-181`; trend column `data_idx+1` vs `t`
- **Why it matters:** Two ADF t-stat implementations with different numerics; observed vs bootstrap stats diverge.
- **Recommendation:** Extract one OLS t-stat helper; align the trend convention.

### [P2] ACF/PACF O(n·lag) recomputed per bootstrap; no FFT — [#138](https://github.com/rtrimble13/arima-garch/issues/138)
- **Lens:** Enhancement (performance)
- **Priority:** P2 · **Impact:** Medium · **Effort:** Medium · **Severity:** — · **Confidence:** High
- **Evidence:** `src/stats/ACF.cpp:90-96`, `src/stats/Bootstrap.cpp:483-494`
- **Why it matters:** O(B·n·lags) bootstrap diagnostics scale poorly on long series.
- **Recommendation:** FFT-based autocovariance (O(n log n)); cache the demeaned series.

### [P2] Residuals const_cast mutates state, duplicating update() — [#139](https://github.com/rtrimble13/arima-garch/issues/139)
- **Lens:** Refactoring
- **Priority:** P2 · **Impact:** Medium · **Effort:** Low · **Severity:** — · **Confidence:** High
- **Evidence:** `src/diagnostics/Residuals.cpp:60-65` vs `src/models/composite/ArimaGarchModel.cpp:62-69`
- **Why it matters:** Two copies of update semantics (a liability for the differencing fix); violates the const contract.
- **Recommendation:** Add a public `stepDiagnostic(y_t)` to the model and call it.

### [P2] ArimaParameters lacks stationarity/invertibility check — [#140](https://github.com/rtrimble13/arima-garch/issues/140)
- **Lens:** Robustness / Enhancement
- **Priority:** P2 · **Impact:** Medium · **Effort:** Medium · **Severity:** Medium · **Confidence:** Medium
- **Evidence:** `src/models/garch/GarchModel.cpp:43-53` (GARCH has one); `include/ag/models/arima/ArimaModel.hpp` (ARIMA does not)
- **Why it matters:** Explosive AR fits diverge to NaN with no actionable diagnostic and no optimizer steering.
- **Recommendation:** Add `isStationary()/isInvertible()`; surface in the fit summary, optionally enforce.

### [P3] CsvReader has no quoted-field support — [#141](https://github.com/rtrimble13/arima-garch/issues/141)
- **Lens:** Enhancement
- **Priority:** P3 · **Impact:** Low · **Effort:** Medium · **Severity:** — · **Confidence:** High
- **Evidence:** `src/io/CsvReader.cpp:25-40`
- **Why it matters:** Spreadsheet-exported CSVs with quoted/embedded-delimiter fields parse incorrectly.
- **Recommendation:** Minimal RFC-4180 quote handling in `split_line`.

### [P3] unconditionalVariance() returns 0 at the stationarity boundary — [#142](https://github.com/rtrimble13/arima-garch/issues/142)
- **Lens:** Robustness
- **Priority:** P3 · **Impact:** Low · **Effort:** Low · **Severity:** Low · **Confidence:** Medium
- **Evidence:** `src/models/garch/GarchModel.cpp:55-70`, `src/models/garch/GarchState.cpp:42-48`
- **Why it matters:** Discontinuous `h_0` near the common IGARCH region (α+β≈1) can stall the optimizer.
- **Recommendation:** Always return a finite positive `h_0`; drop the `0.0` sentinel; epsilon margin on the test.

### Additional lower-priority findings (noted, not filed)
These are real but minor; recorded here rather than as separate issues to keep the backlog tight:
- **Student-t distribution object rebuilt on every draw** — `src/simulation/Innovations.cpp:23` reassigns the distribution each call; the constructor init at `:9` and `reset()` at `:38` are effectively dead. Store df, rebuild only on change.
- **Bootstrap sub-seeds collide via fixed offsets** — `src/diagnostics/DiagnosticReport.cpp:94-119` uses `seed`, `seed+1`, `seed+2`, so runs at `s` and `s+1` share streams. Derive sub-seeds via `seed_seq`/hash.
- **`--t-dist` help says "default: 2.0" but there is no default** — `src/cli/main.cpp:38-43`; the value is mandatory when the flag is given, and `CLI::PositiveNumber` admits df≤2 which the engine then rejects. Tighten the help text and the check.

## New feature ideas

The codebase offered little concrete pull toward net-new capability, and several
"obvious" additions are really the bug fixes above (e.g. true integrated
forecasting). Two evidence-backed ideas:

- **AR-root / MA-root stationarity reporting in the fit summary** (P3) — *Evidence:*
  GARCH already reports stationarity (`GarchModel.cpp:43-53`) but ARIMA has no
  analog; #140 adds the check, and surfacing it is a natural, small follow-on that
  improves the diagnostics report users already consume.
- **FFT-backed spectral diagnostics** (P3) — *Evidence:* once autocovariances move
  to FFT for #138, a periodogram / spectral-density view is nearly free and would
  extend the existing ACF/PACF diagnostics. Only worth it if #138 lands first.

## What's done well

- **Coherent error-handling discipline at the boundaries.** The `expected<T,E>`
  pattern is used consistently in `CsvReader`, `Json`, and `api/Engine`, with
  actionable messages (`src/api/Engine.cpp:16-38`, `src/io/Json.cpp:99-129`).
- **The fit objective is genuinely robust.** `FitDriver` wraps unpack, constraint,
  and likelihood evaluation so infeasible/throwing parameter vectors become a
  finite `CONSTRAINT_PENALTY` rather than aborting the optimization
  (`src/estimation/FitDriver.cpp:70-90`) — a thoughtful, correct design. (The
  remaining gap is non-finite NLL, #131.)
- **Thread-safe, portable details.** `getCurrentTimestamp` uses `gmtime_r`/`gmtime_s`
  behind a platform guard (`src/io/Json.cpp:13-29`); the Student-t constant terms
  are correctly hoisted out of the likelihood inner loop (`Likelihood.cpp:61-64`).
- **The Python wrapper is solid.** `run_ag_command` uses `check=True` and wraps
  failures with command + stderr context (`python/ag_viz/utils.py:92-142`), and
  `find_ag_executable` validates the binary so it doesn't silently invoke The
  Silver Searcher (`utils.py:18-37`) — a nice, real-world-aware touch.
- **Broad, well-organized test coverage.** 29 unit suites mirror the module
  structure; the gap is specifically the untested `d > 0` path (#128), not testing
  in general.
