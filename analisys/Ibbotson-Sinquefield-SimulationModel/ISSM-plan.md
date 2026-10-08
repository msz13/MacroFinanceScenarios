### Ibbotson–Sinquefield Simulation Model

Simulate annual asset returns with the Ibbotson–Sinquefield building-block model, using Shiller data (US, 1934–2025).

#### Model description

Returns are built from four components (annual, log units):

| Component | Symbol | Dynamics |
|---|---|---|
| Inflation | π_t | AR(1): π_t = c_π + φ_π π_{t-1} + e^π_t |
| Real short rate | r_t | AR(1): r_t = c_r + φ_r r_{t-1} + e^r_t |
| Term spread (10y yield − T-bill yield) | tp_t | AR(1): tp_t = c_tp + φ_tp tp_{t-1} + e^tp_t |
| Equity risk premium (valuation-corrected) | erp_t | bootstrapped, mean-adjusted |

- **As a VAR(1):** stacking `y_t = [π, r, tp, erp]` gives `y_t = c + A y_{t-1} + u_t` with a diagonal `A` (φ_π, φ_r, φ_tp, 0). The simulation takes the intercepts `c` and the lag matrix `A` as its inputs (task 4).
- **Joint bootstrap:** each simulated year draws one historical year *s* and uses the whole vector (e^π_s, e^r_s, e^tp_s, erp_s). This keeps the cross-sectional correlation between components; serial dependence comes only from the three AR(1)s. With a general `A`, serial dependence comes from the VAR.
- **Bootstrap variant:** i.i.d. draws of single years, with replacement. There is no block bootstrap.
- **ERP correction:** historical excess return includes valuation re-rating (CAPE rose over the sample), which should not be extrapolated. Corrected ERP: `erp_t = xr_t − Δlog CAPE_t`. The bootstrapped erp series is then shifted so its mean equals an **external target** `erp_target` (in the VAR form: `c_erp = erp_target` with demeaned residuals). This is a required input with no default. A typical choice is the mean of historical returns stripped of valuation changes, i.e. the mean of `erp_t`, possibly from a different period or market than the bootstrap sample. Shifting changes the mean only, not the volatility.
- **Sample periods:** the AR(1) estimation window and the bootstrap window are chosen independently, e.g. full sample (1934–2025), 1975–2025, or any other `(start_year, end_year)`.
- **Returns from components:**
  - T-bill: `rf_t = π_t + r_t`
  - 10y bond: `rf_t + tp_t` is the log 10y yield at the start of year t; `rb_t` is the par-bond return (`calculate_bond_returns`) from the yield at the start of year t to the start of year t+1, so the last simulated year has no bond return
  - Equity: `re_t = rf_t + erp_t`
  - Real returns: subtract π_t.

#### Data (decisions made)

- Source: `data/ie_data.xlsx`, sheet **`Data2`** (monthly, 1934-01 to 2026-03; columns `Date, P, D, E, CPI, Rate GS10, CAPE, TBILL`, …). Do **not** use `ie_data.xls`: XLSX.jl cannot read `.xls`, and the original Shiller sheet has no T-bill column. 1934 is the first year with TBILL data.
- Frequency: annual, calendar year (Dec→Dec). Use full years only, i.e. 1935–2025: Dec→Dec needs December 1933, which Data2 lacks, so 1934 is lost (2026 is partial).
- Annual series (all log, decimals; the EDA notebook shows them in percent, see task 6):
  - `π_t = log(CPI_Dec,t / CPI_Dec,t-1)`
  - `rf_t = log(1 + TBILL_Dec,t-1/100)`: the 3m yield at the start of the year, rolled forward (approximation; note it in the script).
  - `r_t = rf_t − π_t` (ex-post real rate)
  - Bond return from GS10 with the par-bond formula; reuse/move `calculate_bond_returns` from `src/TCVAR/reporting/scenario_stats.jl` (T = 10, annual step).
  - `tp_t = log(1 + GS10_Dec,t-1/100) − rf_t` (term spread: 10y yield over the T-bill yield, both at the start of the year)
  - Equity total return: `re_t = log((P_Dec,t + D_t) / P_Dec,t-1)`, where D_t is the dividend paid over the year (sum of monthly D/12).
  - `xr_t = re_t − rf_t`, `Δlog CAPE_t`, `erp_t = xr_t − Δlog CAPE_t`
  - Fundamental equity return `fr_t = re_t − Δlog CAPE_t` (total return stripped of the valuation change; EDA only, not bootstrapped)

#### Code layout

Wrap the new code in a module `IbbotsonSinquefield` (include it from `MacroFinanceScenarios.jl` the same way as `TCFSimulation`).

```
src/IbbotsonSinquefield/
  IbbotsonSinquefield.jl   # module, includes below
  data.jl                  # load_shiller_annual, subperiod
  ar1.jl                   # fit_ar1, AR1 struct (c, φ, σ, se, R², residuals), unconditional_mean/std
  simulate.jl              # is_coefficients, issm_bootstrap_matrix, simulate_var_bootstrapped_res, returns_from_components
src/EDA/
  eda.jl                   # describe_series, print_correlations
src/ScenariosEvaluation/
  evaluation.jl            # scenario moments, correlations, percentiles, drawdowns
analisys/Ibbotson-Sinquefield-SimulationModel/
  ISSM-plan.md
  issm_eda.ipynb           # exploration: data, EDA, AR fits per period, diagnostics
  issm_report.qmd          # Quarto report: simulation + scenario evaluation (parametrised)
  reports/                 # rendered HTML, one per specification
test/ibbotson_sinquefield_test.jl
```

**Scenario format:** `Array{Float64,3}` of shape `n_vars × T × n_scen`, same as `print_scenarios_summary` in TCVAR, plus a `var_names::Vector{Symbol}`. Wrap it in a small struct `Scenarios(data, names, start_year)` so the evaluation functions don't need extra arguments.

**Dependencies:** no GLM. AR(1) is plain OLS (`[1 x_{t-1}] \ x_t`) in `fit_ar1`, which also returns the OLS standard errors and R², so the EDA notebook's estimation tables come from `fit_ar1` too.

**Reporting:** all EDA and evaluation functions **return** labelled tables (a `DataFrame`, or a matrix plus names) and do not print. A single `print_table(t; backend=:text, digits=4)` wrapper does the printing (PrettyTables 3 syntax `backend = :html`, not v2's `Val(:html)`). The same functions then work in the REPL (`:text`), the notebook and Quarto (`:html`), and in tests.

**Tooling:** Quarto CLI (install the `.deb` from quarto.org in WSL), `engine: julia`; QuartoNotebookRunner.jl installs on the first render, so no Python or Jupyter is needed for the report. The EDA notebook needs IJulia (or the VS Code Julia notebook kernel).

#### Project tasks

1. **EDA helpers** (`src/EDA/eda.jl`, StatsBase + PrettyTables)
   - `describe_series(ta::TimeArray)`: one row per variable with columns mean, std, skewness, excess kurtosis, AR(1) autocorrelation, min, p25, p50, p75, max.
   - `correlation_table(ta::TimeArray)`: correlation matrix table.
   - Return the table (see **Reporting**). Print it with `print_table`, so tests can check the numbers.

2. **Data** (`data.jl`)
   - `load_shiller_annual(path; sheet="Data2") -> TimeArray` with π, rf, r, rb, tp, re, xr, cape, Δlog CAPE, fr, erp.
   - `subperiod(ta, (y0, y1))`: annual slice; reused for EDA, AR fit and the bootstrap window.
   - Test: identities (`re − rf == xr`, `xr − Δlog CAPE == erp`) and no missing values for 1935–2025.

3. **AR(1)** (`ar1.jl`)
   - `fit_ar1(x; years) / fit_ar1(ta[, var]) -> AR1(c, φ, σ, se, r2, resid, fitted, years, period)`: call it on a `subperiod` slice and store the period for reporting. `σ` uses n − 2 degrees of freedom; `years` labels the residuals (the first year is lost to the lag); `period` is the input sample including the lag year.
   - `unconditional_mean(ar) = c/(1−φ)`, `unconditional_std(ar) = σ/√(1−φ²)`.
   - Test: recovers c and φ on simulated data.

4. **Simulation** (`simulate.jl`): a VAR(1) with bootstrapped residuals.
   - **Model:** state `y_t = [π, r, tp, erp]` (n = 4), simulated as `y_t = c + A y_{t-1} + u_t`. The parameters are the intercepts `c` (length n) and the lag matrix `A` (n × n), plus `names::Vector{Symbol}` for the rows.
     - The Ibbotson–Sinquefield model is the special case with a diagonal `A`: π, r and tp rows hold their AR(1) `c` and `φ`; the erp row has all lag coefficients equal to zero, so erp is an i.i.d. draw around its intercept. tp is an AR(1) because it is a yield spread: it is persistent (φ ≈ 0.5–0.56), and i.i.d. draws would make the 10y yield jump every year and double the bond volatility.
     - Intercepts set the means: for tp, `c_tp/(1−φ_tp)` = mean of tp (or a target via `tp_mean`); for erp, `c_erp = erp_target`. So the ERP mean-shift is just the erp intercept, not a separate step.
     - The same code then runs any VAR(1) (off-diagonal terms, e.g. an r → erp effect) without changes.
   - `is_coefficients(ar_π::AR1, ar_r::AR1, ar_tp::AR1; erp_target, tp_mean=nothing) -> (c, A, names)`: builds `c` and `A` for the Ibbotson–Sinquefield case from the three AR(1) fits. `erp_target` is required (no default); `tp_mean` optionally overrides the unconditional mean of tp (keeping φ_tp).
   - `issm_bootstrap_matrix(data::TimeArray, c, A, names, period) -> (years, U)`: residuals `u_t = y_t − c − A y_{t-1}` for the years in `period` (n_years × n). This covers e^π, e^r and e^tp as AR residuals, and erp − c_erp for the i.i.d. row. The AR window (behind `c`, `A`) and the bootstrap window (`period`) are independent. The first year of `period` is lost to the lag in every column, so all columns come from the **same years**.
     - The residuals are always **demeaned** column by column (no option), so the simulated means come from `c` and `A` alone: E[y] = (I − A)⁻¹ c, i.e. c/(1−φ) for π, r and tp, and `erp_target` for erp. A bootstrap window whose residual means are non-zero (for example, because the AR was fitted on a different window) therefore does not shift the simulated means.
   - `simulate_var_bootstrapped_res(c, A, names, U, T, n_scen; y0=:last, data, start_year, rng) -> Scenarios`:
     - Each simulated year draws one historical row of `U` i.i.d. with replacement and uses the whole row, which keeps the cross-sectional correlation between the shocks. Serial dependence comes only from `A`.
     - `y0 = :last`: start from the last observed `y` (default). `y0 = :mean`: start from the unconditional mean (I − A)⁻¹ c, for a steady-state view. A vector sets `y0` explicitly.
     - Check that `A` is stable (all eigenvalues inside the unit circle) before simulating.
   - `returns_from_components(sc::Scenarios; real=false) -> Scenarios` with rf, rb, re (nominal or real).
   - Test: with large `n_scen` and `y0 = :mean`, the simulated means match (I − A)⁻¹ c, so erp ≈ `erp_target` and π ≈ c/(1−φ). This should hold both for the diagonal Ibbotson–Sinquefield `A` and for an `A` with an off-diagonal term. With `U = 0` the paths reproduce the deterministic recursion.

5. **Scenario evaluation** (`src/ScenariosEvaluation/evaluation.jl`; move the TCVAR `max_drawdown_and_length`, `annualise`, `print_percentiles`)
   - **Per-path moments:** for each variable, compute mean, std, skewness, kurtosis, AR(1) within each scenario path. Report the mean across scenarios and percentiles (5, 25, 50, 75, 95) of those statistics. One table per variable (rows = statistics, columns = mean, p5…p95).
   - **Correlations:** correlation matrix computed per scenario, then averaged across scenarios.
   - **Horizon percentiles:** for horizons, e.g. 1, 5, 10, 25 years, report percentiles e.g. (5, 25, 50, 75, 95). What to report depends on the variable:

     | Variable | Annualised cumulative `sum(x_1..h)/h` | Level in year h `x_h` |
     |---|---|---|
     | Asset returns rf, rb, re (nominal and real) | yes, main table | no |
     | Inflation π | yes: annualised inflation = price-level path, needed to deflate wealth | yes: fan chart of the yearly rate |
     | Real short rate r, nominal short rate rf | yes, for rf only: cash return over the horizon | yes: fan chart of the yearly rate |
     | Term spread tp | no (a yield spread, not a return) | yes: fan chart of the yearly spread |
     | Premium erp | yes | no |

     - Why both for π and the short rates: they are persistent AR(1) state variables. The **level** at year h shows the dynamics: the starting point, the speed of mean reversion toward c/(1−φ), and the dispersion widening to the unconditional σ/√(1−φ²). These levels are what the AR(1) should be checked against. The **annualised cumulative** value is what matters for an investor: average inflation and cash return over the holding period. It sits in the same table as the asset returns, so real-vs-nominal comparisons are consistent.
     - erp and asset returns are i.i.d. draws (plus persistent rf and yield changes), so their single-year levels carry no extra information beyond the one-year moments. Use the annualised cumulative only.
     - Optional: cumulative wealth `exp(sum)` for asset returns and the cumulative price level `exp(sum π)`.
   - **Drawdowns:** max drawdown and longest drawdown length per path (nominal and real equity, bonds); report the mean and percentiles.

6. **EDA notebook** (`issm_eda.ipynb`): interactive exploration; the output is the choice of periods and `erp_target`.
   1. Load data → annual TimeArray, then convert to **percent**: every series × 100 except the `cape` level. The library code (`load_shiller_annual`, AR fit, simulation) stays in decimals; only the notebook rescales, for readability. In the notebook, AR(1) constants, σ, residuals and the ERP means are in percentage points (φ, R² and correlations are unit-free), so an `erp_target` chosen there must be divided by 100 before it goes into the report's `params`.
   2. EDA: moments and correlations of π, r, rf, tp, xr, Δlog CAPE, erp, per period.
   3. Define the periods, e.g. `periods = Dict(:full => (1934, 2025), :post1975 => (1975, 2025), ...)`. Fit AR(1) for π, r and tp on each period and print a coefficient table per period from `fit_ar1` (coefficients, s.e., t-stats, R², σ) plus a side-by-side comparison of c, φ, σ and the unconditional mean.
   4. Plot fitted vs actual for π and r; plot residuals (+ residual ACF and a QQ plot).
   5. Build the bootstrap matrix [e^π e^r e^tp erp] (aligned years) per period and print its moments and correlations. Print the mean of raw xr vs corrected erp per period, as input for choosing `erp_target`.

7. **Quarto report** (`issm_report.qmd`): reproducible, parametrised; it reruns the simulation on every render (cheap; no scenario files saved).
   - Header:
     ```yaml
     title: "Ibbotson–Sinquefield scenarios"
     engine: julia
     format: { html: { embed-resources: true, toc: true, code-fold: true } }
     params:
       ar_start: 1934
       ar_end: 2025
       boot_start: 1934
       boot_end: 2025
       erp_target: 0.04
       T: 30
       n_scen: 10000
       seed: 1234
     ```
     The first code cell is tagged `#| tags: [parameters]` and defines the same defaults as Julia variables; Quarto overrides them with `-P`. Check that the installed QuartoNotebookRunner version supports parameters; if not, read them from environment variables.
   - Render: `quarto render issm_report.qmd -P erp_target:0.04 -P boot_start:1975 --output-dir reports --output issm_boot1975_erp4.0.html`. Use a file name that encodes the specification. A small `render_reports.sh` loops over the specifications to compare (at least full vs post-1975).
   - Sections:
     1. Specification: parameters, sample periods, AR(1) coefficients and the historical moments of the bootstrap sample (compact recap from the EDA functions).
     2. Build `c`, `A` (`is_coefficients`) and `U` (`issm_bootstrap_matrix`), simulate (`simulate_var_bootstrapped_res`) and build nominal and real returns (`returns_from_components`).
     3. Fan charts (5–95 percentiles, median, unconditional mean line) of π, r and rf levels by year.
     4. Evaluation tables: per-path moments, mean correlations, horizon percentiles (annualised cumulative; π and rf also as levels), drawdowns.
     5. Validation: simulated one-year moments and correlations vs the historical ones from the bootstrap period. They should match closely for erp and on average for π/r/tp.
   - Rendered HTML files go to `reports/`. Decide whether to commit them (recommendation: commit the final ones only, and gitignore Quarto's `_freeze`/`*_files` folders).

#### Decisions
- ERP target: an external figure passed in by the user, e.g. the mean of returns stripped of valuation changes. It is not computed implicitly from the bootstrap sample.
- Bootstrap: i.i.d. draws of single years.
- Workflow: Jupyter notebook for exploration (EDA, AR fits, choice of `erp_target`); a parametrised Quarto `.qmd` rendered to self-contained HTML for the simulation and evaluation. Functions return tables; printing is a separate concern.
- Horizon evaluation: report annualised cumulative values for all returns and premia. For π, r and rf, report both annualised cumulative and year-h levels (fan charts).
- Periods: the AR(1) fit window and the bootstrap window are parameters, e.g. full sample, 1975–, or others. The script compares at least full vs post-1975.
