# File-structure refactor plan — preparing TCVAR for TCVAR-SV and steady-state BVAR

## Goal

Today `src/TCVAR/` is a flat folder of 11 files where the layers are mixed: the generic
state-space machinery, the generic BVAR steps, the TCVAR-specific model, and unrelated
reporting utilities all sit side by side, and some files carry two layers at once
(`carter_kohn_algorythm.jl` holds the generic Kalman/Carter-Kohn code *and* the
TCVAR-specific `sample_states`; `gibbs_sampler.jl` holds a generic covariance posterior,
the TCVAR sweep, and result post-processing).

Adding a second and third model (TCVAR with stochastic volatility, steady-state BVAR)
against that layout means copy-pasting the shared 80%. The refactor splits the tree into
four layers so a new model is a new folder that only writes its own model definition,
priors and Gibbs sweep:

```
common/   →  state space, Kalman/Carter-Kohn, generic posterior distributions   (model-agnostic)
var/      →  VAR algebra: companion form, lag ordering, Minnesota prior, NIW posterior
models/   →  one folder per model: TCVAR, TCVAR-SV, steady-state BVAR
reporting/→  plots, summaries, scenario/finance utilities
```

**This is a move-and-split refactor. No numerical behaviour changes** except where
explicitly marked `[BEHAVIOUR]` (there are two such items, both dead-code removals).

## Hard constraints

1. **`src/TCVAR/TCVAR.jl` must stay where it is.** Six notebooks and all three test files
   do `include(".../src/TCVAR/TCVAR.jl"); using .TCVAR`. The entry file keeps its path,
   its module name, and its export list — only its `include` lines change.
2. **The export list must not shrink at any stage.** Notebooks call both exported names
   and internal ones via `TCVAR.compute_posterior_statistics`, `TCVAR.is_stationary`,
   `TCVAR.has_intercept`. Since everything stays inside the one `TCVAR` module, qualified
   access keeps working regardless of which file a function lives in.
3. **`module TCVAR` also contains `struct TCVAR`** (hence `TCVAR.TCVAR(...)` in tests).
   Renaming the module to something model-neutral (`MacroVAR`) is the right end state once
   three models live here, but it breaks every notebook's `using .TCVAR`. Out of scope —
   listed under *Deferred*.

## Target tree

```
src/TCVAR/
├── TCVAR.jl                        # module: usings, includes, exports  (path unchanged)
├── common/
│   ├── linalg.jl                   # chol_psd, sample_mvn
│   ├── state_space.jl              # StateSpaceModel (+ abstract type), simulate/sample
│   ├── kalman_filter.jl            # kalman_filter
│   ├── carter_kohn.jl              # carter_kohn_sampler
│   ├── posteriors.jl               # ← NEW: general posterior-distribution constructors
│   ├── metropolis_hastings.jl      # ← NEW (empty scaffold, filled by the SV work)
│   ├── sv_priors.jl                # ← NEW: stochastic-volatility priors (h̄, ρ, Ω)
│   └── sv_block.jl                 # ← NEW: multivariate SV draw — mixture indicators s,
│                                   #   volatilities h via Carter-Kohn
├── var/
│   ├── var_data.jl                 # prepare_var_data
│   ├── companion.jl                # companion form, lag ordering, stationarity, Lyapunov
│   ├── minnesota_prior.jl          # MinnesotaPrior + accessors
│   └── var_sampling.jl             # NIW conditional posterior + sample_var_params
├── models/
│   ├── tcvar/
│   │   ├── tcvar_model.jl          # tc_var skeleton, TCVAR struct/ctor, update_tc_var!
│   │   ├── tcvar_priors.jl         # var_priors, initial_cycle_prior
│   │   ├── tcvar_states.jl         # sample_states (trend/cycle split of a CK draw)
│   │   ├── tcvar_gibbs.jl          # gibbs_sampler(::TCVAR, data)
│   │   └── tcvar_result.jl         # TCVarResult, build_result, posterior_mean,
│   │                               #   simulate_scenarios
│   ├── tcvar_sv/                   # ← NEW, scaffolding only in this refactor
│   │   ├── tcvar_sv_model.jl       # SV-specific state space + parameter bundle
│   │   ├── tcvar_sv_gibbs.jl       # the sweep: reuses common/sv_block.jl for the SV step
│   │   └── tcvar_sv_result.jl
│   └── ss_bvar/                    # ← NEW, empty placeholder (steady-state BVAR)
├── reporting/
│   ├── posterior_summaries.jl      # compute_posterior_statistics
│   ├── plots.jl                    # plot_states, plot_variable_states
│   └── scenario_stats.jl           # today's utils.jl (finance/reporting helpers)
└── docs/                           # unchanged (+ priors-refactor-plan.md moved in)
```

Alternative considered: flat `tcvar/`, `tcvar_sv/`, `ss_bvar/` without the `models/`
parent. Rejected — with `common/`, `var/` and `reporting/` already present, the extra
level is what makes "one folder per model" legible at a glance.

## Move map

Every function in the current tree, and where it lands. Nothing is dropped except the two
`[BEHAVIOUR]` items.

| From | Function | To |
|---|---|---|
| `state_space.jl` | `StateSpaceModel` | `common/state_space.jl` |
| `state_space.jl` | `sample(model, μ₀, Σ₀, n)` / `sample(model, x₀, n)` | `common/state_space.jl` |
| `carter_kohn_algorythm.jl:8` | `eigen_sqrt` | **delete** `[BEHAVIOUR]` — unused since `chol_psd` landed; the only references are its own docstring and a mention in `sample_mvn`'s |
| `carter_kohn_algorythm.jl:26` | `chol_psd` | `common/linalg.jl` |
| `carter_kohn_algorythm.jl:36` | `sample_mvn` | `common/linalg.jl` |
| `carter_kohn_algorythm.jl:43` | `kalman_filter` | `common/kalman_filter.jl` |
| `carter_kohn_algorythm.jl:114` | `carter_kohn_sampler` | `common/carter_kohn.jl` |
| `carter_kohn_algorythm.jl:173` | `sample_states` | `models/tcvar/tcvar_states.jl` — it is not generic: it knows about the trend block, the cycle companion offset and the pre-sample prepend |
| `gibbs_sampler.jl:1` | `covariance_posterior` | `common/posteriors.jl`, rewritten (see below) |
| `gibbs_sampler.jl:23` | `posterior_mean` | `models/tcvar/tcvar_result.jl` |
| `gibbs_sampler.jl:47,75` | `simulate_scenarios` ×2 | `models/tcvar/tcvar_result.jl` |
| `gibbs_sampler.jl:124` | `gibbs_sampler` | `models/tcvar/tcvar_gibbs.jl` |
| `gibbs_var_steps.jl:11` | `prepare_var_data` | `var/var_data.jl` |
| `gibbs_var_steps.jl:37` | `posterior_beta_coefficient_mean` | `var/var_sampling.jl` (renamed, see naming) |
| `gibbs_var_steps.jl:60` | `beta_cholesky_factor` | `common/posteriors.jl` as generic `kron_cholesky_factor` |
| `gibbs_var_steps.jl:73` | `draw_beta` | `common/posteriors.jl` as generic `draw_from_factor` |
| `gibbs_var_steps.jl:76` | `covariance_posterior_dist` | `var/var_sampling.jl` (the NIW-specific scale), delegating the IW construction to `common/posteriors.jl` |
| `gibbs_var_steps.jl:98` | `is_stationary` | `var/companion.jl` |
| `gibbs_var_steps.jl:119` | `sample_var_params` | `var/var_sampling.jl` |
| `cycle_prior.jl:46` | `MinnesotaPrior` + ctor | `var/minnesota_prior.jl` |
| `cycle_prior.jl:123` | `has_intercept` | `var/minnesota_prior.jl` |
| `cycle_prior.jl:134` | `prior_var_coeff` | `var/minnesota_prior.jl` (uses the shared lag-reversal helper) |
| `cycle_prior.jl:148` | `prior_row_covariance` | `var/minnesota_prior.jl` (ditto) |
| `cycle_prior.jl:173` | `initial_cycle_prior` | `models/tcvar/tcvar_priors.jl`, body reduced to a call to `var_stationary_covariance` |
| `cycle_prior.jl:231` | `var_priors` | `models/tcvar/tcvar_priors.jl` — it bundles the *cycle* priors of a TCVAR, so it is model-level, not generic VAR |
| `TCVAR_model.jl:24` | `tc_var` | `models/tcvar/tcvar_model.jl` (companion block from `var/companion.jl`) |
| `TCVAR_model.jl:60,61` | `default_variable_names`, `default_trend_names` | `models/tcvar/tcvar_model.jl` |
| `TCVAR_model.jl:84,113` | `TCVAR` struct + ctor | `models/tcvar/tcvar_model.jl` |
| `TCVAR_model.jl:165` | `stationary_cycle_covariance` | `models/tcvar/tcvar_model.jl`, body reduced to a slice + `lyapunov_covariance` call |
| `TCVAR_model.jl:195` | `update_tc_var!` | `models/tcvar/tcvar_model.jl` |
| `tcvar_result.jl` | `TCVarResult`, `build_result` | `models/tcvar/tcvar_result.jl` |
| `model_visualisation.jl:5` | `compute_posterior_statistics` | `reporting/posterior_summaries.jl` |
| `model_visualisation.jl:31,48` | `plot_variable_states`, `plot_states` | `reporting/plots.jl` |
| `utils.jl` (12 fns) | drawdown / returns / girf / printing | `reporting/scenario_stats.jl` (verbatim) |
| `SV_priors.jl` | `sv_priors` NamedTuple, `const N = 6` | `common/sv_priors.jl` `[BEHAVIOUR]` — the file is currently **not included by the module** at all; on the way in, `const N = 6` becomes a function argument: `sv_priors(n; ...)` |

## Duplication this refactor removes

These are the concrete reasons the split pays for itself — each is one piece of logic
currently written two or three times, in orderings that must agree or the sampler is
silently wrong.

### 1. Companion-matrix construction — 3 copies

`TCVAR_model.jl:33-39` (skeleton), `gibbs_var_steps.jl:102-104` (stationarity check),
`cycle_prior.jl:183` (initial-cycle prior). All three build
`[0 I; A]` with the same oldest-lag-first convention.

→ `var/companion.jl`:
```julia
companion_matrix(A, n, p)          # n×np bottom block  →  np×np companion F
companion_noise(Σ, n, p)           # np×np Q with Σ in the contemporaneous block
is_stationary(A, n, p)             # unchanged semantics, built on companion_matrix
```

### 2. Lyapunov / stationary covariance — 2 copies

`TCVAR_model.jl:169` (`vecP = (I - kron(Tc,Tc)) \ vec(Qc)`) and `cycle_prior.jl:189`
(identical line on the prior-implied companion).

→ `var/companion.jl`:
```julia
lyapunov_covariance(F, Q)                  # solves P = F P F' + Q
var_stationary_covariance(A, Σ, n, p)      # = lyapunov_covariance(companion_matrix(...), companion_noise(...))
```
`stationary_cycle_covariance(ssm, n_trends)` then becomes: slice the cycle block out of
`T`/`Q`, call `lyapunov_covariance`. `initial_cycle_prior` becomes: check stationarity,
call `var_stationary_covariance`, wrap in `MvNormal`. This also settles the open TODO at
the bottom of `temp_spec.md` ("initial_cycle - separate function unconditional variance in
var").

### 3. Lag-ordering conversion — 5 sites

The `MinnesotaPrior` stores regressors **newest-lag-first** (`[lag1 … lagp, const]`), the
state-space companion and `prepare_var_data` use **oldest-lag-first**. The conversion is
currently open-coded in `prior_var_coeff` (`reverse(reshape(...), dims=2)`),
`prior_row_covariance` (the same reversal on the diagonal), the `var_coeff(β)` closure in
`sample_var_params:134`, and the two `collect(reshape(betas[s,:], k, n_obs)')` calls in
`gibbs_sampler.jl:200,222`.

→ `var/companion.jl` gets the two named primitives, and every site above calls them:
```julia
reverse_lag_blocks(M, n, p)        # newest-lag-first  ↔  oldest-lag-first (involutive)
coefficients_to_companion_block(β, n, p)   # vec(β) or k×n β  →  n×np block A = β'
```
This is the highest-value item in the list: it is the one convention TCVAR-SV and
steady-state BVAR must both get right, and right now it is nowhere written down as a
function.

### 4. Inverse-Wishart posteriors — 2 near-copies

`gibbs_sampler.jl:1` (`covariance_posterior`: differences the trend states, adds the prior
scale) and `gibbs_var_steps.jl:76` (`covariance_posterior_dist`: regression residuals plus
the coefficient-shrinkage term). Both end in `InverseWishart(df, S + prior_scale)`.

→ see the next section.

## General posterior-distribution layer (`common/posteriors.jl`)

`temp_spec.md:88` leaves the naming convention open: *"gibbs sampler step — draw
covariance / draw beta, or posterior distributions — inverse_wishart_posterior,
minnesota_inverse_wishart"*. **Recommendation: both, at two levels**, because they are two
different kinds of function and the split is what makes the layer testable.

- **`*_posterior(...)` — pure, no RNG, returns a `Distributions.jl` object or posterior
  moments.** Testable against closed-form conjugate results without seeding anything.
- **`draw_*(...)` / `sample_*(...)` — consumes RNG, returns a draw.** Thin: build the
  posterior, `rand` it.

Proposed contents:

```julia
"""Σ | ε  ~  IW(ν, ε'ε + Ψ).  `residuals` is T×n, already differenced/de-meaned."""
inverse_wishart_posterior(residuals, scale_prior, df_posterior) -> InverseWishart

"""Random-walk state covariance: differences the state path, then the above."""
random_walk_covariance_posterior(states, scale_prior, df_posterior) -> InverseWishart

"""Conjugate normal posterior mean of vec(β):  (X'X + Ω⁻¹)⁻¹ (X'Y + Ω⁻¹β₀)."""
normal_coefficient_posterior_mean(Y, X, β_prior_mean, Ω_inv) -> Matrix

"""chol(A ⊗ B) = chol(A) ⊗ chol(B) — factor of Σ ⊗ (X'X + Ω⁻¹)⁻¹, computed once."""
kron_cholesky_factor(Σ, V) -> Matrix

"""Draw from N(mean, LL') given a precomputed factor."""
draw_from_factor(mean, L) -> Vector
```

Both current IW callers collapse onto `inverse_wishart_posterior`:

- TCVAR trend step (`gibbs_sampler.jl:212`) → `random_walk_covariance_posterior(trend_states, Ψτ, dτ_post)`
  (identical arithmetic — the `diff` currently hidden inside `covariance_posterior` moves
  into the named random-walk wrapper, where it belongs).
- VAR step (`gibbs_var_steps.jl:126`) → `var/var_sampling.jl` computes the NIW scale
  `ε'ε + (β̂−β₀)' Ω⁻¹ (β̂−β₀)` and hands it to `inverse_wishart_posterior`.

TCVAR-SV then reuses all five for its trend covariance, its volatility-covariance block
(`Ω_gap`) and its volatility-AR coefficients, writing only its own SV-specific step.

**Deferred, not part of this refactor:** threading an explicit `rng::AbstractRNG` through
`draw_*`/`sample_*`. It is the right call for reproducible tests, but it touches every
signature — do it as its own commit once the files have settled.

## What TCVAR-SV needs from `common/` — check before, not after

One structural finding worth acting on during the refactor, because it decides whether the
common layer is actually shareable:

**`StateSpaceModel` has a fixed `Q`, and `kalman_filter` hoists `RQR = R*Q*R'` out of the
loop (`carter_kohn_algorythm.jl:59`).** With stochastic volatility the cycle-innovation
covariance is time-varying, so SV cannot reuse the filter as written.

Plan for it in Stage 2 without implementing SV:

```julia
abstract type AbstractStateSpaceModel end
struct StateSpaceModel        <: AbstractStateSpaceModel   # unchanged fields
struct TimeVaryingStateSpaceModel <: AbstractStateSpaceModel  # Q::Array{Float64,3}  (added by the SV work)

process_noise(m::StateSpaceModel, ::Int)        = m.Q      # hoisting still valid
process_noise(m::TimeVaryingStateSpaceModel, t) = @view m.Q[t, :, :]
```
`kalman_filter` and `carter_kohn_sampler` take `AbstractStateSpaceModel` and go through
`process_noise(model, t)`; keep the `RQR` hoist on the constant-`Q` path so the TCVAR
numbers stay bit-identical. Steady-state BVAR needs nothing new here — it needs the
intercept path of `MinnesotaPrior`/`prepare_var_data`, which already exists.

### The SV block and SV priors are `common/`, not model code

The multivariate stochastic-volatility draw is a *sampling block*, not a property of the
trend-cycle model: it takes residuals and volatility parameters (mean, AR coefficient
matrix, volatility covariance) and returns drawn volatilities. Any model with SV
innovations — TCVAR-SV first, a steady-state BVAR-SV later — calls the same block. So it
lives beside the other shared machinery:

```
common/sv_priors.jl   sv_priors(n; ...)  →  (h̄, ρ, Ω)   # n was the hard-coded `const N = 6`
common/sv_block.jl    draw_sv_states(residuals, params, priors)
                        ├─ draw_mixture_indicators(...)     # Kim–Shephard/Omori mixture, s
                        └─ draw_log_volatilities(...)       # h, via carter_kohn_sampler
```

`draw_log_volatilities` builds a linear-Gaussian state space for `h` and hands it to the
existing `common/carter_kohn.jl` — the second consumer of that code, which is the concrete
test of whether the common layer came out right.

`models/tcvar_sv/` is then only: the SV-augmented state space, the parameter bundle, the
Gibbs sweep wiring the steps together, and the result type. Everything reusable is one
level up.

`common/metropolis_hastings.jl` is created empty (docstring only) as the agreed home for
the MH step named in `temp_spec.md:19-21` — it is generic by the same argument (the SV
persistence/AR draws use it, and so would any non-conjugate block in another model).

## Execution stages

Each stage is one commit and ends green on `julia --project test/runtests.jl`
(`tcvar_priors_test.jl` covers the prior/constructor surface, `tcvar_recovery_test.jl`
actually runs the sampler — that is the real safety net for the moves). Use `git mv` so
file history survives.

**Stage 0 — housekeeping.** Move `src/TCVAR/priors-refactor-plan.md` into `docs/`. Delete
`eigen_sqrt`. No code moves.

**Stage 1 — create folders, move files verbatim.** Create `common/`, `var/`, `models/tcvar/`,
`reporting/`. Move whole files with `git mv` and split the two mixed files
(`carter_kohn_algorythm.jl` → linalg/kalman/carter_kohn + `tcvar_states.jl`;
`gibbs_sampler.jl` → posteriors/gibbs/result). Update the `include` list in `TCVAR.jl`,
ordered `common → var → models → reporting` (Julia includes are order-sensitive: structs
must be defined before the files whose signatures mention them). **Function bodies are not
touched in this stage** — pure relocation, so a failure here is a wiring bug, nothing else.

**Stage 2 — extract the shared VAR primitives.** Write `var/companion.jl`
(`companion_matrix`, `companion_noise`, `lyapunov_covariance`, `var_stationary_covariance`,
`reverse_lag_blocks`, `coefficients_to_companion_block`, `is_stationary`) and rewrite the
3+2+5 duplicate sites listed above to call it. Add the `AbstractStateSpaceModel` /
`process_noise` seam. Highest-risk stage — the lag orderings must round-trip. Guard it with
unit tests written *first*: `reverse_lag_blocks` is involutive; `var_stationary_covariance`
on a known AR(1) equals `σ²/(1−ρ²)`; `companion_matrix` of the prior mean matches what
`tc_var` builds today.

**Stage 3 — general posteriors.** Add `common/posteriors.jl` with the five functions,
repoint both IW callers and the beta draw at it. Verify by fixing a seed and confirming the
sampler output is unchanged against a pre-refactor run (the arithmetic is identical, so
this should be bit-for-bit).

**Stage 4 — scaffold the shared SV layer and the new model folders.** In `common/`:
`sv_priors.jl` (from `SV_priors.jl`, `N` parameterised) and docstring-only scaffolds
`sv_block.jl` and `metropolis_hastings.jl`. Create empty `models/tcvar_sv/`
(`tcvar_sv_model.jl` / `tcvar_sv_gibbs.jl` / `tcvar_sv_result.jl`) and `models/ss_bvar/`.
Include `common/sv_priors.jl` from `TCVAR.jl`; leave the empty scaffolds out of the include
list until they have content.

**Stage 5 — tests follow the structure.** Split `test/` to mirror `src/`:
`test/common/`, `test/var/`, `test/models/tcvar/`. `runtests.jl` keeps its single
`include(".../src/TCVAR/TCVAR.jl")` and includes the sub-testsets; the per-file
`isdefined(Main, :TCVAR) || include(...)` guards keep standalone runs working.

## Verification

- `julia --project test/runtests.jl` after every stage.
- Seeded before/after comparison across the whole refactor: run `gibbs_sampler` on the
  synthetic data from `tcvar_test_utils.jl` with a fixed seed on `main`, save the draws,
  and diff against each stage. Stages 1–5 are all behaviour-preserving, so the draws must
  match exactly; a mismatch means a lag ordering flipped.
- Smoke-run one notebook path: `include("../src/TCVAR/TCVAR.jl")` then the
  `MarcoFinanceTCVAR1.ipynb` priors + `gibbs_sampler` cells, to confirm the export surface
  is intact.

## Deferred (explicitly out of scope, listed so they are not lost)

- **Rename `module TCVAR`** (→ `MacroVAR` or similar) to stop it colliding with
  `struct TCVAR` and to stop reading as "the TCVAR model" once three models live in it.
  Breaks `using .TCVAR` in six notebooks; do it as its own commit with a search-replace
  pass over `analisys/`.
- **Rename `sample(::StateSpaceModel, …)` → `simulate`.** The module does `using StatsBase`
  (which exports `sample`) and then defines its own `sample` — that shadows
  `StatsBase.sample` inside the module and collides for any user who does
  `using StatsBase, TCVAR`. `simulate` is the clearer name for a forward state-space
  simulator anyway. Keep `sample` as a one-line alias for a release.
- **Explicit `rng` arguments** on every `draw_*`/`sample_*` (see the posteriors section).
- **`Plots` as a hard dependency of the estimation module.** `reporting/plots.jl` pulls
  `Plots` into every `using .TCVAR`, including headless test runs. Once the folders exist,
  making `reporting/` a package extension (or a separate sub-module) is a contained change.
- **`reporting/scenario_stats.jl` (today's `utils.jl`) does not belong to a VAR module** at
  all — it is drawdowns, return aggregation and pretty-printing. Natural home is
  `src/reporting/` one level up, but that changes the export surface the notebooks use, so
  it stays inside the module for now.
