# Refactor `gibbs_sampler` to a distribution-keyed priors NamedTuple

Supersedes `src/TCVAR/priors-refactor-plan.md` (partially applied — it kept the cycle
prior as a separate positional argument under the key `cycle`; this plan folds it into
the tuple and adds `initial_cycle` / `cycle_covariance`).

## Target API

```julia
Σ_prior, β_prior, c0_prior = var_priors(.2, 4, [2., 1., .1, 25., 1]; δ = zeros(5))

priors = (
    initial_trend    = MvNormal([2., .5, 1., 2., 5.], diagm(ones(5))),
    initial_cycle    = c0_prior,                                        # MvNormal, length n*p
    trend_covariance = InverseWishart(100, diagm([2., 1., 1., 2., 1.].^2 ./ 400)),
    cycle_covariance = Σ_prior,                                         # InverseWishart
    cycle_β          = β_prior,                                         # MinnesotaPrior
)

model = TCVAR.TCVAR(observation_trend_mapping, priors;
                    variable_names = ..., trend_names = ...)
```

One key per parameter name, one Distributions.jl object (or `MinnesotaPrior`) per key.
The sampler stops reconstructing distributions from raw arrays and just reads
`mean` / `cov` / `params` off them. `n` and `p` are read from `priors.cycle_β`, so
`cycle_prior` disappears as a separate constructor argument.

### Key ↔ current source of the same information

| new key | replaces (today) |
| --- | --- |
| `initial_trend` | `priors.initial_trend_mean`, `priors.initial_trend_covariance` |
| `initial_cycle` | `priors.initial_cycle_mean` + `kron(I(p), E[Σc])` built inside the sampler |
| `trend_covariance` | `priors.trend_covariance_df`, `priors.trend_covariance_mean` |
| `cycle_covariance` | `priors.cycle.Ψ`, `priors.cycle.d` (fields already removed from `MinnesotaPrior`) |
| `cycle_β` | `priors.cycle` (`Φ₀`, `Ω`, `n`, `p`, `k`, `λ`) |

## Behaviour changes to be aware of

1. **`trend_covariance` is now the IW scale `Ψ`, not the prior mean/mode.** Today
   `gibbs_sampler.jl:142` computes `scale = mean * (df + n + 1)`, i.e. the notebook
   matrix is interpreted as the IW *mode*. With `InverseWishart(100, Ψ)` the scale is
   taken at face value. To reproduce the old prior numerically the notebook must pass
   `diagm([2., 1., 1., 2., 1.].^2 ./ 400) * (100 + nt + 1)`; passing the bare matrix
   (as in the example above) gives a prior ~106× tighter. **Assumption taken here:**
   the distribution is used as written, and the notebook cell gets a comment stating
   the conversion — no hidden rescaling in the sampler.
2. **`initial_cycle` covariance** becomes the stationary covariance implied by
   `Φ₀` and `E[Σc]` (`initial_cycle_prior`) instead of `kron(I(p), E[Σc])`. These
   coincide for `δ = 0` (`Φ₀ = 0`), which is the default and what the notebook uses;
   they differ for a nonzero `δ`. Either way it only affects the draw-2 state sampling,
   since the cycle block of `initial_state_covariance` is overwritten from
   `stationary_cycle_covariance` at the end of every draw.
3. **The `p` used in the notebook example is 2, the current model uses 4.** `p` now
   comes only from `cycle_β`, so `var_priors(.2, p, …)` must be called with the lag
   count actually wanted.

## Changes

### 1. `src/TCVAR/cycle_prior.jl` — one new accessor

`prior_var_coeff(pr)` already returns the lag-reversed, intercept-dropped coefficient
mean. Add the matching accessor for the row covariance so both lag reversals live next
to each other in the prior file instead of being open-coded in the sampler:

```julia
"""
    prior_row_covariance(pr::MinnesotaPrior) -> Diagonal

Prior row covariance `Ω` (`n*p × n*p`) in the oldest-lag-first ordering used by the
state-space cycle companion, with the intercept row dropped.
"""
function prior_row_covariance(pr::MinnesotaPrior)
    k = pr.n * pr.p
    return Diagonal(vec(reverse(reshape(diag(pr.Ω)[1:k], pr.n, pr.p), dims = 2)))
end
```

Export it from `TCVAR.jl` alongside `var_priors, initial_cycle_prior`.

### 2. `src/TCVAR/TCVAR_model.jl` — fix and finish the constructor

`TCVAR(...)` at line 97 is currently **broken**: `cycle_prior` is commented out of the
signature (`#= cycle_prior::MinnesotaPrior; =#`) but still referenced in the body
(lines 103, 110, 111). Replace with:

```julia
function TCVAR(trend_mapping, priors;
               variable_names = default_variable_names(size(trend_mapping, 1)),
               trend_names    = default_trend_names(size(trend_mapping, 2)))

    n_obs, n_trends = size(trend_mapping)

    required = (:initial_trend, :initial_cycle, :trend_covariance, :cycle_covariance, :cycle_β)
    missing_keys = filter(k -> !haskey(priors, k), required)
    isempty(missing_keys) || throw(ArgumentError("priors is missing key(s): $(join(missing_keys, \", \"))"))

    β_prior = priors.cycle_β
    p = β_prior.p

    β_prior.n == n_obs || throw(DimensionMismatch(
        "priors.cycle_β is built for n = $(β_prior.n) variables, trend_mapping has $n_obs"))
    size(priors.cycle_covariance) == (n_obs, n_obs) || throw(DimensionMismatch(
        "priors.cycle_covariance must be $n_obs × $n_obs"))
    length(priors.initial_cycle) == n_obs * p || throw(DimensionMismatch(
        "priors.initial_cycle must have length n_obs*p = $(n_obs * p)"))
    length(priors.initial_trend) == n_trends || throw(DimensionMismatch(
        "priors.initial_trend must have length $n_trends"))
    size(priors.trend_covariance) == (n_trends, n_trends) || throw(DimensionMismatch(
        "priors.trend_covariance must be $n_trends × $n_trends"))
    length(variable_names) == n_obs   || throw(DimensionMismatch("variable_names must have length $n_obs"))
    length(trend_names)    == n_trends || throw(DimensionMismatch("trend_names must have length $n_trends"))

    return TCVAR(tc_var(trend_mapping; p = p), priors,
                 collect(String, variable_names), collect(String, trend_names))
end
```

No `merge` — the tuple is stored as given. Update the struct docstring (lines 70–79) to
list the five keys and their types.

### 3. `src/TCVAR/gibbs_sampler.jl` — read the distributions

Rewrite the prologue (lines 99–167). Line-by-line replacements:

```julia
priors  = model.priors
β_prior = priors.cycle_β
ssm     = model.ssm

n_time_steps, n_obs = size(data)
n_trends = length(model.trend_names)
p = β_prior.p
k = n_obs * p

n_obs == β_prior.n || throw(DimensionMismatch(
    "data has $n_obs variables but the model was built for $(β_prior.n)"))

# --- prior quantities read straight off the distributions ---
ντ, Ψτ = params(priors.trend_covariance)      # (df, PDMat scale)
νc, Ψc = params(priors.cycle_covariance)

trend_covariance_scale = Matrix(Ψτ)           # replaces the mean → scale conversion
cycle_covariance_scale = Matrix(Ψc)           # replaces Matrix(cycle_prior.Ψ)
cycle_covariance_mean  = mean(priors.cycle_covariance)   # Ψc/(νc - n - 1)

Ω_inv            = inv(prior_row_covariance(β_prior))
cycle_coeff_mean = collect(prior_var_coeff(β_prior)')    # k × n, oldest-lag-first
```

- line 125 `dτ_post = n_trend_time_steps - 1 + ντ`
- line 129 `dc_post = n_cycle_time_steps - p + νc`
- lines 133–140: deleted, replaced by the block above (the `reverse`/`reshape` gymnastics
  move into `prior_row_covariance` / the existing `prior_var_coeff`).
- line 142 `trend_covariance_scale = priors.trend_covariance_mean * (...)`: **deleted**.
- lines 145–146:
  ```julia
  initial_cycle_mean       = mean(priors.initial_cycle)        # already length n_obs*p
  initial_cycle_covariance = Matrix(cov(priors.initial_cycle))
  ```
  (no `repeat`, no `kron` — `c0_prior` is already in companion order)
- lines 151–153:
  ```julia
  initial_state_mean = [mean(priors.initial_trend); initial_cycle_mean]
  initial_state_covariance = [Matrix(cov(priors.initial_trend))  zeros(n_trends, k)
                              zeros(k, n_trends)                 initial_cycle_covariance]
  ```
- line 164 seed `trend_covariance[1, :, :] = mean(priors.trend_covariance)`
  (matches how the cycle block is seeded from `cycle_covariance_mean`; draw 1 is
  discarded in burn-in either way).
- line 167 seed `sigmas[1, :, :] = cycle_covariance_mean` — unchanged.

Update the docstring above `gibbs_sampler` (lines 79–97): it currently says the cycle
`MinnesotaPrior` lives under `priors.cycle` and that the lag blocks are reversed in the
sampler; both statements change.

Nothing else in the sampler moves — the draw loop, `update_tc_var!` calls,
`stationary_cycle_covariance` re-initialisation and `build_result` are untouched.

### 4. `src/TCVAR/tcvar_result.jl`, `posterior_mean`

No changes. Parameter keys (`Στ`, `β`, `Σc`) and shapes are unaffected.

### 4b. `src/TCVAR/gibbs_sampler.jl` — a parameter-driven `simulate_scenarios` method

`simulate_scenarios` currently only accepts a `TCVarResult`, so there is no way to
simulate from *known* parameters — which the recovery script (task 7) needs. Split the
existing body: the new method does the work, the old one supplies posterior means.

```julia
"""
    simulate_scenarios(model::TCVAR, params::NamedTuple, initial_state, n_scenarios, n_steps)

Simulate from `model` at explicit parameters `params = (Στ, β, Σc)` starting from
`initial_state = [τ₀; ξ₀]` (trend block, then the cycle companion oldest-lag-first).
`model.ssm` is left untouched — the parameters are written into a private copy.
"""
function simulate_scenarios(model::TCVAR, params::NamedTuple, initial_state::AbstractVector,
                            n_scenarios::Int, n_steps::Int)
    n_trends = size(params.Στ, 1)
    n_obs    = size(params.Σc, 1)
    k        = size(params.β, 1)
    p        = k ÷ n_obs

    length(initial_state) == n_trends + k || throw(DimensionMismatch(
        "initial_state must have length n_trends + n_obs*p = $(n_trends + k)"))

    ssm = deepcopy(model.ssm)
    update_tc_var!(ssm, collect(params.β'), params.Στ, params.Σc, n_trends, n_obs, p)

    states       = zeros(n_scenarios, n_steps, n_trends + k)
    observations = zeros(n_scenarios, n_steps, n_obs)
    for s in 1:n_scenarios
        states[s, :, :], observations[s, :, :] = sample(ssm, initial_state, n_steps)
    end
    return states, observations
end
```

`simulate_scenarios(result::TCVarResult, n_scenarios, n_steps)` keeps its docstring and
its terminal-state construction, then ends with

```julia
return simulate_scenarios(result.model, posterior_mean(result), initial_state,
                          n_scenarios, n_steps)
```

### 5. `analisys/MarcoFinanceTCVAR1.ipynb` (via NotebookEdit)

- Cell 10 (old 5-field tuple + `MinnesotaPrior(.2, n, p, ψ, n+2; …)`): delete — that
  `MinnesotaPrior` positional signature no longer exists anyway.
- Cell 11 (`c0_prior.μ` scratch): delete.
- Cell 12: becomes the single priors cell, with `p = 4` in the `var_priors` call and a
  comment on the `trend_covariance` scale convention (see "Behaviour changes" 1 and 3).
- Cell 15: `TCVAR.TCVAR(observation_trend_mapping, priors; variable_names = …, trend_names = …)`
  — drop the third positional argument.
- Cell 16 (`gibbs_sampler(model, data; …)`) unchanged.
- Other notebooks (`TCVAR_bivariate`, `macro-finance*`, `del negro replication`) use the
  old signatures and are left alone unless asked.

## Verification

No smoke scripts — the refactor is verified by deterministic tests committed under
`test/`, plus one simulation-recovery script (task 7) that is statistical by nature but
still seeded and assertive.

### Harness facts (checked against the current repo)

- `src/MacroFinanceScenarios.jl` does **not** include the TCVAR module, so
  `using MacroFinanceScenarios` does not load it. The tests must
  `include(joinpath(@__DIR__, "..", "src", "TCVAR", "TCVAR.jl"))` and `using .TCVAR`.
- `using` is only legal at top level, so `runtests.jl` includes the test files at top
  level; each file carries its own `@testset` and an `isdefined(Main, :TCVAR) || include(…)`
  guard so it also runs standalone (`julia --project test/tcvar_priors_test.jl`).
- No `Project.toml` change needed: `Distributions` / `FlexiChains` are deps, and the
  stdlibs (`Test`, `LinearAlgebra`, `Random`, `Statistics`) resolve from `@stdlib` under
  `--project=.` whether the suite is started by `Pkg.test()` (where `Test` is also in
  `[extras]`) or by running `test/runtests.jl` directly (verified on julia 1.12.5).
- `carter_kohn_sampler` only has a method for `Matrix{Union{Missing,Float64}}`, so every
  test must pass `convert(Matrix{Union{Missing,Float64}}, data)` — plain `Matrix{Float64}`
  raises a `MethodError`.
- `result.params[@varname(β)]` returns a `DimMatrix` (iter × chain) **of matrices**, not a
  numeric array. `mean` works directly; an element-wise median needs stacking:
  `reduce(hcat, vec.(vec(collect(draws))))` → `n_elements × n_draws`, then
  `median(·, dims = 2)` and `reshape` back. Put both helpers in `test/tcvar_test_utils.jl`.

### 6. `test/runtests.jl`, `test/tcvar_test_utils.jl`, `test/tcvar_priors_test.jl`

`runtests.jl` loads the module once and includes the test files; `tcvar_test_utils.jl`
holds `tcvar_test_priors(; n, nt, p, λ, ψ, trend_variances, trend_df)` (builds the
five-key tuple, writing `trend_covariance = InverseWishart(df, diagm(v) * (df - nt - 1))`
so `mean(...) == diagm(v)` and the scale convention of "Behaviour changes" 1 is explicit)
plus the draw mean/median helpers.

`tcvar_priors_test.jl` — deterministic assertions, all expected values below were
computed from the current `cycle_prior.jl` and confirmed numerically:

1. **`var_priors(0.2, 2, [2.0, 1.0, 0.5]; δ = zeros(3))`** — `Σ_prior isa InverseWishart`,
   `size == (3,3)`, `df == n+2 == 5`, `Matrix(Ψ) ≈ diagm(ψ)`, `mean(Σ_prior) ≈ diagm(ψ)`;
   `β_prior.k == n*p == 6`, `!has_intercept(β_prior)`, `iszero(β_prior.Φ₀)`;
   `length(c0_prior) == 6`, `mean(c0_prior) == zeros(6)` and
   `Matrix(cov(c0_prior)) ≈ kron(I(p), mean(Σ_prior))` (exact, since `Φ₀ = 0`).
2. **Lag-order accessors** with `n = 2, p = 2, λ = 0.5, ψ = [4.0, 1.0], δ = [0.9, 0.8],
   intercept = true` (so `k = 5` and the intercept must be dropped):
   `prior_var_coeff(pr) == [0 0 0.9 0; 0 0 0 0.8]` and
   `diag(prior_row_covariance(pr)) ≈ [0.015625, 0.0625, 0.0625, 0.25]`
   (= `[λ²/(4σ̄₁), λ²/(4σ̄₂), λ²/σ̄₁, λ²/σ̄₂]`, i.e. lag-2 block first). These two pin the
   oldest-lag-first reversal that moves out of the sampler.
3. **Constructor** — `p` and `n` inferred from `cycle_β`: with `p = 3, n = 2, nt = 2`,
   `size(model.ssm.T) == (nt + n*p, nt + n*p)`, `size(model.ssm.Z) == (n, nt + n*p)`;
   `keys(model.priors) == (:initial_trend, :initial_cycle, :trend_covariance,
   :cycle_covariance, :cycle_β)` (stored verbatim — no `cycle` key merged in);
   the skeleton starts zeroed (`iszero(model.ssm.Q)`).
4. **Constructor error paths** — `@test_throws ArgumentError` for a tuple missing
   `cycle_covariance`; `@test_throws DimensionMismatch` for each of: `initial_cycle` of
   length `n*p ± 1`, `trend_covariance` sized for the wrong `n_trends`, `cycle_β.n` ≠
   rows of `trend_mapping`, and `variable_names` of the wrong length.
5. **Sampler shapes and determinism** — `T = 60`, `burnin = 20`, `n_samples = 20`,
   `thin = 2` ⇒ `kept = 21:2:40`, 10 draws:
   `size(result.trend_states) == (10, T+1, nt)`, `size(result.cycle_states) == (10, T+p, n)`;
   `posterior_mean(result)` returns `Στ` `nt×nt`, `β` `n*p×n`, `Σc` `n×n`, with
   `issymmetric`/`isposdef` on the two covariances and `TCVAR.is_stationary(β', n, p)`;
   `result.model.ssm` comes back zeroed (`iszero(Q)` and the companion bottom block).
   Determinism: two runs under the same `Random.seed!` give `==` draws, and runs under
   different seeds differ.
6. **`simulate_scenarios`** — both methods: from a `TCVarResult`,
   `simulate_scenarios(result, 7, 12)` gives `(7, 12, nt + n*p)` / `(7, 12, n)`; from
   explicit parameters, `simulate_scenarios(model, (Στ =…, β =…, Σc =…), initial_state, 5, 10)`
   gives `(5, 10, nt + n*p)` / `(5, 10, n)`, is reproducible under a fixed seed, throws
   `DimensionMismatch` on a wrong-length `initial_state`, and leaves `model.ssm` zeroed.

### 7. `test/tcvar_recovery_test.jl` — simulate → estimate → compare

A script that doubles as a test: simulate one dataset from the TCVAR at **example
parameters** with the new `simulate_scenarios` method, run `gibbs_sampler` on it, and
compare the posterior **mean and median** of every parameter block against the truth,
printing a `true / mean / median` table and asserting tolerances.

DGP (`n = 2`, `nt = 2`, `p = 1`, `trend_mapping = I`, `T = 400`,
`initial_state = [1.0, 2.0, 0.0, 0.0]`):

```julia
Στ_true = diagm([0.01, 0.02])
A_true  = [0.6 0.0; 0.1 0.5]      # companion bottom block; β_true = A_true'
Σc_true = [1.0 0.2; 0.2 0.5]
_, obs  = simulate_scenarios(model, (Στ = Στ_true, β = β_true, Σc = Σc_true),
                             initial_state, 1, T)
data    = convert(Matrix{Union{Missing,Float64}}, obs[1, :, :])
result  = gibbs_sampler(model, data; burnin = 1_000, n_samples = 2000)
```

Estimation priors are deliberately **off** the truth so the test measures learning, not
prior echo: `var_priors(0.5, 1, [2.0, 1.0]; δ = zeros(2))` (cycle variances 2× true) and
`trend_covariance` centred at `diagm([0.02, 0.02])` with `df = 20`.

Tolerances — measured on a scratchpad replica of the post-refactor sampler over seeds
`{42, 7, 123}` at `T = 400`, then padded:

| block | assertion | worst observed |
| --- | --- | --- |
| `β` | `maximum(abs.(est - β_true)) ≤ 0.2` | 0.12 |
| `Σc` | `maximum(abs.(est - Σc_true) ./ abs.(Σc_true)) ≤ 0.3` | 0.17 |
| `Στ` diagonal | `Στ_true[i,i]/3 ≤ est[i,i] ≤ 3*Στ_true[i,i]` | ratio 2.3 (upward) |
| `Στ` off-diagonal | `abs(est[1,2]) ≤ 0.01` | 0.0041 |

applied to the posterior mean and the posterior median separately.

Two things this measurement settled, both worth keeping in the file as comments:

- **The trend-innovation covariance is systematically over-estimated** (≈1.7–2.3× the
  true 0.01 for the first trend, and it does not shrink when `T` goes 400 → 800). It is
  the weakly identified block of a UC model, so its tolerance is a factor band, not a
  percentage. Worth a separate look later; the test documents the bias instead of hiding it.
- **90% credible-interval containment is not a reliable assertion**: `Στ[1,1]` fell
  outside its own 90% band in all three seeds, and `β`/`Σc` missed occasionally. Assert
  point-estimate tolerances only.

Runtime: ≈6 ms/draw at this size (measured), so ≈20 s for the 3000 draws, plus ~10 s of
compilation. `tcvar_priors_test.jl` is ~1 s.

### 8. Run

```
julia --project test/runtests.jl
```

Run the file directly rather than `Pkg.test()`: `Pkg.test()` resolves a separate test
environment and re-precompiles, while the direct call reuses the project's own manifest.
`Test` and the other stdlibs resolve from `@stdlib` on the default `LOAD_PATH`, so the
`[extras]`/`[targets]` entries in `Project.toml` are not needed for this path (they stay
there so `Pkg.test()` keeps working too).

`test/runtests.jl` keeps its existing (empty) `MacroFinanceScenarios` testset, loads the
TCVAR module once at top level, and includes the three TCVAR files:

```julia
using MacroFinanceScenarios
using Test

include(joinpath(@__DIR__, "..", "src", "TCVAR", "TCVAR.jl"))
using .TCVAR

@testset "MacroFinanceScenarios.jl" begin
    # Write your tests here.
end

include(joinpath(@__DIR__, "tcvar_test_utils.jl"))
include(joinpath(@__DIR__, "tcvar_priors_test.jl"))
include(joinpath(@__DIR__, "tcvar_recovery_test.jl"))
```

Each included file still carries its own `@testset` and `isdefined(Main, :TCVAR) || include(…)`
guard, so a single file can also be run on its own
(`julia --project test/tcvar_priors_test.jl`) while iterating.

## Follow-ups (out of scope)

- Delete the superseded `src/TCVAR/priors-refactor-plan.md`.
- `src/TCVAR/SV_priors.jl` already sketches an `sv_priors` tuple; the same key-per-parameter
  convention should carry into the TCVAR-SV sampler.
