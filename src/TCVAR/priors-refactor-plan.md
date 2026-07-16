# Refactor TCVAR priors to Distributions.jl-based named tuple

## Context

Today `TCVAR` takes a priors named tuple of raw arrays/scalars (`initial_trend_mean`, `initial_trend_covariance`, `trend_covariance_df`, `trend_covariance_mean`, `initial_cycle_mean`), and `gibbs_sampler` reassembles distributions from them (e.g. IW scale = `mean * (df + n + 1)`). The refactor keeps the named-tuple shape but each entry becomes a Distributions.jl distribution keyed by parameter name, so the prior semantics live in one place and the sampler just reads `mean`/`cov`/`params` off the distribution objects.

Decisions confirmed with the user:
- **Preserve current numerics**: the notebook builds `InverseWishart(df, mean * (df + n + 1))` so results are bit-identical to today; the sampler contains no conversion formula.
- **Drop `initial_cycle_mean`**: the cycle is mean-zero by model assumption; hard-code `zeros(n)` in the sampler.
- `cycle::MinnesotaPrior` stays as-is (it's a structured hierarchical prior, not a single Distributions.jl object).

## New priors API (what the notebook will contain)

```julia
priors = (
    initial_trend    = MvNormal([2., .5, 1., 2., 5.], diagm(fill(1., nt))),
    trend_covariance = InverseWishart(100, diagm([2., 1., 1., 2., 1.].^2 ./ 400) .* (100 + nt + 1)),
)
model = TCVAR.TCVAR(observation_trend_mapping, priors, cycle_prior; variable_names=..., trend_names=...)
```

## Changes

### 1. `src/TCVAR/TCVAR_model.jl`
- **`TCVAR` struct docstring** (~line 99): describe the new keys — `initial_trend::MvNormal` (initial trend state prior), `trend_covariance::InverseWishart` (trend innovation covariance prior), plus `cycle::MinnesotaPrior`.
- **`TCVAR` constructor docstring + validation** (~line 121): add dimension checks alongside the existing ones:
  - `length(priors.initial_trend) == n_trends`
  - `size(priors.trend_covariance) == (n_trends, n_trends)` (Distributions.jl defines `size` for matrix-variate distributions)

### 2. `src/TCVAR/gibbs_sampler.jl`
Replace raw-field reads with distribution accessors:
- Top of `gibbs_sampler`, extract once:
  ```julia
  ν_τ, Ψ_τ = params(priors.trend_covariance)   # (df, PDMat)
  trend_covariance_scale = Matrix(Ψ_τ)
  initial_trend_mean = mean(priors.initial_trend)
  initial_trend_covariance = cov(priors.initial_trend)
  initial_cycle_mean = zeros(n_obs)             # cycle is mean-zero by assumption
  ```
- Line 58: `dτ_post = n_trend_time_steps - 1 + ν_τ`
- Line 75: delete the `trend_covariance_scale = priors.trend_covariance_mean * (...)` conversion line (scale now comes from Ψ directly).
- Line 78: `initial_cycle_mean = repeat(initial_cycle_mean, p)` — use the local zeros vector.
- Lines 84–86: use the local `initial_trend_mean` / `initial_trend_covariance`.
- Line 97 (initial parameter value): `trend_covariance[1, :, :] = mode(priors.trend_covariance)`. (IW mode = `Ψ/(ν+n+1)`, which with the preserved-behavior Ψ equals exactly the old `trend_covariance_mean` seed — keeps the sampler bit-identical.)
- Update the docstring (~line 16) to describe the new prior keys.

### 3. `analisys/MarcoFinanceTCVAR1.ipynb` (via NotebookEdit)
- Rewrite the priors cell (currently the 5-field tuple) to the new API shown above, with a short comment noting `Ψ = M ⋅ (df + n + 1)` makes `M` the IW mode (preserves previous behavior).
- The selected line 5 (`priors, cycle_prior`) and the `TCVAR.TCVAR(...)` call keep working unchanged.
- Leave the stale old-signature `gibbs_sampler(data, mapping, priors, cycle_prior; ...)` cell and other notebooks alone unless asked.

## Verification

1. `julia --project -e 'using Pkg; Pkg.test()'` (or run `test/runtests.jl`) — tests don't construct priors, so they must stay green.
2. End-to-end smoke script in the scratchpad: build a small `TCVAR` with the new priors tuple (2 vars, 2 trends, synthetic data), run `gibbs_sampler(model, data; burnin=50, n_samples=50)`, check it returns chains of the right shapes.
3. Numerical preservation check: with a fixed RNG seed, run the sampler once on the old code path and once on the new one with `InverseWishart(df, mean*(df+n+1))` and confirm identical `trend_covariance` / `betas` draws (seed value, scale, and df all match, so the chains should be bit-identical).
