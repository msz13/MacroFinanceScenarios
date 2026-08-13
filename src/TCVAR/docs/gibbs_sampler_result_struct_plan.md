# Plan: Return `gibbs_sampler` results as a struct

## Motivation

`gibbs_sampler` currently returns a 5-tuple that mixes two conceptually different
kinds of output:

- **Raw sample arrays** — `t_trends_states`, `t_cycle_states`
- **`Chains` objects** — `t_trend_covariance`, `t_betas`, `t_sigmas`

```julia
return t_trends_states, t_cycle_states, t_trend_covariance, t_betas, t_sigmas
```

Positional destructuring is error-prone (easy to swap `betas`/`sigmas`), carries no
provenance (burnin, thinning, lags), and grows awkward if diagnostics are added.
Wrapping the output in a struct makes the result self-describing and safer to consume.

## Two design options

### Option A — Flat struct mirroring the current tuple

One field per returned quantity. Minimal change; callers switch from positional
destructuring to named fields.

```julia
struct GibbsResult
    trends_states::Array{Float64,3}      # n_kept × n_trend_time_steps × n_trends
    cycle_states::Array{Float64,3}       # n_kept × n_cycle_time_steps × n_obs
    trend_covariance::Chains
    betas::Chains
    sigmas::Chains
end

# end of gibbs_sampler:
return GibbsResult(t_trends_states, t_cycle_states,
                   t_trend_covariance, t_betas, t_sigmas)
```

**Pros**
- Smallest diff; one-to-one with today's outputs.
- Ergonomic access (`res.sigmas`); good for interactive notebook use.
- Optional `Base.iterate` overload keeps old destructuring working during migration.

**Cons**
- Doesn't express the states-vs-parameters split.
- No provenance / sampler settings.
- Flat growth as diagnostics are added.

### Option B — Grouped struct separating states from parameters (+ metadata)

Group by role, and carry the sampler configuration used to produce the draws.

```julia
struct GibbsStates
    trends::Array{Float64,3}
    cycle::Array{Float64,3}
end

struct GibbsParams
    trend_covariance::Chains
    betas::Chains
    sigmas::Chains
end

struct GibbsResult
    states::GibbsStates
    params::GibbsParams
    settings::NamedTuple   # (; burnin, n_samples, thin, p, n_obs, n_trends)
end

# usage: res.params.sigmas, res.states.trends, res.settings.thin
```

**Pros**
- Encodes model structure (states vs. parameters) in the type.
- `settings` carries provenance for reproducible save/reload pipelines.
- Extensible: add diagnostics/timing without touching state/param groups.

**Cons**
- Larger diff; deeper access paths (`res.params.sigmas`).
- Two extra types to maintain.

### Trade-off summary

| | Option A (flat) | Option B (grouped + meta) |
|---|---|---|
| Diff size | Minimal | Moderate |
| Access | `res.sigmas` | `res.params.sigmas` |
| Provenance | none | `res.settings` |
| Extensibility | flat growth | clean grouping |

**Recommendation:** Option A if consumption is mostly interactive notebook analysis
(`analisys/*.ipynb`). Option B if results are saved/reloaded in a reproducible
pipeline where draw settings must be recoverable later.

## Implementation steps

1. Define the chosen struct(s) — likely near the top of `gibbs_sampler.jl` or in a
   dedicated file, and export from `TCVAR.jl`.
2. Replace the final `return` in `gibbs_sampler` with a struct construction.
3. (Option A only, if needed) add a `Base.iterate`/`Base.getindex` overload so existing
   positional-destructuring call sites keep working during migration.
4. Update all call sites:
   - `analisys/*.ipynb` notebooks
   - any tests under `test/`
5. Run the test suite and re-run the affected notebook cells to confirm parity.

## Open questions

- Confirm actual call sites and how the tuple is currently destructured before choosing
  A vs. B (grep across notebooks + tests).
- Should `settings` (Option B) also include the prior objects for full reproducibility,
  or just scalar sampler config?
