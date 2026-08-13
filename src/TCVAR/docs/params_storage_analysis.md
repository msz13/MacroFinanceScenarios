# Storing estimated parameters in `TCVarResult`

Analysis of how to keep the Gibbs-sampled parameters (`Στ`, `β`, `Σc`) so they
serve both **reporting** (convergence plots, summary tables split by parameter
kind and by variable, state plots) and **simulation** (with the posterior mean,
or with all draws).

## Options considered

- **A. NamedTuple of raw arrays** — `(Στ=Array, β=Array, Σc=Array)`.
- **B. Single `FlexiChain{VarName}`** with matrix-valued params (current design).
- **C. NamedTuple of FlexiChains** — one `FlexiChain` per parameter kind.

## Bottom line

**Keep the single `FlexiChain{VarName}` with matrix-valued parameters (option B,
the current design).** It is the only option that satisfies *all* the reporting
requirements out of the box while staying clean for simulation. Add a couple of
thin accessor helpers for the simulation side. Do not move to raw arrays; only
move to a NamedTuple-of-FlexiChains if you specifically want dot-access
ergonomics — that is a preference, not a capability gain.

## Why — mapped to the requirements

| Requirement | Raw arrays (A) | **Single FlexiChain (B)** | NamedTuple of FlexiChains (C) |
|---|---|---|---|
| Convergence / trace plots | hand-roll everything | `plot(params[@varname(β)])` ✅ | `plot(params.β)` ✅ |
| Summary split **by kind** (coef vs cov) | manual | index by key: `st[@varname(β)]`, `st[@varname(Σc)]` ✅ | structural: `params.β` ✅ |
| Summary split **by variable** (var-1 coefs…) | manual reshaping | slice: `params[@varname(β[:,1])]` ✅ | needs per-element keys inside each chain |
| mean over draws (simulation) | trivial `mean(...)` ✅ | `mean` over matrix-valued draws → matrix ✅ | reconstruct matrix from scalars ⚠️ |
| all draws (simulation) | trivial ✅ | iterate matrix-valued draws ✅ | reassemble per draw ⚠️ |
| 100+ params, one script | tons of boilerplate ❌ | native machinery ✅ | native, but 3× the calls |

The decisive point: FlexiChains treats each of `Στ` / `β` / `Σc` as **one
matrix-valued VarName**, so a single object gives you *both* granularities:

- **Simulation** wants whole matrices — each draw of `params[@varname(β)]` is
  already a `k×n_obs` matrix, so `mean(...)` gives the posterior-mean β directly
  and iterating gives all draws. This is where matrix-valued keys beat
  per-scalar keys (option C), which would force reassembling matrices from
  `β[i,j]` scalars.
- **Reporting** wants per-scalar — `summarystats(params; split_varnames=true)`
  expands to per-element rows, and `params[@varname(β[:,1])]` slices out
  variable 1's coefficient column for its own table/plot.

Raw arrays (A) do the simulation half trivially but throw away every bit of the
reporting machinery already in use — for a 100+ parameter model that is a lot of
hand-written trace-plot / summary / labeling code. Reject A as the *storage*
type (it is fine as a transient: derive raw matrices on demand from the chain).

## What to add

Thin accessors so simulation code never touches FlexiChains internals:

```julia
# posterior-mean matrices for point-simulation
posterior_mean(r::TCVarResult) = (
    Στ = mean(r.params[@varname(Στ)]),
    β  = mean(r.params[@varname(β)]),
    Σc = mean(r.params[@varname(Σc)]),
)

# matrices for draw s, for full-posterior simulation
draw(r::TCVarResult, s) = (
    Στ = r.params[@varname(Στ)][s],
    β  = r.params[@varname(β)][s],
    Σc = r.params[@varname(Σc)][s],
)
```

## When to prefer NamedTuple-of-FlexiChains (C) instead

Only if `st[@varname(β)]` key-indexing feels clunky and you want `params.β`
dot-access plus separate `plot(params.β)` figures per kind. That is an ergonomic
taste, not a functional gain — a single chain already splits by kind
(key-index) and by variable (slice), and it *keeps* the clean matrix-valued
draws for simulation that option C gives up.
