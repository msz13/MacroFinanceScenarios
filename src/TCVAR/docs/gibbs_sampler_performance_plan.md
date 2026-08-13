# Gibbs sampler performance plan

Analysis of `gibbs_sampler.jl` and its hot-loop dependencies, with a prioritized
plan for speeding up the sampler. References use `file:line`.

## How the cost is structured

`gibbs_sampler` runs `burnin + n_samples` draws. Each draw (`gibbs_sampler.jl:147-174`) does:

1. `sample_states` -> `carter_kohn_sampler` -> `kalman_filter` (the `StateSpaceModel`
   version in `carter_kohn_algorythm.jl`)
2. trend covariance draw
3. `sample_var_params` -> `draw_beta` (+ stationarity rejection loop)
4. `tc_var` — full model rebuild
5. `update_initial_cycle_covariance!` -> `stationary_cycle_covariance`

The dominant costs are dense linear-algebra calls that use the most expensive
available method (`pinv` / `eigen` / `svd`) where a Cholesky solve would do, plus
several quantities recomputed every draw that are actually constant.

## Findings, by impact

### 1. `stationary_cycle_covariance` — likely the single biggest cost
`gibbs_sampler.jl:31`
```julia
vecP = pinv(I - kron(model.T, model.T)) * vec(model.Q)
```
`kron(T,T)` is `n_states^2 x n_states^2` and `pinv` is an SVD — roughly
`O(n_states^6)` **every draw**. With `n_states ~ 15-20` that's a 200-400 square
`pinv` per sweep.

It only exists to handle the singular trend block. But the trend is a decoupled
random walk; only the **cycle** companion is stationary and it is *guaranteed
stable* by the `is_stationary` rejection in `sample_var_params`. Solve the
Lyapunov equation on the cycle block alone with a plain solve:
```julia
Tc = model.T[n_trends+1:end, n_trends+1:end]
Qc = model.Q[n_trends+1:end, n_trends+1:end]
vecP = (I - kron(Tc, Tc)) \ vec(Qc)   # invertible, no pinv
```
Dimension drops from `n_states^2` to `n_cycle_states^2` and SVD -> LU.

**Effort: low. Impact: very high. Risk: low (isolated to one function).**

### 2. Kalman filter / smoother use `pinv` + `eigen` per time step — DONE
`carter_kohn_algorythm.jl`
- `kalman_filter:60` `pinv(innovation_covariance)` every t
- `carter_kohn_sampler:168` `pinv(covariance_predicted_t_plus_1)` every t
- `eigen_sqrt` (`:8`) does a full `eigen` for every sampled state, every t

The standalone Cholesky version in `kalman_filter_smoother.jl` could not be used
verbatim: it handles neither **missing observations** (the Del Negro data sets
`BILL` missing after 2008Q4) nor the **t = 0 initial-state draw** that
`sample_states` depends on, and it is not even `include`d in the module. Instead
the Cholesky technique was brought into the existing `kalman_filter` /
`carter_kohn_sampler`, preserving their interfaces, the missing-data subsetting,
and the pre-sample draw:

- every `pinv(S)` gain solve → a Cholesky solve `(P·Zᵀ) / chol_psd(S)`;
- every `eigen_sqrt(cov)*randn` draw → `sample_mvn` = `mean + chol_psd(cov).L*randn`;
- `model.R*model.Q*model.R'` is now hoisted out of the time loop (the step-5 piece).

`chol_psd` (new helper) symmetrizes, then `cholesky(...; check=false)` with a
trace-scaled jitter fallback so the near-singular innovation covariance from
`H = eps()*I` stays factorable.

Validated: against an inline `pinv` reference filter the deterministic filtered /
predicted means and covariances match to ~1e-14 on a TC-VAR(2) with a missing
series; FFBS draws stay finite and `sample_states` returns the right shapes. (Note:
bit-identical draws are *not* expected under a fixed seed — a Cholesky factor is a
different matrix square root than `eigen_sqrt`, so the same `randn` maps to a
different point of the same distribution; equivalence is checked on the moments.)

**Effort: medium. Impact: high. Risk: medium (swaps the sampling kernel — moment
equivalence checked against the pinv reference).**

### 3. `draw_beta` — `eigen` of the full kron, recomputed inside the rejection loop
`gibbs_var_steps.jl:149`
```julia
beta_var = kron(Σ, inv(X'X + Ω_inv)) + I(m)*1e-5
F = eigen(Hermitian(beta_var))      # m x m eigdecomp, m = n*k
```
Two problems:
- (a) `beta_var` is **identical across every rejection iteration**
  (`sample_var_params:224`), yet `draw_beta` is called afresh each time, redoing
  the eigendecomposition up to `max_draws = 100x`.
- (b) the factor of a Kronecker product is the Kronecker of the factors — no need
  to form/decompose the full `m x m`:
```julia
L = kron(cholesky(Symmetric(Σ)).L, cholesky(Symmetric(inv(X'X + Ω_inv))).L)
```
Factor **once** before the loop, then each rejection draw is just
`vec(β̂) + L * randn(m)`.

**Effort: low. Impact: high. Risk: low.**

### 4. Rebuilding the whole `tc_var` model every draw
`gibbs_sampler.jl:135,156`

`R`, `Z`, `H`, `initial_*_mean`, and the trend blocks of `T` / initial-cov never
change. Only the cycle block of `T` and the two covariance blocks of `Q` change
per draw. Since `StateSpaceModel` holds mutable arrays, build once and mutate
those blocks in place — avoids allocating several dense `n_states^2` matrices each
sweep.

**Effort: medium. Impact: medium. Risk: low-medium.**

### 5. `kalman_filter` recomputes constants and stores cache-unfriendly slices
`carter_kohn_algorythm.jl`
- `model.R * model.Q * model.R'` (`:37,:40`) recomputed every t, but `R = I` so it
  is just `Q`, constant per draw.
- Covariances stored as `cov[t,:,:]` (time in the **first** dim) -> every
  `[t-1,:,:]` slice is strided / non-contiguous and copies. Store as `[:,:,t]` or a
  `Vector{Matrix}`.

Mostly folded into step 2, whose existing implementation already stores `[:,:,t]`.

**Effort: low. Impact: medium. Risk: low.**

### 6. Minor: `prepare_var_data` rebuilds the lag matrix each draw
`gibbs_var_steps.jl:116` — row-wise assignment each draw; preallocate / use views.
Low priority next to 1-4.

## Suggested order

| Step | Change | Effort | Impact |
|------|--------|--------|--------|
| 1 | Cycle-only Lyapunov solve (drop full-system `pinv`) | low | very high |
| 2 | Cholesky filter/smoother in `carter_kohn_algorythm.jl` (DONE) | medium | high |
| 3 | `draw_beta`: kron-factor once, outside rejection loop | low | high |
| 4 | Build model once, mutate cycle/cov blocks in place | medium | medium |
| 5 | Precompute `Q`, fix slice layout (folded into 2) | low | medium |
| 6 | `prepare_var_data` preallocation / views | low | low |

## StaticArrays.jl considered

Question: would switching the hot-loop arrays to `StaticArrays.jl` (`SMatrix` /
`SVector`) help? Answer for this model: **no — and it would likely hurt.**

### Actual problem sizes (Del Negro replication)

From `analisys/del negro replication.ipynb`: `n_obs = 5`, `n_trends = 3`, `p = 4`.

| Object | Size | Where |
|--------|------|-------|
| state covariance `P`, `T`, `Q` (per time step in Kalman) | **23 x 23** | `n_trends + n_obs*p = 3 + 20` |
| `kron(T,T)` in `stationary_cycle_covariance` | **529 x 529** | finding #1 |
| `beta_var` in `draw_beta` | **100 x 100** | `m = n_obs*k = 5*20` |
| innovation cov `S` | 5 x 5 | per time step |
| cycle cov `Σ`, trend cov | 5 x 5, 3 x 3 | per draw |

### Why it does not help here

StaticArrays' sweet spot is fixed-size matrices up to roughly **12 x 12**
(multiply degrades around 14-16; `inv` / `cholesky` / `eigen` and compile time
degrade earlier). Above that, `SMatrix` is slower than BLAS and `MArray` overflows
the stack, while compile times explode from full loop unrolling.

The dominant costs here are all **above** that threshold:
- The per-time-step Kalman workhorse runs on **23 x 23** matrices — over the line;
  StaticArrays would be worse than the regular `Array` / BLAS path.
- `kron(T,T)` at **529 x 529** and `beta_var` at **100 x 100** are nowhere near —
  these are exactly the matrices the algorithmic fixes (cycle-only Lyapunov,
  Kronecker-factored `draw_beta`) target, and they stay large regardless.

The only StaticArrays-friendly pieces are the 3 x 3 and 5 x 5 blocks, which are not
the bottleneck.

### Practical blockers even where sizes are small

1. **Dimensions are runtime values.** `n_obs`, `p`, `n_trends` come from function
   arguments. StaticArrays needs sizes as compile-time type parameters, so you would
   thread `Val` / static-dim parameters through the whole stack and recompile per
   problem shape — a large refactor.
2. **Library boundaries.** `Distributions.InverseWishart` `rand` and
   `MCMCChains.Chains` do not consume StaticArrays; you would convert at every
   boundary.
3. **Storage.** States / covariances are stored as 3D `Array` slices; you would
   restructure to `Vector{SMatrix}`.

### Recommendation

Skip StaticArrays for this model. The algorithmic fixes above are order-of-magnitude
wins on matrices too big for StaticArrays anyway, whereas StaticArrays would be a
large refactor for, at best, a small constant-factor gain on the already-cheap
3-5-dim blocks.

It would only pay off in a different regime — a small cycle (`n_states <= ~10`, e.g.
1-3 variables with `p = 1`) run for many draws. If the sampler is used heavily in
that small-model regime, a targeted, measured try on the Kalman inner loop could be
worthwhile *after* the algorithmic fixes land.

## Validation

- Add a `@btime` (BenchmarkTools) harness on a small fixed config to measure each
  change rather than assume it.
- For step 2 (and any change that touches the sampling kernel), verify identical
  draws under a fixed RNG seed before/after.
