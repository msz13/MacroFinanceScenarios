# Gibbs Sampler Comparison: Julia vs MATLAB Reference

**Julia**: `src/TCVAR/gibbs_sampler.jl`  
**Reference**: [rstarBrookings2017 / tvar / MainModel1.m](https://github.com/FRBNY-DSGE/rstarBrookings2017/blob/master/tvar/MainModel1.m)

Both implement a Gibbs sampler for a Trend-Cycle VAR model (state-space form) using Carter-Kohn backward simulation. The main loop structure is the same: sample states → sample trend covariance → sample VAR parameters (beta, sigma). However, there are meaningful algorithmic differences.

---

## Algorithmic Differences

| Feature | MATLAB | Julia |
|---|---|---|
| **VAR lags** | `p=4` | `p=1` (hardcoded) |
| **Stationarity enforcement** | Resamples VAR coefficients until all companion matrix eigenvalues are inside the unit circle | No stationarity check — can draw explosive VAR |
| **Initial cycle covariance `P0`** | Updated **every iteration** via Lyapunov equation: `pinv(I − kron(A,A)) * vec(Q)` | Fixed at `priors.initial_cycle_covariance`, never updated |
| **Initial trend state `S0`** | Metropolis-Hastings random-walk step with log-likelihood acceptance | Not implemented |
| **Log-likelihood** | Computed in Kalman filter, used in MH acceptance ratio | Hardcoded to `0.0` (TODO in `kalman_filter_smoother.jl:59`) |
| **Missing data** | NaN-aware Kalman filter | `Missing`-aware Kalman filter ✓ |
| **Burnin** | 50% of `Ndraws` | Separate `burnin` parameter |

---

## Beta / Sigma Posterior — Algebraically Equivalent

Both use conjugate matrix-normal inverse-Wishart posteriors:

- **Posterior coefficient mean**: `β̂ = (X'X + Ω⁻¹)⁻¹ (X'Y + Ω⁻¹ β₀)` — identical
- **Sigma posterior scale**: `ε'ε + (β̂ − β₀)' Ω⁻¹ (β̂ − β₀) + S₀` — identical structure
- **Beta drawn** conditional on sigma with covariance `kron(Σ, (X'X + Ω⁻¹)⁻¹)` — identical

### Minnesota prior shrinkage

MATLAB applies lag-decay: `ω_{i,j} = (d−n−1) · λ² · (1/i^α) / ψⱼ` with `α=2`.  
Julia uses `Ω = λ² / diag(Σ_c)` with no decay — equivalent to MATLAB at `p=1`, but would diverge if lags were increased.

---

## Most Critical Missing Pieces

### 1. Stationarity check (`gibbs_var_steps.jl: sample_var_params`)

MATLAB's `BVAR.m` loops until the drawn companion matrix has all eigenvalues inside the unit circle:

```matlab
stationary = 0;
while stationary == 0
    % ... draw beta, sigma ...
    AA(1:n, 1:n*lags) = beta';
    AA(n+1:end, 1:n*(lags-1)) = eye(n*(lags-1));
    stationary = (sum(abs(eig(AA)) > 1) == 0);
end
```

Julia's `sample_var_params` does no such check. An explosive draw will propagate into subsequent state sampling and can destabilize the chain.

### 2. Lyapunov update of initial cycle covariance (`gibbs_sampler.jl`)

After sampling a new VAR companion matrix `A_cyc`, MATLAB recomputes the stationary covariance and uses it as the initial covariance for the cycle block in the next Kalman filter pass:

```matlab
vecP0full = pinv(eye((r+n*p)^2) - kron(A,A)) * Q(:);
P0full = reshape(vecP0full, r+n*p, r+n*p);
P0(r+1:end, r+1:end) = P0full(r+1:end, r+1:end);
```

Julia keeps `initial_cycle_covariance` fixed at the prior value throughout all iterations. This breaks the feedback between VAR dynamics and the filter initialisation.

### 3. MH step for initial trend state

MATLAB proposes a random-walk perturbation to `S0` (initial trend mean), evaluates the log-likelihood ratio, and accepts/rejects. Julia has no equivalent, so the initial trend state is never directly sampled.

---

## Minor Bugs in Julia

| Location | Issue |
|---|---|
| `gibbs_sampler.jl:86` | `t_betas` chain uses `cycle_covariance_names(k)` — should be `coeff_names(k)` |
| `gibbs_sampler.jl:45` | `trend_covariance_scale` uses `n_obs` instead of `n_trends` in the IW scale formula — wrong when `n_obs ≠ n_trends` |
| `gibbs_sampler.jl:24` | Function name `gibs_sampler` is missing a `b` |
| `kalman_filter_smoother.jl:59` | Log-likelihood contribution hardcoded to `0.0` |
