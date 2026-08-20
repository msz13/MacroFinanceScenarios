# TCVAR-SV — trend-cycle VAR with multivariate stochastic volatility

Implementation plan for extending the Gibbs sampler of `models/tcvar/` with a stochastic
volatility block, following Cogley & Sargent (2005) for the constant simultaneity matrix
and Kim, Shephard & Chib (1998) for the mixture approximation of the volatility
measurement equation.

This plan continues `file_structure_refactor_plan.md` (stages 0–1 and 3 are committed;
this document supersedes its stages 4–5 and folds in the parts of stage 2 that TCVAR-SV
actually needs). **The guiding rule stays the same: everything a second model could reuse
goes into `common/` or `var/`; `models/tcvar_sv/` contains only the SV-specific state
space, its priors, the sweep and the result type.** The existing TCVAR sampler must come
out of this bit-for-bit unchanged.

---

## 1. The model

Observation (unchanged from TCVAR — `Λ` is the `trend_mapping`):

```
y_t = Λ τ_t + c_t                                     t = 1 … T
```

Trends — random walk (unchanged):

```
τ_t = τ_{t-1} + u_t,          u_t ~ N(0, Στ)
```

Cycle — VAR(p) with **time-varying innovation covariance**:

```
c_t = A_1 c_{t-1} + … + A_p c_{t-p} + ε_t,            ε_t ~ N(0, Σ_t)
```

written in the codebase's oldest-lag-first regression layout `c_t = B' x_t + ε_t`,
`x_t = [c_{t-p}; …; c_{t-1}]` (`B` is `k × n`, `k = n·p`, `β = vec(B)`).

Volatility factorisation — **constant simultaneity matrix, time-varying variances**
(Cogley–Sargent 2005):

```
Σ_t = A₀⁻¹ H_t A₀⁻ᵀ,     H_t = diag(exp(h_{1t}), …, exp(h_{nt}))
```

`A₀` is unit lower triangular; its `n(n-1)/2` free elements below the diagonal are the
"correlation parameters L" of the spec. Equivalently `A₀ ε_t = u_t` with
`u_t ~ N(0, H_t)` diagonal — that is the form every conditional posterior below is
written in. The spec's `L` is `A₀⁻¹` (`Σ_t = L H_t Lᵀ`); we draw `A₀` because *that* is
the parameterisation with a conjugate conditional, and expose `L = A₀⁻¹` on the result.

> **Precision note.** "Constant correlations" is Cogley–Sargent's *constant contemporaneous
> impact structure*, not literally constant conditional correlations: with `A₀` fixed and
> `H_t` moving, `corr(Σ_t)` still drifts as relative volatilities change. A genuinely
> constant-correlation model would be `Σ_t = D_t R D_tᵀ` with `D_t = diag(exp(h_t/2))` and
> a fixed correlation matrix `R` — no conjugate step, needs Metropolis. The spec says
> "as cogley sergant", so we take the `A₀` form. Flagging it because the two are easy to
> conflate.

Log volatilities — VAR(1) in the mean-adjusted level, with **correlated** innovations:

```
h_t = μ + Φ (h_{t-1} - μ) + ν_t,     ν_t ~ N(0, Ω)          t = 1 … T
h_0 ~ N(μ, P₀)
```

with `μ ∈ ℝⁿ`, `Φ` an `n × n` AR matrix (default **diagonal**, matching `SV_priors.jl`'s
scalar `ρ` per series and Clark–Ravazzolo 2015; a full matrix is supported), and `Ω` a
full `n × n` covariance. Writing `h̃_t = h_t - μ` gives the plain VAR(1) `h̃_t = Φ h̃_{t-1} + ν_t`
used by every step below. `P₀` defaults to the stationary covariance
`vec(P₀) = (I - Φ⊗Φ)⁻¹ vec(Ω)` and falls back to a supplied diffuse matrix when `Φ` has a
unit root (the random-walk special case; every script here uses the stationary default).

**Parameter vector:** `Στ, β, A₀, h_{0:T}, μ, Φ, Ω` plus the states `τ_{0:T}, c_{-p+1:T}`.

---

## 2. The Gibbs sweep

Eight blocks, in the order the spec lists them. Each cell of the "reuse" column says
whether the step is an existing function, an existing function behind a new caller, or
new code.

| # | Block | Draws | Reuse |
|---|---|---|---|
| 1 | TCVAR states | `τ_{0:T}, c_{-p+1:T}` | `carter_kohn_sampler` + `sample_states`, generalised to time-varying `Q_t` |
| 2 | Trend covariance | `Στ` | `random_walk_covariance_posterior` **verbatim** |
| 3 | Cycle VAR coefficients | `β` | new GLS posterior + existing `is_stationary`, `prepare_var_data`, Minnesota prior |
| 4 | Simultaneity matrix | `A₀` | new: `n-1` weighted univariate regressions on the generic `normal_posterior` |
| 5 | Volatilities | `h_{0:T}` (and mixture indicators `s`) | new SV block, built on `carter_kohn_sampler` |
| 6 | Volatility mean | `μ` | new 5-line assembly on the generic `normal_posterior` |
| 7 | Volatility AR | `Φ` | new assembly (diagonal) / existing `normal_coefficient_posterior_mean` + `kron_cholesky_factor` (full) |
| 8 | Volatility covariance | `Ω` | `inverse_wishart_posterior` **verbatim** |

### Step 1 — TCVAR states

Same Carter–Kohn draw as TCVAR, but the state-noise covariance is now time varying:

```
Q_t = blockdiag(Στ, 0_{n(p-1)}, Σ_t),      Σ_t = A₀⁻¹ H_t A₀⁻ᵀ
```

`A₀⁻¹` is computed once per sweep (unit triangular, `n` small), `Σ_t` for `t = 1…T` from
the current `h`. This is the one structural change to `common/`: see §3.1.

Initial-state covariance of the cycle block is re-initialised each sweep from the
Lyapunov solve at the **unconditional** volatility `Σ̄ = A₀⁻¹ diag(exp(μ)) A₀⁻ᵀ`
(the SV analogue of what `stationary_cycle_covariance` does today with the constant `Σc`).

### Step 2 — trend covariance `Στ`

Unchanged from TCVAR:

```
Στ | τ ~ IW(dτ_post, Δτ'Δτ + Ψτ),     Δτ = diff(τ_{0:T})
```

> The spec annotates this step "estimate posteriors with gls". The trend block has no
> regressors and homoskedastic innovations, so GLS collapses onto exactly this conjugate
> IW draw — no change is needed here. Where GLS genuinely enters is steps 3, 4, 6 and 7,
> all of which are heteroskedastic. **Decision D1 below flags this for confirmation.**

### Step 3 — cycle VAR coefficients `β` (GLS, heteroskedastic)

With `Σ_t` time varying the NIW conjugacy of `sample_var_params` is gone: `Σ` is no longer
a free parameter that integrates out. The conditional of `vec(B)` is Normal:

```
c_t = (I_n ⊗ x_tᵀ) vec(B) + ε_t,   ε_t ~ N(0, Σ_t)

P_d = Σ_t  Σ_t⁻¹ ⊗ x_t x_tᵀ                    (nk × nk)
b_d = Σ_t  vec( x_t c_tᵀ Σ_t⁻¹ )               (nk)

vec(B) | · ~ N( (P₀ + P_d)⁻¹ (P₀ vec(B₀) + b_d),  (P₀ + P_d)⁻¹ )
```

Prior precision `P₀ = Σ̄⁻¹ ⊗ Ω_M⁻¹` where `Ω_M = prior_row_covariance(β_prior)` is the
Minnesota row covariance already in the codebase and `Σ̄ = mean(priors.cycle_covariance)`
is the prior mean of the cycle innovation covariance — the Minnesota prior is
`Φ|Σ ~ MN(Φ₀, Ω_M, Σ)` and with SV there is no single `Σ` to condition on, so it is fixed
at its prior mean. This is the standard Cogley–Sargent choice; recorded as **D2**.

Same stationarity-rejection loop as `sample_var_params` (`is_stationary` on the companion
bottom block, `max_draws` cap).

Cost: `T · (n²k²)` accumulation — for `n = 5, p = 2` that is 400 × 2500 flops per sweep,
negligible next to the Kalman pass.

### Step 4 — simultaneity matrix `A₀`

`A₀ ε_t = u_t` with `u_t ~ N(0, H_t)` diagonal means row `i` is a heteroskedastic
univariate regression, exactly as in Primiceri (2005) §Appendix:

```
ε_{it} = -Σ_{j<i} A₀[i,j] ε_{jt} + u_{it},     Var(u_{it}) = exp(h_{it})
```

so with regressors `z_t = -[ε_{1t}, …, ε_{i-1,t}]` and prior `a_i ~ N(a_{i0}, V_{i0})`:

```
P_d = Σ_t  z_t z_tᵀ / exp(h_{it})
b_d = Σ_t  z_t ε_{it} / exp(h_{it})
a_i | · ~ N( (V_{i0}⁻¹ + P_d)⁻¹ (V_{i0}⁻¹ a_{i0} + b_d), (V_{i0}⁻¹ + P_d)⁻¹ )
```

Drawn row by row, `i = 2 … n`; row 1 has no free element. The regressor sign convention
is fixed in the code so the drawn coefficients **are** `A₀[i,j]` with no post-hoc negation.

`ε_t = c_t - Bᵀ x_t` are the cycle residuals from the just-drawn `β` (step 3), so this
step follows step 3, matching the spec's ordering.

> `A₀` is a Cholesky-type identification: results depend on the variable ordering in
> `y`. Documented on the model struct.

### Step 5 — volatilities `h` (the SV block, KSC mixture)

The reusable piece, and the one the spec asks for as a standalone function. Input:
residuals and volatility parameters; output: `h_{0:T}` and the mixture indicators.

Orthogonalised residuals `e_t = A₀ ε_t`, then per series `i`:

```
y*_{it} = log(e_{it}² + c̄),          c̄ = 0.001        (offset, KSC)
y*_{it} = h_{it} + z_{it},           z_{it} ~ log χ²₁
```

`log χ²₁` is approximated by the 7-component Gaussian mixture of Kim, Shephard & Chib
(1998, Table 4) — the "Chib approximation" of the spec:

| j | q_j | m_j | v_j² |
|---|---|---|---|
| 1 | 0.00730 | -10.12999 | 5.79596 |
| 2 | 0.10556 | -3.97281 | 2.61369 |
| 3 | 0.00002 | -8.56686 | 5.17950 |
| 4 | 0.04395 | 2.77786 | 0.16735 |
| 5 | 0.34001 | 0.61942 | 0.64009 |
| 6 | 0.24566 | 1.79518 | 0.34023 |
| 7 | 0.25750 | -1.08819 | 1.26261 |

with `m_j` shifted by `-1.2704 = E[log χ²₁] = ψ(1/2) + log 2` (a unit test asserts
`Σ q_j (m_j - 1.2704) ≈ -1.2704` and that `Σ q_j = 1`).

**5a — mixture indicators `s`** (independent across `i` and `t`):

```
P(s_{it} = j | ·) ∝ (q_j / v_j) exp( -(y*_{it} - h_{it} - m_j)² / (2 v_j²) )
```

**5b — log volatilities `h`** — conditional on `s` the system is linear Gaussian. Demean
by `μ` and remove the mixture means so it fits `StateSpaceModel` exactly:

```
measurement:   ỹ_t = h̃_t + z_t,        ỹ_t = y*_t - m_{s_t} - μ,    z_t ~ N(0, diag(v²_{s_t}))
state:         h̃_t = Φ h̃_{t-1} + ν_t,  ν_t ~ N(0, Ω)
```

i.e. `T = Φ, R = I, Z = I, Q = Ω, H_t = diag(v²_{s_t})` — a **time-varying observation
covariance**, the second structural requirement on `common/` (§3.1). Hand it to the
existing `carter_kohn_sampler`; its `initial_state` return is `h̃_0`, which steps 6–8 need.
Add back `μ` to get `h_{0:T}`.

The alignment with the rest of the sweep: `prepare_var_data` on the `T+p`-point cycle path
yields exactly `T` residual rows, so `h_{1:T}` lines up one-for-one with the periods whose
`Σ_t` step 1 consumes.

### Step 6 — volatility mean `μ`

`h̃_t = h_t - μ` gives `h_t - Φ h_{t-1} = (I - Φ) μ + ν_t`, a regression on the constant
design `M = I - Φ` with known `Ω`. With prior `μ ~ N(μ₀, V₀)`:

```
P_d = T · Mᵀ Ω⁻¹ M
b_d = Mᵀ Ω⁻¹ Σ_t (h_t - Φ h_{t-1})
μ | · ~ N( (V₀⁻¹ + P_d)⁻¹ (V₀⁻¹ μ₀ + b_d), (V₀⁻¹ + P_d)⁻¹ )
```

(The `h_0` draw is used through `h_1 - Φ h_0`; the `h_0 ~ N(μ, P₀)` term is dropped from
the conditional, the usual conditional-likelihood treatment.) `μ` is **not identified when
`Φ = I`** — the sampler skips this block in that case and anchors the level at `h_0`.

### Step 7 — volatility AR coefficients `Φ`

*Diagonal (default).* `h̃_t = D_t ρ + ν_t` with `D_t = Diagonal(h̃_{t-1})`, so with prior
`ρ ~ N(ρ₀, V_ρ)`:

```
P_d = Σ_t D_t Ω⁻¹ D_t        b_d = Σ_t D_t Ω⁻¹ h̃_t
```

and the same `normal_posterior` update. Note this is *not* `n` independent regressions —
the correlated `Ω` couples them, which is why the assembly is written out rather than
looped equation by equation.

*Full matrix (option).* `Y = h̃_{1:T}ᵀ`, `X = h̃_{0:T-1}ᵀ`, natural-conjugate prior
`vec(Φᵀ) ~ N(vec(Φ₀), Ω ⊗ V₀)` reuses **the existing functions verbatim**:
`normal_coefficient_posterior_mean(Y, X, Φ₀, V₀⁻¹)` → `kron_cholesky_factor(Ω, inv(X'X + V₀⁻¹))`
→ `draw_from_factor`.

Both variants reject draws with `is_stationary(Φ, n, 1) == false` (same `max_draws` cap as
the VAR step), so `P₀` in §1 always exists.

### Step 8 — volatility covariance `Ω`

```
ν_t = h̃_t - Φ h̃_{t-1},      Ω | · ~ IW(T + ν_Ω, ν'ν + Ψ_Ω)
```

`inverse_wishart_posterior(ν, Ψ_Ω, T + ν_Ω)` — existing function, no change.

### Sweep pseudocode

```julia
for s in 2:n_draws
    Σ_series  = simultaneity_covariances(A₀, h[:, 1:T])          # Σ_t = A₀⁻¹ H_t A₀⁻ᵀ
    update_tc_var_sv!(ssm, var_coeff, Στ, Σ_series, …)

    τ, c   = sample_states(ssm, data, μ₀_state, P₀_state, n_trends, n_obs; p)   # 1
    Στ     = rand(random_walk_covariance_posterior(τ, Ψτ, dτ_post))       # 2
    β      = draw_var_coefficients_gls(c, p, Σ_series, β₀, P₀_β)          # 3
    ε      = cycle_residuals(c, β, p)
    A₀     = draw_simultaneity(ε, h[:, 1:T], A₀_prior)                    # 4
    h, ω   = draw_stochastic_volatility(A₀ * ε', h, (μ, Φ, Ω))            # 5
    μ      = draw_volatility_mean(h, Φ, Ω, μ_prior)                       # 6
    Φ      = draw_volatility_ar(h, μ, Ω, Φ_prior)                         # 7
    Ω      = rand(volatility_covariance_posterior(h, μ, Φ, Ψ_Ω, dΩ_post)) # 8
end
```

---

## 3. Changes to the shared layers

### 3.1 `common/` — the time-varying state-space seam (the only structural change)

Both new consumers need what `StateSpaceModel` cannot express today: step 1 needs
time-varying `Q`, step 5b needs time-varying `H`. `file_structure_refactor_plan.md`
already scoped the `Q` half; this adds `H` on the same seam.

```julia
# common/state_space.jl
abstract type AbstractStateSpaceModel end

struct StateSpaceModel <: AbstractStateSpaceModel        # fields unchanged
    T; R; Z; Q; H
end

struct TimeVaryingStateSpaceModel <: AbstractStateSpaceModel
    T; R; Z
    Q::Array{Float64,3}      # n_time × n_states × n_states  (or the constant matrix)
    H::Array{Float64,3}      # n_time × n_obs    × n_obs
end

process_noise(m::StateSpaceModel, ::Int)             = m.Q
process_noise(m::TimeVaryingStateSpaceModel, t::Int) = @view m.Q[t, :, :]
observation_noise(m::StateSpaceModel, ::Int)             = m.H
observation_noise(m::TimeVaryingStateSpaceModel, t::Int) = @view m.H[t, :, :]
```

`kalman_filter` / `carter_kohn_sampler` / `sample_states` take `AbstractStateSpaceModel`
and go through the two accessors. **The `RQR = R*Q*R'` hoist stays on the constant-`Q`
path** (dispatch on the concrete type, or hoist once when
`m isa StateSpaceModel`), so TCVAR's arithmetic and therefore its draws are unchanged.

Second, smaller widening: the two signatures pin `observations::Matrix{Union{Missing,Float64}}`
and `initial_state_mean::Vector{Float64}`. The SV block passes a plain `Matrix{Float64}`,
so relax to `AbstractMatrix{<:Union{Missing,Real}}` / `AbstractVector{<:Real}`. Non-breaking.

### 3.2 `common/linalg.jl`

```julia
lyapunov_covariance(F, Q)      # solves P = F P F' + Q  →  reshape((I - kron(F,F)) \ vec(Q))
```

Three call sites today open-code this line (`stationary_cycle_covariance`,
`initial_cycle_prior`, and now the SV `P₀`). It is pure linear algebra with no VAR content,
so it lands in `common/linalg.jl` rather than `var/companion.jl` as the earlier plan had it
— that also keeps `common/sv/` from depending on `var/`.

### 3.3 `common/posteriors.jl` — three additions

Keeping the file's existing split (`*_posterior` pure / `draw_*` consumes RNG):

```julia
"""Conjugate normal update: N( (P₀+P_d)⁻¹(P₀μ₀ + b_d), (P₀+P_d)⁻¹ )."""
normal_posterior(prior_mean, prior_precision, data_precision, data_information) -> MvNormal

"""(P_d, b_d) for a multivariate heteroskedastic regression Y_t = (I ⊗ x_tᵀ)vec(B) + ε_t,
   ε_t ~ N(0, Σ_t):  P_d = Σ Σ_t⁻¹⊗x_t x_tᵀ,  b_d = Σ vec(x_t y_tᵀ Σ_t⁻¹)."""
gls_information(Y, X, Σ_inv_series) -> (Matrix, Vector)

"""(P_d, b_d) for a univariate regression with known per-observation variances."""
weighted_regression_information(y, Z, variances) -> (Matrix, Vector)
```

`normal_posterior` is the single conjugate-normal primitive; steps 3, 4, 6 and diagonal-7
each become a 3–5 line assembly of `(P_d, b_d)` in front of it. That is the whole point of
the split: the model-specific arithmetic is the assembly, and each assembly is separately
checkable against a closed form.

### 3.4 `common/sv/` — new subfolder (model-agnostic, reused by any SV model)

```
common/sv/ksc_mixture.jl     KSC_MIXTURE constants; draw_mixture_indicators(y_star, h)
common/sv/sv_block.jl        draw_stochastic_volatility(...); draw_log_volatilities(...)
common/sv/sv_parameters.jl   draw_volatility_mean / draw_volatility_ar /
                             volatility_covariance_posterior
common/sv/sv_priors.jl       sv_priors(n; …)   ← replaces the un-included src/TCVAR/SV_priors.jl
```

Signatures, matching the spec's "take residuals and parameters — mean, AR matrix,
volatility covariance":

```julia
"""
    draw_stochastic_volatility(residuals, h, params; h0_covariance, offset = 1e-3)

Multivariate SV draw with the KSC mixture approximation.

  residuals : T × n orthogonalised residuals (e_t = A₀ ε_t)
  h         : (T+1) × n current log-volatility path, rows t = 0 … T
  params    : NamedTuple (μ, Φ, Ω)

Returns (h_new, s) — the drawn path (T+1 rows, including h_0) and the T × n mixture
indicators. Calls draw_mixture_indicators then draw_log_volatilities.
"""

draw_mixture_indicators(y_star, h)               -> Matrix{Int}      # T × n
draw_log_volatilities(y_star, s, params; …)      -> Matrix{Float64}  # (T+1) × n, via carter_kohn_sampler

draw_volatility_mean(h, Φ, Ω, prior)             -> Vector
draw_volatility_ar(h, μ, Ω, prior; structure = :diagonal, max_draws = 100) -> Matrix
volatility_covariance_posterior(h, μ, Φ, scale_prior, df_posterior) -> InverseWishart
```

`sv_priors(n; …)` replaces the current `SV_priors.jl` (which hard-codes `const N = 6` and
is not included by the module at all) and returns a NamedTuple in the same
distribution-keyed style as `tcvar_priors.jl`:

```julia
(volatility_mean       = MvNormal(fill(log(0.1^2), n), …),   # unconditional, no ρ/σ² closure
 volatility_ar         = MvNormal(fill(0.8, n), 0.2^2 I),    # diagonal Φ
 volatility_covariance = InverseWishart(n + 11, 0.04*(n+10)*I),
 simultaneity          = MvNormal(zeros(n*(n-1)÷2), 10.0 I))
```

The current `μ` prior is written as a closure `(ρ, σ²) -> Normal(...)` over the ergodic
variance. That makes it a *conditional* prior that changes with the draws, which breaks
the conjugate update in step 6. Replacing it with a fixed, generously wide `MvNormal`
centred at the same `log(0.1²)` is the change (**D3**).

### 3.5 `var/` — two additions

```
var/simultaneity.jl    draw_simultaneity(residuals, h, prior)  →  unit-lower-triangular A₀
                       simultaneity_covariances(A₀, h)         →  T-vector of Σ_t (and Σ_t⁻¹)
var/var_sampling.jl    + draw_var_coefficients_gls(cycle, p, Σ_inv_series, β₀, P₀; max_draws)
```

`draw_simultaneity` is VAR-layer rather than `common/` because it is the structural
factorisation of a VAR innovation covariance; `simultaneity_covariances` returns both
`Σ_t` and `Σ_t⁻¹ = A₀ᵀ H_t⁻¹ A₀` (the inverse is available in closed form, so nothing is
ever factorised numerically).

`sample_var_params` (the NIW/homoskedastic draw) is left exactly as it is — TCVAR keeps
using it.

### 3.6 What is reused with **no change at all**

`random_walk_covariance_posterior`, `inverse_wishart_posterior`,
`normal_coefficient_posterior_mean`, `kron_cholesky_factor`, `draw_from_factor`,
`chol_psd`, `sample_mvn`, `is_stationary`, `prepare_var_data`, `MinnesotaPrior` +
`prior_var_coeff` + `prior_row_covariance`, `compute_posterior_statistics`.

---

## 4. `models/tcvar_sv/`

```
tcvar_sv_model.jl    TCVARSV struct (ssm skeleton with 3-D Q, priors, names, ar_structure);
                     tc_var_sv(trend_mapping, T; p) skeleton builder;
                     update_tc_var_sv!(ssm, var_coeff, Στ, Σ_series, …)  — in place, no
                     reallocation of the T × n_states × n_states array between draws
tcvar_sv_priors.jl   tcvar_sv_priors(...) — the five TCVAR keys plus the four SV keys;
                     constructor validation mirroring TCVAR's
tcvar_sv_gibbs.jl    gibbs_sampler(model::TCVARSV, data; burnin, n_samples, thin, logging)
tcvar_sv_result.jl   TCVarSVResult, build_result, posterior_mean, simulate_scenarios
```

`sample_states` is **not** duplicated: generalising its signature to
`AbstractStateSpaceModel` (§3.1) is enough for TCVAR-SV to call the existing one.

`TCVarSVResult` follows `TCVarResult` exactly — one `FlexiChain{VarName}` of matrix-valued
parameters plus state arrays:

```julia
struct TCVarSVResult
    model::TCVARSV
    params::FlexiChain{VarName}          # Στ, β, A₀, μ, Φ, Ω  (matrix-valued)
    trend_states  ::Array{Float64,3}     # n_kept × (T+1) × n_trends
    cycle_states  ::Array{Float64,3}     # n_kept × (T+p) × n_obs
    volatilities  ::Array{Float64,3}     # n_kept × (T+1) × n_obs   (h, log scale)
end
```

`h` goes in the state arrays rather than the chain, for the same reason the trend and
cycle paths do: it is a `T`-length path, not a fixed-size parameter block, and
`compute_posterior_statistics` / `plot_states` already work on that shape.

Initialisation of the sweep: `h[1,:,:]` at `log` of the OLS residual variances of a
homoskedastic pilot VAR on the initial cycle path, `A₀ = I`, `μ` at its prior mean,
`Φ = 0.95 I`, `Ω`, `Στ` at their prior means, `β` at the same identity-on-last-lag start
TCVAR uses.

Reporting: `plot_volatilities(result)` in `reporting/plots.jl`, plotting `exp(h/2)`
posterior mean with a credible band per series (`compute_posterior_statistics` verbatim).

---

## 5. Scripts

New folder `analisys/simulated-data/tcvar_sv/`, each file runnable as
`julia --project analisys/simulated-data/tcvar_sv/<name>.jl`, saving figures to
`analisys/simulated-data/tcvar_sv/output/` and printing a comparison table to stdout.
(The rest of `analisys/` holds notebooks, which are not reproducible from the CLI.)

### 5.1 `sv_block_recovery.jl` — the SV block alone, `h` only

Isolates step 5. **Only the state `h_{0:T}` is inferred**; every volatility parameter is
held at its true value, and the volatilities are simulated from the *same* stationary
AR(1)-with-mean law the target model assumes rather than from a random walk, so the block
is exercised against the process it will actually meet in the sweep (**D7**).

1. Simulate `n = 3, T = 500` at known `(μ, Φ, Ω)`, in the §1 form:

   ```
   h_t = μ + Φ (h_{t-1} - μ) + ν_t,   ν_t ~ N(0, Ω_true),   h_0 ~ N(μ, P₀)
   e_t ~ N(0, H_t),                   H_t = diag(exp(h_t))
   ```

   with `μ = log([0.5, 1.0, 2.0].^2)` (so the series sit at genuinely different volatility
   levels), `Φ = 0.95·I` (diagonal, the default structure), a deliberately **correlated**
   `Ω_true` (e.g. `0.02·[1 .5 .2; .5 1 .3; .2 .3 1]`), and `P₀` the stationary covariance
   `vec(P₀) = (I - Φ⊗Φ)⁻¹ vec(Ω_true)`.
2. Run **step 5 only** — `draw_mixture_indicators` + `draw_log_volatilities` — with
   `μ, Φ, Ω` fixed at the truth. `Ω` is *not* drawn here; recovering it is 5.2's job. With
   `Φ` stationary and `μ` known, the level of `h` is anchored by the model itself, so no
   diffuse-`h_0` workaround is needed and any level error is a real defect of the block.
3. Plot per series: true `exp(h_t/2)` vs posterior mean and 90% band; plus the realised
   `|e_t|` as a scatter for context. One figure, `n` stacked panels.
4. Assert-style printout: RMSE of `exp(ĥ/2)` against truth, share of periods where the
   truth lies inside the 90% band (target ≈ 0.9), and the mean level `mean(ĥ_i)` vs `μ_i`
   per series (the sharpest check that the `-1.2704` shift and the de-meaning are right).

This script is the acceptance test for the block: it isolates `draw_mixture_indicators` +
`draw_log_volatilities` + Carter–Kohn from every other moving part, including the
volatility-parameter draws.

### 5.2 `sv_posterior_checks.jl` — one block at a time, everything else at the truth

Matches "draw each new posterior with example params and compare posterior mean and median
with example params". For each new step, simulate data at known parameters, draw `N = 5000`
times **conditioning on the true values of every other block**, and report
`true / mean / median / 5% / 95%` per element:

| block | data simulated at | drawn by |
|---|---|---|
| `β` (GLS) | known `B, A₀, h` | `draw_var_coefficients_gls` |
| `A₀` | known `ε, h` | `draw_simultaneity` |
| `μ` | known `h, Φ, Ω` | `draw_volatility_mean` |
| `Φ` | known `h, μ, Ω` | `draw_volatility_ar` (both `:diagonal` and `:full`) |
| `Ω` | known `h, μ, Φ` | `volatility_covariance_posterior` |

Conditioning on the truth turns each into a single-block sampling problem whose posterior
concentrates around the truth as `T` grows, so a bias in any one assembly shows up here
and nowhere else. Also prints the sampling-error-scaled deviation
`(mean - truth) / posterior sd` so the numbers are readable without a tolerance table.

### 5.3 `tcvar_sv_recovery.jl` — full model on simulated data

Matches "inference full TCVAR with stochastic volatility with simulated data on model with
example parameters".

1. Simulate a `n = 3, n_trends = 3, p = 1, T = 400` TCVAR-SV forward from known
   `Στ, B, A₀, μ, Φ, Ω` (a `simulate_tcvar_sv` helper on the model, the SV analogue of
   `simulate_scenarios`).
2. Estimate with `gibbs_sampler(::TCVARSV, data; burnin = 5000, n_samples = 5000)`.
3. Output: `plot_states` (trends + cycles vs simulated truth), `plot_volatilities`
   (`exp(h/2)` vs truth), a parameter table (true / mean / median per block), and trace
   plots for the scalar blocks as a convergence check.

---

## 6. Tests

Mirroring `src/` as `file_structure_refactor_plan.md` intended, and extending
`runtests.jl`'s include list:

```
test/common/state_space_test.jl     process_noise/observation_noise dispatch;
                                    TimeVaryingStateSpaceModel with a constant Q_t
                                    reproduces StateSpaceModel bit-for-bit
test/common/posteriors_test.jl      (existing) + normal_posterior vs closed form;
                                    gls_information with all Σ_t = Σ equals the
                                    homoskedastic X'Σ⁻¹X; weighted_regression_information
                                    vs weighted least squares
test/common/sv/ksc_mixture_test.jl  Σq = 1; Σ q_j(m_j-1.2704) ≈ E[log χ²₁] = -1.2704;
                                    mixture variance ≈ π²/2; indicator draw is a valid
                                    categorical and concentrates on the right component
test/common/sv/sv_block_test.jl     h fixed at truth ⇒ indicator frequencies match the
                                    analytic weights; small-T posterior mean of h tracks
                                    the truth (seeded, loose tolerance)
test/common/sv/sv_parameters_test.jl  each of steps 6/7/8 against its closed form on
                                    synthetic h with known (μ, Φ, Ω)
test/var/simultaneity_test.jl       homoskedastic limit equals the OLS Cholesky factor;
                                    round-trip Σ_t = A₀⁻¹H_tA₀⁻ᵀ ⇒ A₀ recovered
test/models/tcvar_sv/tcvar_sv_recovery_test.jl
                                    the §5.3 recovery at a smaller T/n_samples, with
                                    point-estimate tolerances in the style of
                                    tcvar_recovery_test.jl (which documents that credible
                                    -interval containment is not a reliable assertion here)
```

Plus the **non-regression guard that matters most**: a seeded `gibbs_sampler(::TCVAR, …)`
run whose draws must be bit-identical before and after §3.1/§3.2, because those touch the
filter TCVAR itself uses.

---

## 7. Execution stages

Each stage is one commit and ends green on `julia --project test/runtests.jl`.

**Stage 1 — the time-varying seam.** §3.1 + §3.2. No new model code. Verify: existing
tests green *and* the seeded TCVAR draws are bit-identical to `main`.

**Stage 2 — generic posteriors.** §3.3 (`normal_posterior`, `gls_information`,
`weighted_regression_information`) with `test/common/posteriors_test.jl` extended first.
Pure additions, nothing repointed.

**Stage 3 — the SV block.** §3.4: `ksc_mixture.jl`, `sv_block.jl`, `sv_priors.jl`; delete
`src/TCVAR/SV_priors.jl`. Ships with `test/common/sv/…` and **script 5.1**, which is the
real acceptance test. This is the highest-risk stage — the mixture constants, the `-1.2704`
shift, the `log(e² + c̄)` offset and the demeaning all have to be right simultaneously, and
script 5.1 is what tells you they are.

**Stage 4 — the remaining new blocks.** §3.5 (`simultaneity.jl`,
`draw_var_coefficients_gls`) and `sv_parameters.jl`. Ships with its unit tests and
**script 5.2**, which checks every one of them one at a time.

**Stage 5 — the model.** `models/tcvar_sv/`: struct, skeleton, priors, result type,
`simulate_tcvar_sv`. No sweep yet; the simulator is testable on its own (simulate at
`Ω = 0` and check it reproduces a homoskedastic TCVAR path).

**Stage 6 — the sweep.** `tcvar_sv_gibbs.jl` wiring steps 1–8, plus `plot_volatilities`,
the exports in `TCVAR.jl`, **script 5.3** and the recovery test.

Ordering rationale: every stage before 6 is independently verifiable, and stages 3 and 4
each land with the script that proves them. Nothing about TCVAR changes after stage 1, so
a regression in the existing model can only come from one commit.

---

## 8. Decisions taken, and the ones worth confirming

- **D1 — "trend covariance … with gls".** Read as: no change (the trend block has no
  regressors, so GLS ≡ the existing conjugate IW draw), and GLS is applied where it
  actually bites — steps 3, 4, 6, 7. **Confirm if something else was meant.**
- **D2 — Minnesota prior under SV.** `Σ` is no longer a free parameter, so the
  `Φ|Σ ~ MN(Φ₀, Ω_M, Σ)` prior is fixed at `Σ̄ = mean(priors.cycle_covariance)`
  (Cogley–Sargent's choice). The alternative — rescaling by the current `Σ̄_t` average
  each sweep — makes the prior data-dependent; not taken.
- **D3 — `μ` prior.** The closure form in today's `SV_priors.jl` conditions on `(ρ, σ²)`
  and would break step 6's conjugacy; replaced by a fixed wide `MvNormal` at the same
  centre.
- **D4 — `Φ` default is diagonal**, matching `SV_priors.jl`'s scalar `ρ` and
  Clark–Ravazzolo; full-matrix supported via `structure = :full` and reuses the existing
  NIW helpers verbatim.
- **D5 — `A₀`, not `L`.** Drawn as unit-lower-triangular `A₀` with `Σ_t = A₀⁻¹H_tA₀⁻ᵀ`
  because that is the conjugate parameterisation; `L = A₀⁻¹` is exposed on the result for
  the spec's vocabulary. Ordering-dependent identification, documented on the struct.
- **D6 — 7-component KSC mixture.** Omori et al. (2007)'s 10-component table is a strictly
  better approximation and is a drop-in replacement for the constants in
  `ksc_mixture.jl`; not taken now, noted as a one-line upgrade path.
- **D7 — script 5.1 simulates stationary AR(1) volatilities, and infers `h` alone.**
  The spec words it as "simulated data of multivariate random walk model"; the block is
  instead driven by the `h_t = μ + Φ(h_{t-1}-μ) + ν_t` law of §1 with `(μ, Φ, Ω)` fixed
  at the truth. Two reasons: the acceptance test then exercises exactly the process the
  sweep will hand it (a unit root is a corner case of it, not the case), and with nothing
  but `h` drawn there is no second block whose error could offset a bug in the first. `Ω`
  recovery moves entirely to 5.2, where it is checked against a closed form.

## 9. Explicitly out of scope

- The Metropolis–Hastings scaffold (`common/metropolis_hastings.jl`) named in
  `temp_spec.md`. Every block above is conjugate — nothing in TCVAR-SV needs MH. It
  becomes necessary only for a genuinely-constant-correlation `Σ_t = D_t R D_t` model or a
  bounded-support prior on `Φ`.
- Time-varying `A₀` (Primiceri 2005). The state-space seam from §3.1 is what a later
  `A₀,t` block would build on, but it is a different model.
- Steady-state BVAR (`models/ss_bvar/`), the module rename, and explicit `rng` threading —
  all still deferred per `file_structure_refactor_plan.md`.

## References

- Cogley, T. & Sargent, T. (2005), *Drifts and Volatilities: Monetary Policies and Outcomes
  in the Post WWII U.S.*, Review of Economic Dynamics — constant `A₀`, SV innovations.
- Kim, S., Shephard, N. & Chib, S. (1998), *Stochastic Volatility: Likelihood Inference and
  Comparison with ARCH Models*, REStud — the 7-component mixture and the offset `c̄`.
- Primiceri, G. (2005), *Time Varying Structural Vector Autoregressions and Monetary
  Policy*, REStud — the row-by-row `A₀` draw.
- Clark, T. & Ravazzolo, F. (2015), *Macroeconomic Forecasting Performance under
  Alternative Specifications of Time-Varying Volatility*, JAE — the AR(1) SV priors.
- Omori, Y., Chib, S., Shephard, N. & Nakajima, J. (2007), *Stochastic Volatility with
  Leverage*, J. Econometrics — the 10-component mixture (D6).
