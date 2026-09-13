# Steady-state BVAR — Gibbs sampler implementation plan

Implementation plan for the steady-state Bayesian VAR of Villani (2009): a VAR written in
deviations from its unconditional mean, with an informative prior placed directly on that
mean. It is the no-trend benchmark of `project_plan.md` ("Steady state bvar - tradycyjny")
and the model `file_structure_refactor_plan.md` reserved `models/ss_bvar/` for.

The work is five tasks. Every subtask ends in a **human check** — a named checkpoint (H1, H2,
H3.1, …) where you run something, look at something, and approve before the next subtask
starts. Every Gibbs step goes through the same four subtasks: *interface → implementation →
unit tests → integration test*.

**Guiding rule (unchanged from `tcvar_sv_plan.md`):** whatever a second model could reuse goes
into `common/` or `var/`; `models/ss_bvar/` holds only the priors, the struct, the sweep and the
result. TCVAR and TCVAR-SV come out unchanged — `test/models/tcvar/tcvar_gibbs_regression_test.jl`
guards that for free.

---

## 0. Checkpoint map

| Checkpoint | Subtask | Test suite after it | Commit |
|---|---|---|---|
| H1 | Task 1 — model, priors, likelihood, simulator, simulation script | green | yes |
| H2 | Task 2 — Gibbs sampler returning the initial values | green | yes |
| H3.1 | Step 1 ($\beta$) — interface + call in the sampler | red: the stub throws (expected) | no |
| H3.2 | Step 1 — implementation (+ `normal_posterior`) | red: "$\beta$ stays at its initial value" fails (expected — proves the step is wired) | no |
| H3.3 | Step 1 — unit tests | as H3.2 | no |
| H3.4 | Step 1 — integration test updated | green | yes |
| H4.1–H4.4 | Step 2 ($\Sigma$) — the same four subtasks | same pattern | at H4.4 |
| H5.1–H5.4 | Step 3 ($\mu$) — the same four subtasks, ending in full recovery | same pattern | at H5.4 |

One commit per task, after its last check, so every commit is green.
`julia --project test/runtests.jl` runs everything; every new test file carries the usual
`isdefined(Main, :TCVAR) || include(…)` guard and reaches members as `TCVAR.f`, so it also runs
alone, e.g. `julia --project test/var/steady_state_posteriors_test.jl`.

A failed human check is fixed inside its subtask; the next subtask does not start on a red check.

---

## 1. The model

### 1.1 Equations

For $y_t \in \mathbb{R}^n$, $t = 1, \dots, T$, a VAR($p$) in deviations from the steady state $\mu$:

$$
y_t - \mu = A_1 (y_{t-1} - \mu) + \dots + A_p (y_{t-p} - \mu) + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \Sigma)
$$

This is Villani's $\Pi(L)(y_t - \Psi d_t) = \varepsilon_t$ with the deterministic term $d_t = 1$, so $\Psi = \mu$
(**D3**).

### 1.2 Layout — the codebase's conventions, not new ones

- $k = n \cdot p$; regressor row $x_t = [y_{t-p}; \dots; y_{t-1}]$, **oldest-lag-first** (the
  `prepare_var_data` layout).
- $\beta$ is $k \times n$; $A = \beta^\top = [A_p \ \cdots \ A_1]$ ($n \times k$) is the companion bottom block, exactly as in
  `update_tc_var!` and `is_stationary`.
- Demeaned regression form, which every posterior below uses:

  $$
  \tilde{y}_t = \beta^\top \tilde{x}_t + \varepsilon_t, \qquad \tilde{y}_t = y_t - \mu, \qquad \tilde{x}_t = x_t - (\mathbf{1}_p \otimes \mu)
  $$

  ```julia
  Ỹ, X̃ = prepare_var_data(data .- μ', p)        # (T−p) × n  and  (T−p) × k
  ```

- $T_{\text{eff}} = T - p$: the likelihood conditions on the first $p$ observations (**D4**).
- $\operatorname{vec}(\beta)$ is column-major, so it stacks **equation blocks**; every Kronecker product below is
  written for that ordering — the one `kron_cholesky_factor(Σ, V)` already uses.

The example truth is written as `β = collect(hcat(A_2, A_1)')` for $p = 2$. That line is the
easiest place to introduce a layout bug, so the script and the recovery test both spell it out.

### 1.3 Priors — independent blocks (D1, D2)

$$
\begin{aligned}
\mu &\sim \mathcal{N}(\mu_0, V_\mu) && \text{steady-state prior — the informative one, the point of the model} \\
\operatorname{vec}(\beta) &\sim \mathcal{N}(\operatorname{vec}(\beta_0), V_\beta) && \text{Minnesota, } V_\beta = \bar{\Sigma} \otimes \Omega_M \\
\Sigma &\sim \mathcal{IW}(d, S)
\end{aligned}
$$

$\beta_0$ = `prior_var_coeff(β_prior)ᵀ` (oldest-lag-first), $\Omega_M$ = `prior_row_covariance(β_prior)`,
$\bar{\Sigma}$ = `mean(Σ_prior)` — all existing code; it is the same "Minnesota at the prior-mean $\Sigma$" choice
as D2 of `tcvar_sv_plan.md`. There are no latent states, so no Kalman filter anywhere.

### 1.4 Likelihood

$$
\log L(Y \mid \beta, \Sigma, \mu) = \sum_{t=p+1}^{T} \log \mathcal{N}\Big( y_t ;\ \mu + \sum_{l=1}^{p} A_l (y_{t-l} - \mu),\ \Sigma \Big)
$$

Implemented **naively**: a loop over $t$, `logpdf(MvNormal(…), y_t)`, $A_l$ sliced out of $\beta^\top$.
It is the oracle for every unit test of §4, so it must not share the posterior algebra — no
`prepare_var_data`, no Kronecker products.

---

## 2. The Gibbs sweep — three conditional posteriors

Order $\beta \to \Sigma \to \mu$, the order the tasks implement them (any fixed order leaves the stationary
distribution unchanged). Each step conditions on the latest draws of the other two.

### Step 1 — $\beta \mid \Sigma, \mu, Y$

$$
\begin{aligned}
\operatorname{vec}(\tilde{Y}) &= (I_n \otimes \tilde{X}) \operatorname{vec}(\beta) + \operatorname{vec}(E), \qquad \operatorname{vec}(E) \sim \mathcal{N}(0,\ \Sigma \otimes I) \\[6pt]
P_d &= \Sigma^{-1} \otimes \tilde{X}^\top \tilde{X} \in \mathbb{R}^{nk \times nk} \\
b_d &= \operatorname{vec}(\tilde{X}^\top \tilde{Y} \Sigma^{-1}) \in \mathbb{R}^{nk} \\[6pt]
\operatorname{vec}(\beta) \mid \cdot &\sim \mathcal{N}\big( (V_\beta^{-1} + P_d)^{-1} (V_\beta^{-1} \operatorname{vec}(\beta_0) + b_d),\ (V_\beta^{-1} + P_d)^{-1} \big)
\end{aligned}
$$

The draw is truncated to the stationary region by rejection — the same `is_stationary` /
`max_draws` loop as `sample_var_params` (**D6**). Stationarity also guarantees $M$ of step 3 is
invertible.

### Step 2 — $\Sigma \mid \beta, \mu, Y$

$$
\begin{aligned}
E &= \tilde{Y} - \tilde{X} \beta \\
\Sigma \mid \cdot &\sim \mathcal{IW}\big( T_{\text{eff}} + d,\ E^\top E + S \big)
\end{aligned}
$$

`inverse_wishart_posterior(E, S, T_eff + d)` from `common/posteriors.jl`, verbatim.

### Step 3 — $\mu \mid \beta, \Sigma, Y$

Substituting $\tilde{x}_t = x_t - \mathbf{1}_p \otimes \mu$ turns the model into a regression of the raw residual on a
constant design:

$$
\begin{aligned}
w_t &= y_t - \beta^\top x_t, \qquad W = Y - X \beta \\
M &= I_n - (A_1 + \dots + A_p) \qquad \text{the long-run matrix, Villani's } \Pi(1) \\[6pt]
w_t &= M \mu + \varepsilon_t \\[6pt]
P_d &= T_{\text{eff}} \cdot M^\top \Sigma^{-1} M \\
b_d &= M^\top \Sigma^{-1} \sum_t w_t \\[6pt]
\mu \mid \cdot &\sim \mathcal{N}\big( (V_\mu^{-1} + P_d)^{-1} (V_\mu^{-1} \mu_0 + b_d),\ (V_\mu^{-1} + P_d)^{-1} \big)
\end{aligned}
$$

where $Y, X$ = `prepare_var_data(data, p)` on the raw data (**NOT** demeaned).

Steps 1 and 3 are the same conjugate normal update $\mathcal{N}\big((P_0 + P_d)^{-1}(P_0 m_0 + b_d),\ (P_0 + P_d)^{-1}\big)$.
It lands once, as `normal_posterior` in `common/posteriors.jl` — the primitive
`tcvar_sv_plan.md` §3.3 specified and that was never built, with that plan's signature.

### The final sweep

```julia
for s in 2:n_draws
    β = draw_ss_var_coefficients(data, p, Σ, μ, β_prior)             # step 1 — task 3
    Σ = rand(ss_var_covariance_posterior(data, p, β, μ, Σ_prior))     # step 2 — task 4
    μ = rand(steady_state_posterior(data, p, β, Σ, μ_prior))          # step 3 — task 5
    betas[s, :] = vec(β);  sigmas[s, :, :] = Σ;  mus[s, :] = μ
end

#second version 

for s in 2:n_draws
    c = Y .- u
    β = rand(independent_normal_posterior(y,X, Σ, β_prior))         # step 1 — task 3
    res = residuals(T,X,β)
    Σ = rand(independent_inverse_wishart_posterior(res, Σ_prior))         # step 2 — task 4
    μ = rand(steady_state_posterior(data, p, β, Σ, μ_prior))          # step 3 — task 5
    betas[s, :] = vec(β);  sigmas[s, :, :] = Σ;  mus[s, :] = μ
end
```

---

## 3. Files

```
src/TCVAR/
├── common/posteriors.jl               + normal_posterior                                  task 3
├── var/
│   ├── steady_state_var.jl            NEW  long_run_matrix, ss_var_loglikelihood            task 1
│   └── steady_state_posteriors.jl     NEW  ss_var_coefficients_posterior,                  tasks 3–5
│                                           draw_ss_var_coefficients,
│                                           ss_var_covariance_posterior,
│                                           steady_state_posterior
└── models/ss_bvar/
    ├── ss_bvar_priors.jl              NEW  ss_bvar_priors                                   task 1
    ├── ss_bvar_model.jl               NEW  SSBVAR struct + constructor                      task 1
    ├── ss_bvar_result.jl              NEW  simulate_scenarios(::SSBVAR, …)                  task 1
    │                                       SSBVarResult, build_result, posterior_mean       task 2
    └── ss_bvar_gibbs.jl               NEW  ss_bvar_initial_values, gibbs_sampler(::SSBVAR)  task 2

analisys/simulated-data/ss_bvar/ss_bvar_simulation.jl    NEW                                 task 1

test/
├── tcvar_test_utils.jl                + ss_bvar_test_priors                                 task 1
├── tcvar_posteriors_test.jl           + normal_posterior testset                            task 3
├── var/steady_state_var_test.jl       NEW                                                   task 1
├── var/steady_state_posteriors_test.jl  NEW — one testset per step                         tasks 3–5
└── models/ss_bvar/
    ├── ss_bvar_model_test.jl          NEW                                                   task 1
    └── ss_bvar_recovery_test.jl       NEW in task 2, extended by tasks 3–5
```

`TCVAR.jl`: the two `var/` files after `var_sampling.jl`; `models/ss_bvar/` after
`models/tcvar_sv/`, ordered priors → model → result → gibbs (`SSBVarResult` names `SSBVAR`).
New exports: `SSBVAR`, `ss_bvar_priors`, `SSBVarResult`; `gibbs_sampler`, `posterior_mean` and
`simulate_scenarios` are already exported and just gain methods. `runtests.jl` gets the four new
test files.

---

## 4. The two unit-test checks every step uses

### 4.1 Shape of the posterior

| step | distribution | `mean` | `cov` | extra |
|---|---|---|---|---|
| 1 $\beta$ | `MvNormal` | $(n^2 p,)$ | $(n^2 p, n^2 p)$, `isposdef` | — |
| 2 $\Sigma$ | `InverseWishart` | $(n, n)$ | $(n^2, n^2)$ — Distributions.jl returns the covariance of $\operatorname{vec}(\Sigma)$ | `params(post)[1]` $= T - p + d$ |
| 3 $\mu$ | `MvNormal` | $(n,)$ | $(n, n)$, `isposdef` | — |

Each for $(n, p) \in \lbrace (1, 1), (2, 2), (3, 1) \rbrace$, so $n = 1$ and $p > 1$ are both covered.

### 4.2 The log-density ratio identity

A ratio of densities is a difference of log densities. For **any** data, **any** fixed values of
the other two blocks and **any** two values $\theta_1, \theta_2$ of the drawn block:

$$
\begin{aligned}
\operatorname{logpdf}(\text{post}, \theta_1) - \operatorname{logpdf}(\text{post}, \theta_2)
  = {} & \big[ \operatorname{logpdf}(\text{prior}, \theta_1) + \log L(Y \mid \theta_1, \text{rest}) \big] \\
       & {} - \big[ \operatorname{logpdf}(\text{prior}, \theta_2) + \log L(Y \mid \theta_2, \text{rest}) \big]
\end{aligned}
$$

because $p(\theta \mid \text{rest}, Y) = p(\theta) \cdot L(Y \mid \theta, \text{rest}) / p(Y \mid \text{rest})$ and the evidence $p(Y \mid \text{rest})$ does
not depend on $\theta$ — it cancels. (The priors are independent across blocks, so $p(\theta \mid \text{rest}) = p(\theta)$.)

**Why it is a strong test.** The right-hand side uses only the prior distribution object and the
naive likelihood oracle of §1.4 — none of the conjugate algebra — so it pins the posterior
**kernel** independently. A wrong mean, precision, degrees of freedom, scale, Kronecker order or
transpose breaks the identity at almost every random pair. What it does not see is the
normalising constant, and that Distributions.jl computes itself.

Recipe, identical for every step (shown for step 1):

```julia
Random.seed!(…)
data = randn(T, n) .+ randn(n)'           # any data — the identity is algebra, not inference
Σ    = rand(InverseWishart(n + 3, Matrix(1.0I, n, n)))
μ    = randn(n)
prior = ss_bvar_test_priors(; n = n, p = p)

post = TCVAR.ss_var_coefficients_posterior(data, p, Σ, μ, prior.var_β)
log_joint(b) = logpdf(prior.var_β, vec(b)) + TCVAR.ss_var_loglikelihood(data, p, b, Σ, μ)

worst = 0.0
for _ in 1:50
    b₁, b₂ = 0.3 * randn(n * p, n), 0.3 * randn(n * p, n)
    lhs = logpdf(post, vec(b₁)) - logpdf(post, vec(b₂))
    rhs = log_joint(b₁) - log_joint(b₂)
    @test lhs ≈ rhs rtol = 1e-8 atol = 1e-8
    worst = max(worst, abs(lhs - rhs))
end
println("  step 1, (n, p) = ($n, $p): max |lhs − rhs| = ", worst)   # read at the human check
```

Random pairs: `0.3·randn(k, n)` for $\beta$, `rand(InverseWishart(n + 3, I))` for $\Sigma$, `randn(n)` for
$\mu$. The $\beta$ pairs need not be stationary: the identity is about the untruncated posterior
returned by the pure `*_posterior` function, and truncation in `draw_*` only divides by
$P(\text{stationary})$, which cancels too.

---

## 5. Example parameters — shared by the script and the recovery test

```julia
N_SERIES, N_LAGS, N_TIME = 3, 2, 500

MU_TRUE = [2.0, 3.0, 5.0]

A1_TRUE = [ 0.50  0.00  0.00
            0.20  0.40  0.00
           -0.10  0.15  0.60]
A2_TRUE = [ 0.20  0.00  0.00
            0.00  0.15  0.00
            0.05  0.00 -0.20]
Β_TRUE  = collect(hcat(A2_TRUE, A1_TRUE)')          # k × n, oldest-lag-first

Σ_TRUE  = [1.0 0.3 0.1
           0.3 0.5 0.1
           0.1 0.1 0.8]
```

Why these values:

- **Stationarity is known analytically.** Both lag matrices are lower triangular, so the
  companion roots are those of $\lambda^2 - a_{1,ii} \lambda - a_{2,ii} = 0$ per series; the largest modulus is
  **0.762** (confirmed with `TCVAR.is_stationary` on the current `main`).
- **Layout errors fail loudly.** Off-diagonal coefficients sit only below the diagonal, so a
  transposed $A_l$ moves them above it; the two lag blocks differ, so a swapped lag order shows;
  $(A_1)_{31}$ and $(A_2)_{33}$ are negative, so a dropped sign shows.
- **$\mu$ is moderately identified.** $M = I - A_1 - A_2$ has $M_{11} = 0.3$, so the posterior sd of
  $\mu_1$ is about $\sqrt{\Sigma_{11} / (T \cdot 0.3^2)} \approx 0.15$ — enough for the step to matter, tight enough to
  test.

**Estimation priors — deliberately off the truth**, as in `tcvar_recovery_test.jl`, so the tests
measure learning rather than prior echo:

```julia
ss_bvar_priors(MU_TRUE .+ [1.0, -1.0, 1.0],     # steady-state prior mean: 1 sd off the truth
               fill(1.0, 3),                    # steady-state prior sd
               N_LAGS,
               2 .* diag(Σ_TRUE);               # ψ: IW scale at twice the true variances
               λ = 0.5, δ = zeros(3))           # d = n + 2 default ⇒ mean(Σ_prior) = diagm(ψ)
```

---

## Task 1 — the model and the simulation script

**1a. Priors** — `models/ss_bvar/ss_bvar_priors.jl`

```julia
ss_bvar_priors(steady_state_mean, steady_state_sd, p, ψ; λ = 0.2, δ = zeros(n), d = n + 2)
    -> (steady_state = MvNormal, var_β = MvNormal, var_covariance = InverseWishart)
```

Builds `Σ_prior = InverseWishart(d, Diagonal(ψ))` and `β_prior = MinnesotaPrior(λ, p, Σ_prior; δ)`,
then turns `β_prior` into the independent `MvNormal` on $\operatorname{vec}(\beta)$ of §1.3. It does not go through
`var_priors`, which would also build an initial-cycle prior this model has no use for.
Validation: lengths agree, `all(δ .< 1)` (a stationary model has no random-walk prior mean). The
docstring notes Villani's convention of stating the steady-state prior as a 95% interval:
$\text{sd} = (\text{upper} - \text{lower}) / (2 \cdot 1.96)$.

**1b. Model** — `models/ss_bvar/ss_bvar_model.jl`

```julia
struct SSBVAR
    priors::NamedTuple               # the three keys of ss_bvar_priors
    p::Int
    variable_names::Vector{String}
end
SSBVAR(priors::NamedTuple, p; variable_names = default_variable_names(n))
```

$n$ = `length(priors.steady_state)`. The constructor throws `ArgumentError` on a missing key and
`DimensionMismatch` on any shape mismatch (`length(var_β)` $= n^2 p$, `size(var_covariance)` $= (n, n)$,
names), following `TCVAR(trend_mapping, priors)`. No state-space field — there are no latent states.

**1c. Likelihood oracle** — `var/steady_state_var.jl`

```julia
long_run_matrix(β, n, p) -> Matrix                     # M = I − ∑_l A_l, blocks sliced from βᵀ
ss_var_loglikelihood(data, p, β, Σ, μ) -> Float64      # §1.4, deliberately naive
```

**1d. Simulator** — `models/ss_bvar/ss_bvar_result.jl`

```julia
simulate_scenarios(model::SSBVAR, params::NamedTuple, initial_lags::AbstractMatrix,
                   n_scenarios, n_steps) -> Array{Float64,3}      # n_scenarios × n_steps × n
```

`params = (β, Σ, μ)`, `initial_lags` is $p \times n$, oldest row first. **No new companion code:** an
SS-BVAR is the cycle of a TCVAR with zero trends, shifted by $\mu$:

```julia
ssm = tc_var(zeros(n, 0); p = p)
update_tc_var!(ssm, collect(params.β'), zeros(0, 0), params.Σ, 0, n, p)
ξ₀  = vec(permutedims(initial_lags .- params.μ'))        # oldest-lag-first
_, obs = sample(ssm, ξ₀, n_steps; jitter = 0)
obs .+ params.μ'
```

Verified on the current `main`: `tc_var(zeros(3, 0); p = 2)` builds a 6-state skeleton, the
observation matches the last state block to within 6e-8 (the skeleton's `H = eps()·I`), and a
20 000-step path at the example parameters has sample mean `[1.97, 2.98, 5.00]`. As with every
`sample`-based simulator, row 1 is the starting point (the last initial lag) and the path
carries `n_steps − 1` transitions — documented on the function.

**1e. Script** — `analisys/simulated-data/ss_bvar/ss_bvar_simulation.jl`, in the style of
`tcvar_sv_recovery.jl`; run as `julia --project analisys/simulated-data/ss_bvar/ss_bvar_simulation.jl`.

1. The §5 constants; assert `is_stationary`.
2. Stationary start: $\xi_0 \sim \mathcal{N}(\mathbf{1}_p \otimes \mu,\ P)$ with $P$ = `lyapunov_covariance(F, Q)` and `psd_factor`,
   so the path begins mid-series rather than in a transient.
3. Simulate one $T = 500$ path.
4. Print, per series: $\mu$ vs sample mean with its standard error
   $\sqrt{\operatorname{diag}(M^{-1} \Sigma M^{-\top}) / T}$; the unconditional sd (Lyapunov) vs the sample sd.
5. Print the OLS fit on the simulated data: $\max_{ij} \lvert \hat{\beta}_{ij} - \beta_{ij} \rvert$, and the scaled deviation of
   the residual covariance from $\Sigma$.
6. Print `ss_var_loglikelihood` at the truth and at the OLS estimate. The gap
   $\ell(\text{OLS}) - \ell(\text{truth})$ should be positive and of the order of half the parameter count
   (≈ 13; twice the gap is roughly $\chi^2$ with 27 degrees of freedom) — a sanity check on the oracle.
7. Figure `output/ss_bvar_simulated_sample.png`: one panel per series — the path, a dashed $\mu$
   line and a $\mu \pm 2 \cdot \text{unconditional sd}$ band.

**1f. Tests**

- `test/var/steady_state_var_test.jl` — `long_run_matrix` against hand-computed $p = 1$ and
  $p = 2$ cases; `ss_var_loglikelihood` against the closed form
  $-\tfrac{T_{\text{eff}} n}{2} \log 2\pi - \tfrac{T_{\text{eff}}}{2} \log\det\Sigma - \tfrac{1}{2} \operatorname{tr}(\Sigma^{-1} E^\top E)$ with $E$ built through
  `prepare_var_data` (which also cross-checks the oracle's lag slicing against that layout).
- `test/models/ss_bvar/ss_bvar_model_test.jl` — priors: keys, shapes, the `var_β` mean carries
  $\delta$ on the own **first** lag (the **last** block, oldest-lag-first), the `var_β` covariance is
  $\bar{\Sigma} \otimes \Omega_M$, $\delta = 1$ throws; constructor errors; simulator: shapes, $\Sigma = 0$ started at $\mu$
  stays at $\mu$, started off $\mu$ converges to $\mu$.
- `test/tcvar_test_utils.jl` — `ss_bvar_test_priors(; n, p, …)`.

**Human check H1**
- Run the script. Figure: every series fluctuates around its own dashed $\mu$ with no drift,
  roughly 95% of points inside the band.
- Table: sample means within ~2 standard errors of $\mu$; OLS $\hat{\beta}$ within ~0.1 of the truth; the
  log-likelihood gap positive and of the stated order.
- Read the `SSBVAR` and `simulate_scenarios` docstrings: is the oldest-lag-first layout stated
  where a caller will look for it?
- Approve → commit *"SS-BVAR: model, priors, likelihood, simulator"*.

---

## Task 2 — a Gibbs sampler that returns the initial values

**2a. Initial values** — `models/ss_bvar/ss_bvar_gibbs.jl`

```julia
ss_bvar_initial_values(model::SSBVAR, data) -> (β = k × n, Σ = n × n, μ = n)
```

$\mu^0$ = column means; $\beta^0$ = OLS on the demeaned data, falling back to the prior mean if that is
not stationary; $\Sigma^0 = E^\top E / T_{\text{eff}}$. Data-driven by design (**D5**).

**2b. Result** — `models/ss_bvar/ss_bvar_result.jl`

```julia
struct SSBVarResult
    model::SSBVAR
    params::FlexiChain{VarName}      # β (k, n), Σ (n, n), μ (n,)
end
build_result(model::SSBVAR, betas, sigmas, mus, burnin, thin)
posterior_mean(result::SSBVarResult) -> (β, Σ, μ)
```

Follows `TCVarSVResult`, which already stores a vector-shaped `μ => (n_obs,)` parameter.

**2c. Sampler**

```julia
gibbs_sampler(model::SSBVAR, data; burnin = 1000, n_samples = 1000, thin = 1,
              initial_values = ss_bvar_initial_values(model, data), logging = false)
```

Validates `size(data, 2) == n`, rejects `missing` with an `ArgumentError` (**D4**), converts to
`Matrix{Float64}` once, unpacks the priors into locals before the loop, allocates
`betas` ($n_{\text{draws}} \times n \cdot k$), `sigmas` ($n_{\text{draws}} \times n \times n$), `mus` ($n_{\text{draws}} \times n$). The loop body only
stores the current $(\beta, \Sigma, \mu)$; the three step lines of §2 sit there as comments naming the task
that fills them.

**2d. Integration test, first version** — `test/models/ss_bvar/ss_bvar_recovery_test.jl`

- Simulate the §5 example (fixed seed), §5 priors, `burnin = 1000`, `n_samples = 2000`.
- Result shapes; number of kept draws `== length(burnin+1:thin:n_draws)`, also for `thin = 2`.
- Initial values vs truth: $\mu^0$ within 0.45, $\beta^0$ within 0.15, $\Sigma^0$ diagonal within 20%.
- **Blocks not drawn yet:** *every* draw of $\beta$, $\Sigma$ and $\mu$ equals its initial value, exactly.
  These assertions are removed one at a time as the steps land — they document which blocks are
  live, and they catch a step that was implemented but never wired into the sweep.
- `initial_values = truth` returns the truth in every draw.
- Print a truth / initial-value table.

**Human check H2**
- Run the recovery test; read the table — are the initial values sensible starting points?
- Read the sampler skeleton: storage layout, prior unpacking, the three commented step slots
  in $\beta \to \Sigma \to \mu$ order.
- Approve → commit *"SS-BVAR: Gibbs sampler skeleton returning initial values"*.

---

## Tasks 3–5 — one Gibbs step each

The same four subtasks for every step; what differs is in this table.

| | Task 3 — step 1 $\beta$ | Task 4 — step 2 $\Sigma$ | Task 5 — step 3 $\mu$ |
|---|---|---|---|
| posterior (pure) | `ss_var_coefficients_posterior(data, p, Σ, μ, prior::MvNormal) -> MvNormal` | `ss_var_covariance_posterior(data, p, β, μ, prior::InverseWishart) -> InverseWishart` | `steady_state_posterior(data, p, β, Σ, prior::MvNormal) -> MvNormal` |
| draw in the sweep | `draw_ss_var_coefficients(data, p, Σ, μ, prior; max_draws = 100) -> k × n`, rejection on `is_stationary` | `rand(post)` | `rand(post)` |
| math | §2 step 1 | §2 step 2 | §2 step 3 |
| reuses | `prepare_var_data`, `is_stationary`, new `normal_posterior` | `prepare_var_data`, `inverse_wishart_posterior` verbatim | `prepare_var_data`, `long_run_matrix`, `normal_posterior` |
| random pairs (§4.2) | `0.3·randn(k, n)` | `rand(InverseWishart(n + 3, I))` | `randn(n)` |
| recovery tolerance | $\max_{ij} \lvert \bar{\beta}_{ij} - \beta_{ij} \rvert \le 0.15$ (posterior sd ≈ 0.04–0.06) | diagonal relative ≤ 0.2, off-diagonal absolute ≤ 0.15 (sd of $\Sigma_{11}$ ≈ 6%) | $\max_i \lvert \bar{\mu}_i - \mu_i \rvert \le 0.45$ (≈ 3 sd for $\mu_1$, and below the prior offset of 1.0, so prior echo fails) |

### X.1 — Interface and call in the sampler (no implementation)

- Add the function(s) to `var/steady_state_posteriors.jl` with the final signature, a docstring
  carrying the posterior formula, argument shapes and layout, and the body
  `error("<name>: not implemented (task X.2)")`.
- In `gibbs_sampler`, replace that block's commented slot with the real call, conditioning on the
  latest draws of the other blocks; unpack its prior before the loop.
- Suite: **red** — the recovery test hits the stub. Expected; no commit.

**Human check HX.1** — Does the docstring formula match §2? Are argument order and shapes
consistent with the other steps (`data, p`, then the conditioning blocks in $\beta, \Sigma, \mu$ order, then
the prior)? Is the call in the right place in the sweep, and does it read the *current* values,
not the stored initial ones?

### X.2 — Implementation

- Write the body.
- **Task 3 only:** add
  `normal_posterior(prior_mean, prior_precision, data_precision, data_information) -> MvNormal` to
  `common/posteriors.jl` (symmetrise; solve through a Cholesky factor), with its testset in
  `test/tcvar_posteriors_test.jl`: closed form; vanishing prior precision → the GLS mean;
  overwhelming prior precision → the prior mean. Steps 1 and 3 call it as
  `normal_posterior(mean(prior), invcov(prior), P_d, b_d)`.
- Suite: the recovery test's "block stays at its initial value" assertion now **fails** —
  expected, and the proof the step is wired in. X.4 replaces it.

**Human check HX.2** — Read the body line by line against the docstring. Step 1: the Kronecker
order ($\Sigma^{-1} \otimes \tilde{X}^\top \tilde{X}$, not $\tilde{X}^\top \tilde{X} \otimes \Sigma^{-1}$) and $\operatorname{vec}(\tilde{X}^\top \tilde{Y} \Sigma^{-1})$. Step 2: degrees of freedom
$T - p + d$. Step 3: $W$ built from the **non-demeaned** data. Task 3: run
`test/tcvar_posteriors_test.jl`.

### X.3 — Unit tests

A testset in `test/var/steady_state_posteriors_test.jl` with:
- the shape check of §4.1;
- the log-density ratio identity of §4.2 — 50 random pairs per $(n, p)$, max discrepancy printed;
- **task 3 only:** on a random-walk dataset (`cumsum(randn(T, n), dims = 1)`), 200 draws of
  `draw_ss_var_coefficients` are all stationary, and `max_draws = 1` returns after one proposal.

**Human check HX.3** — Run the file alone. The printed max discrepancy should sit orders of
magnitude below the tolerance (≲ 1e-9), not just under it. Then break one thing on purpose —
flip the Kronecker order, or drop $d$ from the degrees of freedom — watch the identity test fail,
and revert: a one-minute check that the test has teeth.

### X.4 — Integration test

- In `ss_bvar_recovery_test.jl`, replace the block's "equals its initial value" assertion with
  posterior mean **and** median against the truth within the tolerance of the table above, plus
  "the draws vary" (`std > 0`).
- Extend the printed table to `truth / mean / median / sd / z`, with $z = (\text{mean} - \text{truth}) / \text{sd}$.
- Blocks not yet implemented keep their "equals its initial value" assertion:

| after | $\beta$ | $\Sigma$ | $\mu$ |
|---|---|---|---|
| H2 | = init | = init | = init |
| H3.4 | ≈ truth | = init | = init |
| H4.4 | ≈ truth | ≈ truth | = init |
| H5.4 | ≈ truth | ≈ truth | ≈ truth — full recovery under the off-truth priors of §5 |

**Human check HX.4** — Read the table: $\lvert z \rvert$ within ~3 everywhere, no systematic sign across
elements. Optional REPL trace to confirm no drift after burn-in, e.g.
`plot([b[1, 1] for b in vec(collect(result.params[@varname(β)]))])`. Task 5: confirm the
posterior mean of $\mu$ moved from the prior mean (truth ± 1) to the truth. Approve → commit, one
per task: *"SS-BVAR step 1: VAR coefficients"*, *"… step 2: innovation covariance"*,
*"… step 3: steady state"*.

---

## 6. Decisions taken, and the ones worth confirming

- **D1 — independent (Villani) prior on $\beta$, not the conjugate NIW $\beta \mid \Sigma \sim \mathcal{MN}(\beta_0, \Omega, \Sigma)$.** It is
  Villani's formulation and gives three genuinely separate conditional steps — the structure the
  tasks are built around. The alternative, the conjugate prior, draws $(\beta, \Sigma) \mid \mu$ jointly by
  calling the existing `sample_var_params` on demeaned data: fewer tasks and faster mixing, but
  steps 1 and 2 merge into one. **Confirm if you prefer the conjugate version.**
- **D2 — $\Sigma \sim \mathcal{IW}(d, S)$, not Villani's Jeffreys $\lvert \Sigma \rvert^{-(n+1)/2}$.** Jeffreys is improper, so it has
  no `logpdf`, and the §4.2 identity needs a proper prior. $\mathcal{IW}(d, S)$ approaches it as $d, S \to 0$.
- **D3 — constant steady state only** ($d_t = 1$). Deterministic regressors — a break dummy, a
  linear trend — generalise step 3 to $\operatorname{vec}(\Psi)$ with Villani's $U$, $D$ matrices without changing
  the sweep; out of scope for now.
- **D4 — likelihood conditional on the first $p$ observations.** The exact likelihood adds the
  stationary density of $y_{1:p}$, which breaks conjugacy in all three steps. Missing data is
  rejected: without a Kalman filter there is nothing to handle it.
- **D5 — data-driven initial values** (sample mean, OLS), overridable through `initial_values`.
  The step-by-step integration test is only meaningful if not-yet-drawn blocks start where the
  data puts them. With prior means, step 1 alone would condition on a $\mu$ pinned at its
  deliberately wrong prior mean, recover a biased $\beta$, and fail for a reason that has nothing to
  do with step 1.
- **D6 — stationarity by rejection** inside `draw_ss_var_coefficients`, as `sample_var_params`
  does. The pure posterior stays untruncated, so the §4.2 identity applies to it directly.

## 7. Out of scope

- SS-BVAR with stochastic volatility — would reuse `common/sv/` plus heteroskedastic versions of
  steps 1 and 3 (only the $(P_d, b_d)$ assembly changes).
- Forecasting from a result (`simulate_scenarios(::SSBVarResult, …)`), an estimation-and-plots
  script, marginal / predictive likelihood (`project_plan.md`) — `ss_var_loglikelihood` is the
  natural starting point for the last.
- Explicit `rng` threading — still deferred per `file_structure_refactor_plan.md`; tests use
  `Random.seed!`.

## References

- Villani, M. (2009), *Steady-state priors for vector autoregressions*, Journal of Applied
  Econometrics 24(4) — the model, the steady-state prior and its Gibbs sampler.
- Karlsson, S. (2013), *Forecasting with Bayesian Vector Autoregression*, Handbook of Economic
  Forecasting vol. 2B — a textbook statement of the steady-state VAR Gibbs sampler.
- Giannone, D., Lenza, M. & Primiceri, G. (2015), *Prior Selection for Vector Autoregressions*,
  REStat — the Minnesota prior already in `var/minnesota_prior.jl`.
