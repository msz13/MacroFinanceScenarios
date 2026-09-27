# `NormalInverseWishart` — refactor plan for `common/distributions.jl`

Turn `normal_inverse_wishart_posterior` from a NamedTuple-of-closure into a first-class
`Distributions.jl` distribution: one struct carrying the four NIW parameters, with a
constructor, `rand` (draws **both** coefficients and covariance) and `logpdf` (evaluates the
**joint** density of both). The distribution itself is model-agnostic and lands in
`common/distributions.jl`; `var/var_sampling.jl` keeps only the VAR-specific job of reading
the four parameters off `(Y, X, prior)`.

## 0. Why

`normal_inverse_wishart_posterior` has now been through two shapes, both unsatisfying:

* `product_distribution((β = …, Σ = …))` (commit `d998dd0^`) — wrong: a product distribution
  claims independence, but `Cov(vec β) = Σ ⊗ V` depends on `Σ`, so `rand` returned a `β` drawn
  under a `Σ` it was not paired with.
* `(β = Σ -> MvNormal, Σ = InverseWishart)` (HEAD) — correct but not a distribution: no `rand`,
  no `logpdf`, the conditional hidden in a closure, and the NIW parameters unreachable
  (`post.β` cannot tell you `β̂`).

A struct fixes both and buys three things the codebase already has uses for:

1. **`logpdf`** — needed the moment a Metropolis–Hastings step appears (`temp_spec.md`:
   *"add metropolis hastings step"*; `ss_bvar_plan.md` §D). A ratio needs a density, and the
   joint NIW density is currently unspellable in one call.
2. **The same type for the prior** — an NIW is conjugate, so prior and posterior are the same
   object. That is the enabler for `temp_spec.md`'s *"zrefactorowac tcvar — aby był pełny
   natural conjugate priors"*: `MinnesotaPrior` + `cycle_covariance` collapse into one
   `NormalInverseWishart(Φ₀, Ω, d, Ψ)`, and `sample_var_params` takes it instead of four loose
   arguments.
3. **A joint draw in one call** — for prior/posterior check scripts and simulation, where the
   caller wants `(β, Σ)` and does not care about the two-step conditional structure.

## 1. The distribution

For coefficients `β` (`k × n`) and innovation covariance `Σ` (`n × n`), parameters
`(M, V, ν, Ψ)`:

$$
\begin{aligned}
\Sigma &\sim \mathcal{IW}(\nu, \Psi) \\
\operatorname{vec}(\beta) \mid \Sigma &\sim \mathcal{N}\big(\operatorname{vec}(M),\ \Sigma \otimes V\big)
\end{aligned}
$$

equivalently `β | Σ ~ MN(M, V, Σ)` (matrix normal, row covariance `V`, column covariance `Σ`).
`vec` is column-major, so it stacks **equation blocks** and `Σ ⊗ V` is written for that
ordering — the one `kron_cholesky_factor(Σ, V)` already assumes. `V` is `k × k`
(*within*-equation), `Σ` is `n × n` (*across*-equation).

Joint log density, which is what `logpdf` computes:

$$
\log p(\beta, \Sigma) = \log \mathcal{N}\big(\operatorname{vec}\beta;\ \operatorname{vec}M,\ \Sigma \otimes V\big) \;+\; \log \mathcal{IW}(\Sigma;\ \nu, \Psi)
$$

As the VAR posterior of `d998dd0`, the four parameters are

| parameter | value | built by |
|---|---|---|
| `M` | `β̂ = (X'X + Ω⁻¹)⁻¹ (X'Y + Ω⁻¹ β₀)` | `normal_coefficient_posterior_mean` (unchanged) |
| `V` | `(X'X + Ω⁻¹)⁻¹` | `inv(Symmetric(X'X + Ω_inv))`, hoisted into the constructor |
| `ν` | `df` | caller |
| `Ψ` | `ε'ε + (β̂ − β₀)' Ω⁻¹ (β̂ − β₀) + S`, `ε = Y − X β̂` | `var_covariance_scale` (was `var_covariance_posterior`) |

## 2. Tree delta

```
src/TCVAR/
├── common/
│   ├── posteriors.jl          # unchanged except one added 3-arg method (§3.2)
│   └── distributions.jl       # ← NEW: NormalInverseWishart + rand/logpdf/accessors
└── var/
    └── var_sampling.jl        # normal_inverse_wishart_posterior now returns the struct
test/
└── common/
    └── distributions_test.jl  # ← NEW
```

`TCVAR.jl`: `include("common/distributions.jl")` immediately after `common/posteriors.jl`
(it calls `inverse_wishart_posterior`, `kron_cholesky_factor` and `normal_coefficient_posterior`,
and `var/var_sampling.jl` names the type in its docstring and return position).
`test/runtests.jl`: include the new test file right after `tcvar_posteriors_test.jl`.

## 3. The interface

### 3.1 `common/distributions.jl`

```julia
"""
    NormalInverseWishart(β_mean, V, df, scale)

Matrix-normal–inverse-Wishart distribution over a coefficient matrix and an innovation
covariance … [math of §1, field table, the two accessors, the jitter caveat of §4]
"""
struct NormalInverseWishart{T<:Real} <: Distribution{NamedTupleVariate{(:β, :Σ)}, Continuous}
    β_mean::Matrix{T}   # k × n  coefficient mean M
    V::Matrix{T}        # k × k  row covariance of vec(β):  Cov = Σ ⊗ V
    df::T               # ν > n − 1
    scale::Matrix{T}    # n × n  IW scale Ψ

    function NormalInverseWishart(β_mean, V, df, scale)   # inner: validate + symmetrise
        # size(β_mean) == (k, n); size(V) == (k, k); size(scale) == (n, n); df > n − 1
        # V and scale stored as collect(Hermitian(·)) — see §4.3
    end
end

Distributions.params(d) = (d.β_mean, d.V, d.df, d.scale)
coefficient_size(d)     = size(d.β_mean)            # (k, n)

"""Σ's marginal, IW(ν, Ψ) — the block to draw first."""
marginal_covariance(d) = inverse_wishart_posterior(d.scale, d.df)

"""vec(β) | Σ ~ N(vec(M), Σ ⊗ V), as an MvNormal carrying the Kronecker Cholesky factor."""
conditional_coefficients(d, Σ) = normal_coefficient_posterior(d.β_mean, Σ, d.V)

"""(β = vec-draw, Σ = matrix-draw): Σ from its marginal, then β given that draw."""
function Base.rand(rng::AbstractRNG, d::NormalInverseWishart)
    Σ = rand(rng, marginal_covariance(d))
    return (β = rand(rng, conditional_coefficients(d, Σ)), Σ = Σ)
end

"""Joint log density; `x.β` may be the k×n matrix or its vec."""
Distributions.logpdf(d::NormalInverseWishart, x::NamedTuple{(:β, :Σ)}) =
    logpdf(conditional_coefficients(d, x.Σ), vec(x.β)) + logpdf(marginal_covariance(d), x.Σ)

Distributions.pdf(d, x)       = exp(logpdf(d, x))
Distributions.insupport(d, x) = size checks && isposdef(x.Σ)
```

`rand(d)` (no RNG) comes free: `rand(s::Sampleable, dims::Int...)` in
`Distributions/src/genericrand.jl` forwards to `rand(default_rng(), s)`, so the global-RNG
path still runs through the method above. Nothing else is inherited — for
`NamedTupleVariate`, Distributions defines its methods on `ProductNamedTupleDistribution`
only, not generically on the variate form — so `logpdf`, `pdf` and `insupport` are ours to
write, which is what the sketch does. `NamedTupleVariate` is exported by Distributions and
`TCVAR.jl` already does `using Distributions`.

### 3.2 `common/posteriors.jl` — one added method, nothing removed

```julia
normal_coefficient_posterior(β_posterior_mean, Σ, V) =            # ← NEW 3-arg core
    MvNormal(vec(β_posterior_mean), PDMat(Cholesky(LowerTriangular(kron_cholesky_factor(Σ, V)))))

normal_coefficient_posterior(β_posterior_mean, X, Σ, Ω_inv) =     # existing 4-arg, unchanged behaviour
    normal_coefficient_posterior(β_posterior_mean, Σ, inv(Symmetric(X'X + Ω_inv)))
```

Argument order `(mean, Σ, V)` mirrors `kron_cholesky_factor(Σ, V)`. Arities 3 and 4 do not
collide, the existing 4-arg body is unchanged line for line, so
`test/tcvar_posteriors_test.jl` needs no edit. `normal_coefficient_posterior_mean`,
`kron_cholesky_factor` and `draw_from_factor` are untouched — `tcvar_sv_plan.md` §7 and
`ss_bvar_plan.md` §4 call them directly.

### 3.3 `var/var_sampling.jl`

```julia
# NIW scale only — the distribution is assembled by the constructor below.
function var_covariance_scale(Y, X, β_posterior_μ, variance_prior, β_prior_μ, Ω_inv)
    ε = Y - X * β_posterior_μ
    β_diff = β_posterior_μ - β_prior_μ
    return ε' * ε + β_diff' * Ω_inv * β_diff + variance_prior      # same terms, same order
end

function normal_inverse_wishart_posterior(Y, X, β_prior_μ, Ω_inv, S, df)   # signature unchanged
    β_hat = normal_coefficient_posterior_mean(Y, X, β_prior_μ, Ω_inv)
    return NormalInverseWishart(β_hat,
                                inv(Symmetric(X'X + Ω_inv)),
                                df,
                                var_covariance_scale(Y, X, β_hat, S, β_prior_μ, Ω_inv))
end
```

`sample_var_params` keeps its two-step structure — it must, because the rejection loop
re-draws `β` at a **fixed** `Σ`, and `rand(d)` would draw a new `Σ` each time:

```julia
posterior = normal_inverse_wishart_posterior(Y, X, β_prior_μ, Ω_inv, S, df)

Σ = rand(marginal_covariance(posterior))
β_posterior = conditional_coefficients(posterior, Σ)   # factor built once, reused below
β = rand(β_posterior)
# … unchanged stationarity-rejection loop on β_posterior
```

So `rand(posterior)` and these three lines consume the RNG identically (§4.1); the difference
is only that the sampler keeps the conditional around. The docstring says this explicitly, so
nobody "simplifies" the sampler into `rand(posterior)` and silently rebuilds the Kronecker
factor once per rejection draw — the reuse that `gibbs_sampler_performance_plan.md` §3 exists
to protect.

## 4. Constraints and decisions

### 4.1 The gate: `tcvar_gibbs_regression_test.jl` must stay green with unchanged checksums

It folds the IEEE bits of every draw of a seeded sweep, so it catches any change in RNG
consumption *or* in floating-point association. Three things therefore may not move:

* **Draw order** stays `rand(::InverseWishart)` then `rand(::MvNormal)`, one of each per sweep.
* **`normal_coefficient_posterior_mean` keeps its own `inv(X'X + Ω_inv)`** — note it has *no*
  `Symmetric`, while the covariance path has `inv(Symmetric(X'X + Ω_inv))`. Tempting as it is
  to compute `V` once and reuse it for the mean, the two spellings return different bits.
  **D1: keep both spellings.** Unifying them is a separate, deliberate commit that re-derives
  the checksums.
* **Scale assembly** keeps the term order `ε' * ε + β_diff' * Ω_inv * β_diff + variance_prior`
  (float addition is not associative — the note in `inverse_wishart_posterior`'s docstring).

Hoisting `inv(Symmetric(X'X + Ω_inv))` from `normal_coefficient_posterior` into the
constructor is the same expression on the same inputs, so it is bit-identical; it just runs
once per sweep instead of once per call.

### 4.2 `logpdf` is the density of what `rand` samples, jitter included (D2)

`kron_cholesky_factor` adds `1e-5·I` to both blocks before factorising, so the conditional
covariance is `(Σ + εI) ⊗ (V + εI)`, not `Σ ⊗ V`. Two options:

* **(chosen)** build `logpdf` from the same jittered factor: `logpdf` is then exactly the
  density of the distribution `rand` draws from, which is the property an MH ratio needs, and
  there is one code path instead of two.
* the exact closed-form NIW density: bit-exact against textbook formulas, but inconsistent
  with `rand` at `O(1e-5)`.

The closed form becomes the **test oracle** (§5), and the docstring states the caveat — the
same wording `normal_coefficient_posterior` already uses for `cov`. Measured on a probe of the
sketch in §3.1 (`k = 4`, `n = 2`, `V = 0.5·I`, `ν = 12`): jittered `logpdf` `11.68676` against
the oracle's `11.68718`, a gap of `4.1e-4`, so **`atol = 1e-3`** is the tolerance for that
comparison.

### 4.3 Remaining decisions

* **D3 — where the type lives:** `common/distributions.jl`. The distribution knows nothing
  about VARs; only the `(Y, X, prior) → parameters` assembly does, and that stays in
  `var/var_sampling.jl`. Alternative (a generic `niw_regression_posterior` in `common/`) is
  deferred: nothing but the VAR needs it yet.
* **D4 — name:** `NormalInverseWishart`, matching `normal_inverse_wishart_posterior`. It does
  not collide with Distributions.jl (which has no such type). It *does* differ from
  `ConjugatePriors.NormalInverseWishart`, which is the vector-mean version — the docstring
  says "matrix-normal" in its first line to head that off. Not a dependency.
* **D5 — `params`-only struct:** `marginal_covariance` builds the `InverseWishart` on demand
  (one `n × n` Cholesky per call, ~nothing next to the Kalman filter) rather than the struct
  caching a distribution object. Keeps `params`, `==` and `show` meaning what they usually do.
* **D6 — `rand` returns `β` as the `nk` vector**, matching `sample_var_params`'s return and how
  `tcvar_gibbs.jl` stores it (`betas[s, :]`). `logpdf` accepts either shape (`vec` of a vector
  is free). Key order is `(:β, :Σ)` — as today's NamedTuple — which is deliberately *not* the
  draw order; the docstring says so.
* **D7 — exports:** export `NormalInverseWishart` (a user-facing type, like `MinnesotaPrior`);
  leave `marginal_covariance` / `conditional_coefficients` / `var_covariance_scale` internal,
  reached as `TCVAR.f`, as the whole `posteriors.jl` layer is today.
* **D8 — `rand(d, n)` (many draws) is not implemented.** `ProductNamedTupleDistribution` needs
  its own `_rand!` for that and no caller wants it; `[rand(d) for _ in 1:n]` is what the check
  scripts use.

## 5. Tests

**`test/common/distributions_test.jl`** (new; carries the usual
`isdefined(Main, :TCVAR) || include(…)` guard, reaches members as `TCVAR.f`, and the
stale-module guard `isdefined(TCVAR, :NormalInverseWishart) || error("restart the session…")`
that `tcvar_posteriors_test.jl` established):

* **constructor** — `params` round-trip; `DimensionMismatch` on each of the three bad shapes;
  `ArgumentError` on `df ≤ n − 1`; an asymmetric `scale`/`V` comes back symmetrised.
* **`marginal_covariance`** — `params` equal `(df, Hermitian(scale))`.
* **`conditional_coefficients`** — `mean ≈ vec(β_mean)`; `cov ≈ kron(Σ, V)` at `atol = 1e-4`;
  `cov(·, 2Σ) ≈ 2 cov(·, Σ)` (the jitter-tolerant scaling check already in
  `var_sampling_test.jl`).
* **`rand`** — keys `(:β, :Σ)`, sizes `(k*n,)` and `(n, n)`; `Σ` symmetric positive definite;
  `rand(Xoshiro(1), d) == rand(Xoshiro(1), d)`; and the order pin: under one seed,
  `rand(d)` equals `Σ = rand(marginal_covariance(d)); β = rand(conditional_coefficients(d, Σ))`
  — the assertion that ties the struct to §4.1.
* **`logpdf`** — additivity (`= logpdf(conditional) + logpdf(marginal)`); equal for `β` passed
  as `k × n` and as `vec`; and against a **closed-form oracle written naively in the test**
  no Kronecker algebra, no `kron_cholesky_factor`), at the `atol = 1e-3` of §4.2 — the one test
  that would catch a wrong Kronecker ordering:

  ```julia
  E = β - d.β_mean                      # k × n
  log_mn = -0.5n*logdet(d.V) - 0.5k*logdet(Σ) - 0.5k*n*log(2π) -
           0.5tr(inv(Σ) * E' * inv(d.V) * E)
  log_mn + logpdf(InverseWishart(d.df, d.scale), Σ)
  ```
* **Monte Carlo** — over ~20 000 draws, `mean(Σ) ≈ Ψ/(ν − n − 1)` and `mean(β) ≈ vec(M)`, loose
  tolerances. Cheap, and it is what catches `rand` pairing `β` with the wrong `Σ` (the bug the
  `product_distribution` version had, which no moment-free test noticed).

**`test/var/var_sampling_test.jl`** (rewritten, same fixture): `post.β_mean ≈ β̂`;
`post.V ≈ inv(X'X + Ω_inv)`; `post.df == df`; `post.scale ≈ ε'ε + (β̂−β₀)'Ω⁻¹(β̂−β₀) + S`; and
the joint-density assertion deleted in the HEAD diff, restored now that it is meaningful:
`logpdf(post, (β = b, Σ = Σ))` ≈ the sum of the two blocks.

**Unchanged and used as the gate:** `test/tcvar_posteriors_test.jl` (no edit — §3.2) and
`test/models/tcvar/tcvar_gibbs_regression_test.jl` (must pass with the checksums it has).

## 6. Tasks and checkpoints

Each task ends in a check you run before the next starts; the suite is
`julia --project test/runtests.jl`, and each new file also runs alone.

| Checkpoint | Task | State after it | Commit |
|---|---|---|---|
| H0 | — (done: §3.1 probed in a scratch script against this project's `Distributions` 0.25.127) | the five claims of §8 hold | no |
| H1 | `common/distributions.jl` + the 3-arg `normal_coefficient_posterior` + include in `TCVAR.jl`. No caller changed. | green — nothing calls it yet | no |
| H2 | `test/common/distributions_test.jl` (§5) + runtests entry | green; the closed-form oracle and the MC moments both pass | yes |
| H3 | `var/var_sampling.jl`: `var_covariance_scale`, constructor returns the struct, `sample_var_params` on the accessors | green **and** the regression checksums unchanged — the §4.1 gate | no |
| H4 | `test/var/var_sampling_test.jl` rewritten (§5) | green | yes |
| H5 | Docstrings: the struct's own, the two accessors, the "why the sampler does not call `rand`" note, `sample_var_params`' `-> NormalInverseWishart` | green | yes |

If H3 moves a checksum, the cause is one of the three items in §4.1 — find it rather than
re-deriving the checksums, which is what the regression test's own header tells you.

## 7. Out of scope

* **`sample_var_params(data, p, prior::NormalInverseWishart)`** — the natural follow-up (D-why
  §0.2) and the one that makes the Minnesota prior an NIW prior. It changes `tcvar_gibbs.jl`
  and `tcvar_priors.jl`, so it is its own plan.
* **Stationarity truncation inside the distribution** (a `truncated`-style wrapper whose `rand`
  rejects explosive draws and whose `logpdf` carries the normalising constant). The rejection
  loop stays in `sample_var_params`; the constant is not identified by anything we sample.
* **Threading `rng` through `draw_from_factor`** — still deferred, as
  `file_structure_refactor_plan.md` says. `NormalInverseWishart`'s `rand` takes an `rng` and
  honours it (it goes through `rand(rng, ::MvNormal)`, not `draw_from_factor`), so this plan
  does not depend on that cleanup.
* **The marginal `p(β) = MatrixTDist`** and the marginal likelihood `p(Y)`. Both are one-liners
  off these parameters and neither has a caller yet.

## 8. What was already checked

The §3.1 sketch was run as a standalone script against this project's `Distributions`
0.25.127 before this plan was written, because five of its claims are load-bearing and none is
documented:

1. `rand(d)` with no RNG resolves through `rand(s::Sampleable, dims::Int...)` to our
   `rand(rng, d)` — confirmed, `(β, Σ)` of sizes `(nk,)` and `(n, n)`.
2. `rand(d)` is **bit-identical** to `Σ = rand(marginal_covariance(d)); β = rand(conditional_coefficients(d, Σ))`
   under one seed — the §4.1 gate holds, so `sample_var_params` and `rand` consume the same stream.
3. `rand(Xoshiro(7), d)` is reproducible: the RNG reaches both blocks.
4. `logpdf` is additive across the blocks and identical for `β` as `k × n` or as `vec`;
   against the closed form, the gap is the `4.1e-4` of §4.2.
5. `Base.show`'s `Distribution` fallback prints the four fields without extra methods, and
   `20 000` draws recover `Ψ/(ν−n−1)` to three digits and `vec(M)` to `2e-3`.
