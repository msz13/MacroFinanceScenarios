"""
    MinnesotaPrior{T}

Coefficient block of the conjugate Normal–Inverse-Wishart ("Minnesota") prior for
a reduced-form BVAR, following Giannone, Lenza & Primiceri (2015), *Prior
Selection for Vector Autoregressions*, REStat.

Model (stacked form):

    Y = X Φ + E,     vec(E) ~ N(0, Σ ⊗ I_T)

where each row of `X` is `[ yₜ₋₁' … yₜ₋ₚ' ]`, optionally followed by an intercept
column `1`, so `X` has `k = n*p` columns (`n*p + 1` with an intercept) and `Φ` is
`k × n`. The intercept is off by default: the cycle VAR this prior is built for is
mean-zero by construction.

Prior:

    Σ      ~ IW(Ψ, d)              # supplied by the caller, kept by the caller
    Φ | Σ  ~ MN(Φ₀, Ω, Σ)          # vec(Φ) | Σ ~ N(vec(Φ₀), Σ ⊗ Ω)

Only the coefficient block `(Φ₀, Ω)` is stored. The innovation-covariance prior
is an `InverseWishart` owned by the caller: it enters the construction through
`E[Σ]` (which sets the scale of `Ω`) and is not copied into the struct, so there
is a single source of truth for `Ψ` and `d`.

Minnesota structure:
* `Φ₀` is zero except the own-first-lag of each equation, set to `δᵢ`
  (random walk when `δᵢ = 1`).
* `Ω` is diagonal; the entry for the coefficient on lag `s` of variable `j` is
  `λ² / (s² * σ̄ⱼ)`, with `σ̄ = diag(E[Σ]) = diag(Ψ)/(d-n-1)`. The *marginal*
  prior variance of that coefficient (in equation `i`) is therefore
  `λ²/s² * σ̄ᵢ/σ̄ⱼ` — the classic Minnesota shrinkage: tighter for higher lags,
  scaled by relative residual variances, and independent of `d`.
* When present, the intercept is the last regressor and gets a loose prior
  row-variance `ω_c`; see [`has_intercept`](@ref).

# Fields
- `n::Int`  number of endogenous variables
- `p::Int`  number of lags
- `k::Int`  regressors per equation (`= n*p`, or `n*p + 1` with an intercept)
- `λ::T`    overall tightness hyperparameter (→0 dogmatic, →∞ flat)
- `Φ₀`      `k × n` prior coefficient mean
- `Ω`       `k × k` diagonal prior row-covariance
"""
struct MinnesotaPrior{T<:Real}
    n::Int
    p::Int
    k::Int
    λ::T
    Φ₀::Matrix{T}
    Ω::Diagonal{T,Vector{T}}
end

"""
    MinnesotaPrior(λ, p, Σ_prior::InverseWishart; δ = ones(n), intercept = false, ω_c = 1e6)

Build the coefficient block of the Giannone–Lenza–Primiceri Minnesota prior.

# Arguments
- `λ`       : tightness hyperparameter (`> 0`).
- `p`       : number of lags (`≥ 1`).
- `Σ_prior` : `InverseWishart(d, Ψ)` prior on the innovation covariance. `n` is
              read off its size and `d > n+1` is required so that `E[Σ]` exists
              (the GLP default is `d = n+2`, where `E[Σ] = Ψ`). Only `E[Σ]` is
              used, to scale `Ω`; `Ψ` and `d` are not stored on the result.

# Keywords
- `δ`   : length-`n` prior mean of each variable's own first lag
          (default `1` ⇒ random-walk prior; use `0` for white-noise prior).

# Keywords (intercept)
- `intercept` : include an intercept regressor (default `false`, no intercept —
                the cycle VAR is mean-zero, so its prior is not needed).
- `ω_c`       : prior row-variance assigned to the intercept (default `1e6`,
                flat). Ignored when `intercept = false`.
"""
function MinnesotaPrior(λ::Real, p::Integer, Σ_prior::InverseWishart;
                        δ::AbstractVector{<:Real} = ones(size(Σ_prior, 1)),
                        intercept::Bool = false,
                        ω_c::Real = 1e6)
    n = size(Σ_prior, 1)
    d = first(params(Σ_prior))              # IW degrees of freedom

    p ≥ 1          || throw(ArgumentError("p must be ≥ 1"))
    λ > 0          || throw(ArgumentError("λ must be > 0"))
    intercept && ω_c ≤ 0 && throw(ArgumentError("ω_c must be > 0"))
    length(δ) == n || throw(DimensionMismatch("δ must have length n = $n"))
    d > n + 1      || throw(ArgumentError(
        "Σ_prior must have d > n+1 for E[Σ] to exist (got d = $d, n = $n)"))

    σ̄ = diag(mean(Σ_prior))                 # diag(E[Σ]) = diag(Ψ)/(d-n-1)
    all(>(0), σ̄) || throw(ArgumentError("all diagonal entries of E[Σ] must be > 0"))

    T = float(promote_type(typeof(λ), eltype(σ̄), eltype(δ), typeof(ω_c)))
    λ_, ωc_ = T(λ), T(ω_c)
    σ̄_, δ_  = collect(T, σ̄), collect(T, δ)

    k = n * p + (intercept ? 1 : 0)

    # ---- prior mean Φ₀: own first lag = δᵢ, everything else 0 ----
    Φ₀ = zeros(T, k, n)
    @inbounds for i in 1:n
        Φ₀[i, i] = δ_[i]
    end

    # ---- diagonal of Ω (row-covariance) ----
    # regressor order: [lag1 vars 1..n, lag2 vars 1..n, …, lagp vars 1..n(, const)]
    ω = Vector{T}(undef, k)
    @inbounds for s in 1:p, j in 1:n
        ω[(s - 1) * n + j] = λ_^2 / (s^2 * σ̄_[j])
    end
    intercept && (ω[k] = ωc_)               # loose intercept

    return MinnesotaPrior{T}(n, p, k, λ_, Φ₀, Diagonal(ω))
end

"""
    has_intercept(prior::MinnesotaPrior) -> Bool

Whether the prior carries an intercept regressor (the last row of `Φ₀` / `Ω`).
"""
has_intercept(pr::MinnesotaPrior) = pr.k > pr.n * pr.p

"""
    prior_var_coeff(prior::MinnesotaPrior) -> A

Companion bottom block `[A_p … A_1]` (`n × n*p`) of the VAR evaluated at the prior
coefficient mean `Φ₀`, in the **oldest-lag-first** ordering used by the state-space
cycle companion (`ξ_t = [c_{t-p+1}; …; c_t]`) and by [`is_stationary`](@ref).
`MinnesotaPrior` stores its regressors newest-lag-first (`[lag1 … lagp(, const)]`),
so the lag blocks are reversed and the intercept row, if any, is dropped.
"""
function prior_var_coeff(pr::MinnesotaPrior)
    n, p = pr.n, pr.p
    k = n * p                                # regressors without the intercept
    return collect(reshape(reverse(reshape(pr.Φ₀[1:k, :], n, p, n), dims = 2), k, n)')
end

"""
    initial_cycle_prior(β_prior::MinnesotaPrior, Σ_prior::InverseWishart) -> MvNormal

Prior on the initial cycle state `ξ₀ = [c_{-p+1}; …; c_0]` (length `n*p`,
oldest-lag-first, matching the state-space companion ordering).

The cycle is mean-zero by model assumption, so the mean is `0`. The covariance is
the *unconditional* (stationary) covariance of the companion vector implied by the
other two priors — the prior coefficient mean `Φ₀` and the prior mean innovation
covariance `E[Σ] = Ψ/(d-n-1)` — obtained from the discrete Lyapunov equation

    P = F P F' + Q,    vec(P) = (I − F⊗F)⁻¹ vec(Q)

where `F` is the companion matrix built from `Φ₀` and `Q` carries `E[Σ]` in the
contemporaneous-cycle block. With a white-noise prior mean (`δ = 0`, so `Φ₀ = 0`)
this reduces to `I_p ⊗ E[Σ]`.

Requires `Φ₀` to be stationary: a random-walk prior mean (`δ = 1`) has no
stationary distribution and throws.
"""
function initial_cycle_prior(β_prior::MinnesotaPrior, Σ_prior::InverseWishart)
    n, p = β_prior.n, β_prior.p
    n == size(Σ_prior, 1) || throw(DimensionMismatch(
        "β_prior is built for n = $n variables, Σ_prior for $(size(Σ_prior, 1))"))

    A = prior_var_coeff(β_prior)             # n × n*p, oldest-lag-first
    is_stationary(A, n, p) || throw(ArgumentError(
        "the prior coefficient mean Φ₀ is not stationary, so the initial cycle has " *
        "no unconditional variance (use δ < 1, e.g. δ = 0, for the cycle VAR)"))

    F = p == 1 ? A : vcat(hcat(zeros(n * (p - 1), n), I(n * (p - 1))), A)

    nc = n * p
    Q = zeros(nc, nc)
    Q[end-n+1:end, end-n+1:end] = mean(Σ_prior)   # E[Σ], contemporaneous cycle block

    P = reshape((I - kron(F, F)) \ vec(Q), nc, nc)

    return MvNormal(zeros(nc), Symmetric((P + P') / 2))
end

"""
    var_priors(λ, p, ψ; d = n + 2, δ = zeros(n), intercept = false, ω_c = 1e6)
        -> (Σ_prior, β_prior, c₀_prior)

Build the three priors of the cycle VAR(`p`) as **three separate objects** (no
wrapper struct):

1. `Σ_prior::InverseWishart` — innovation covariance, `Σ ~ IW(Ψ, d)` with
   `Ψ = Diagonal(ψ)`. With the default `d = n+2`, `ψ` is the prior mean of
   `diag(Σ)`.
2. `β_prior::MinnesotaPrior` — coefficients, `Φ | Σ ~ MN(Φ₀, Ω, Σ)`, built by
   [`MinnesotaPrior`](@ref) from `Σ_prior` (only `E[Σ]` enters, to scale `Ω`).
3. `c₀_prior::MvNormal` — initial cycle state `ξ₀` (length `n*p`,
   oldest-lag-first), mean zero with the unconditional variance implied by the
   first two, see [`initial_cycle_prior`](@ref).

`n` is read off `length(ψ)`.

# Arguments
- `λ` : Minnesota tightness (`> 0`).
- `p` : number of lags (`≥ 1`).
- `ψ` : length-`n` diagonal of the IW scale matrix `Ψ`.

# Keywords
- `d`   : IW degrees of freedom (`> n+1`, default `n+2`).
- `δ`   : prior mean of each variable's own first lag. Defaults to `0`
          (white-noise prior), because the cycle is stationary by construction —
          a random-walk prior mean (`δ = 1`) leaves `c₀_prior` undefined.
- `intercept` : include an intercept in the cycle VAR (default `false`).
- `ω_c` : prior row-variance of the intercept (default `1e6`, flat); ignored
          when `intercept = false`.

# Example
```julia
Σc_prior, β_prior, c₀_prior = var_priors(0.2, 4, [2.0, 1.0, 0.1, 25.0, 1.0])
```
"""
function var_priors(λ::Real, p::Integer, ψ::AbstractVector{<:Real};
                    d::Real = length(ψ) + 2,
                    δ::AbstractVector{<:Real} = zeros(length(ψ)),
                    intercept::Bool = false,
                    ω_c::Real = 1e6)
    n = length(ψ)
    n ≥ 1        || throw(ArgumentError("ψ must be non-empty"))
    all(>(0), ψ) || throw(ArgumentError("all ψⱼ must be > 0"))

    Σ_prior  = InverseWishart(float(d), Matrix(Diagonal(float.(ψ))))
    β_prior  = MinnesotaPrior(λ, p, Σ_prior; δ = δ, intercept = intercept, ω_c = ω_c)
    c₀_prior = initial_cycle_prior(β_prior, Σ_prior)

    return Σ_prior, β_prior, c₀_prior
end
