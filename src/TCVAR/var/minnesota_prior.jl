#= """
    coef_cov(prior, Σ) -> Symmetric

Full conditional prior covariance of `vec(Φ)` given `Σ`, i.e. `Σ ⊗ Ω`.
"""
coef_cov(pr::MinnesotaPrior, Σ::AbstractMatrix) =
    Symmetric(kron(Matrix(Σ), Matrix(pr.Ω))) =#


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
    prior_row_covariance(prior::MinnesotaPrior) -> Diagonal

Prior row covariance `Ω` (`n*p × n*p`) in the **oldest-lag-first** ordering used by the
state-space cycle companion, with the intercept row dropped. Counterpart of
[`prior_var_coeff`](@ref) for the second moment: `MinnesotaPrior` stores its regressors
newest-lag-first (`[lag1 … lagp(, const)]`), so the lag blocks are reversed here.
"""
function prior_row_covariance(pr::MinnesotaPrior)
    k = pr.n * pr.p                          # regressors without the intercept
    return Diagonal(vec(reverse(reshape(diag(pr.Ω)[1:k], pr.n, pr.p), dims = 2)))
end
