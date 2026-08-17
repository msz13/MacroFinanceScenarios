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
