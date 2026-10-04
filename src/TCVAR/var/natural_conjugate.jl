# Natural-conjugate (normal–inverse-Wishart) posterior of a VAR, as a joint
# distribution over the coefficients and the innovation covariance.
#
# Model, for `Y` (`T × n`) on `X` (`T × k`) with coefficients `B` (`k × n`):
#
#   Y = X B + E,         rows of E ~ N(0, Σ)
#   vec(B) | Σ  ~  N(vec(B₀), Σ ⊗ Ω)
#   Σ           ~  IW(ν₀, S₀)
#
# Posterior:
#
#   Σ | Y        ~  IW(ν₀ + T, S̄),       S̄ = S₀ + Ê'Ê + (B̄ − B₀)' Ω⁻¹ (B̄ − B₀)
#   vec(B) | Σ, Y ~ N(vec(B̄), Σ ⊗ Ω̄),    Ω̄ = (X'X + Ω⁻¹)⁻¹,  B̄ = Ω̄ (X'Y + Ω⁻¹ B₀)
#
# with Ê = Y − X B̄. Coefficients are handled as `vec(B)` (length `k·n`), the layout
# the Gibbs sampler stores them in.

"""
    inverse_wishart_posterior(scale_posterior, df_posterior) -> InverseWishart
    inverse_wishart_posterior(residuals, scale_prior, df_posterior) -> InverseWishart

Conjugate covariance posterior `Σ ~ IW(df_posterior, scale_posterior)`.

The three-argument form assembles the usual `ε'ε + scale_prior`, where `residuals`
is a `T × n` matrix of innovations already differenced / de-meaned /
regression-residualised by the caller.

Callers whose scale carries extra model-specific terms — e.g. the coefficient
shrinkage `(β̂ − β₀)' Ω⁻¹ (β̂ − β₀)` of a normal–inverse-Wishart VAR posterior —
assemble it themselves and use the two-argument form. Prefer that to folding the
extra term into `scale_prior`: floating-point addition is not associative, so the
two spellings do not agree bit-for-bit.

The scale is symmetrised before being handed to `InverseWishart`: `ε'ε` is exactly
symmetric, but scale terms built from three-factor products such as
`β_diff' * Ω_inv * β_diff` are only symmetric up to rounding.
"""
inverse_wishart_posterior(scale_posterior, df_posterior) =
    InverseWishart(df_posterior, collect(Hermitian(scale_posterior)))

inverse_wishart_posterior(residuals, scale_prior, df_posterior) =
    inverse_wishart_posterior(residuals' * residuals .+ scale_prior, df_posterior)

"""
    normal_coefficient_posterior_mean(Y, X, β_prior_mean, Ω_inv) -> Matrix

Posterior mean of the conjugate normal regression coefficients,

    (X'X + Ω⁻¹)⁻¹ (X'Y + Ω⁻¹ β₀)

for `Y` (`T × n`) on `X` (`T × k`), with prior mean `β_prior_mean` (`k × n`) and
prior precision `Ω_inv` (`k × k`). Returns the `k × n` posterior mean; note this
is the *mean only* — the posterior covariance factor is
[`kron_cholesky_factor`](@ref).
"""
normal_coefficient_posterior_mean(Y, X, β_prior_mean, Ω_inv) =
    inv(X'X + Ω_inv) * (X'Y + Ω_inv * β_prior_mean)

"""
    kron_cholesky_factor(Σ, V) -> L

Lower-triangular Cholesky factor `L` of the Kronecker-structured coefficient
posterior covariance `Σ ⊗ V` (so `L * L' ≈ Σ ⊗ V`). For a conjugate VAR,
`V = (X'X + Ω⁻¹)⁻¹`.

Uses the identity that the Cholesky factor of a Kronecker product is the
Kronecker product of the factors:

    chol(A ⊗ B) = chol(A) ⊗ chol(B)

so the `m × m` factor (`m = n·k`) is assembled from the small `n × n` and `k × k`
blocks instead of decomposing the full matrix. Because this factor does not depend
on the proposed coefficients, it is computed once and reused across the
stationarity-rejection draws in [`sample_var_params`](@ref).

A small jitter is added to both blocks before factorising to keep them positive
definite against rounding.
"""
function kron_cholesky_factor(Σ, V)
    jitter = 1e-5
    Σ_L = cholesky(Symmetric(Σ) + jitter * I).L   # n × n
    V_L = cholesky(Symmetric(V) + jitter * I).L   # k × k
    return kron(Σ_L, V_L)
end

"""
    NaturalConjugate(Y, X, β_prior::MinnesotaPrior, sigma_prior::InverseWishart)

Joint posterior of `(vec(B), Σ)` for the conjugate VAR described above.

The coefficient prior is read off `β_prior`: `B₀` from [`prior_coeff_mean`](@ref) and
`Ω⁻¹` from [`prior_row_precision`](@ref). Both are in the oldest-lag-first, no-intercept
layout of [`prepare_var_data`](@ref), so `X` must have `k = n*p` columns.
`sigma_prior` is the `IW(ν₀, S₀)` prior on `Σ`.

`rand(d)` returns `(β, Σ)` with `β = vec(B)`; `logpdf(d, β, Σ)` is the joint posterior
log density. The marginal of `Σ` is [`covariance_posterior`](@ref) and the conditional
of `β` given `Σ` is [`coefficient_posterior`](@ref).
"""
struct NaturalConjugate{TB<:AbstractMatrix,TΩ<:AbstractMatrix,TS<:InverseWishart} <: Sampleable{Multivariate,Continuous}
    β_μ::TB        # B̄, k × n posterior mean of the coefficients
    Ω::TΩ          # Ω̄, k × k posterior row covariance of the coefficients
    Σ_posterior::TS
end

function NaturalConjugate(Y, X, β_prior::MinnesotaPrior, sigma_prior::InverseWishart)
    size(X, 2) == β_prior.n * β_prior.p || throw(DimensionMismatch(
        "X has $(size(X, 2)) columns but the prior expects n*p = $(β_prior.n * β_prior.p)"))

    β_prior_μ = prior_coeff_mean(β_prior)
    Ω_inv = prior_row_precision(β_prior)
    ν₀, S₀ = params(sigma_prior)
    T = size(Y, 1)

    β_μ = normal_coefficient_posterior_mean(Y, X, β_prior_μ, Ω_inv)
    Ω = inv(Symmetric(X'X + Ω_inv))

    # Scale assembled in this order on purpose: the Gibbs non-regression test pins
    # the draws bit-for-bit, and floating-point addition is not associative.
    ε = Y - X * β_μ
    β_diff = β_μ - β_prior_μ
    S = ε' * ε + β_diff' * Ω_inv * β_diff + Matrix(S₀)
    

    return NaturalConjugate(β_μ, Ω, inverse_wishart_posterior(S, ν₀ + T))
end

Base.length(d::NaturalConjugate) = length(d.β_μ)

"""
    covariance_posterior(d::NaturalConjugate) -> InverseWishart

Marginal posterior of the innovation covariance, `Σ | Y ~ IW(ν₀ + T, S̄)`.
"""
covariance_posterior(d::NaturalConjugate) = d.Σ_posterior

"""
    coefficient_posterior(d::NaturalConjugate, Σ) -> MvNormal

Conditional posterior of the coefficients, `vec(B) | Σ, Y ~ N(vec(B̄), Σ ⊗ Ω̄)`.
Dense in `k·n`, so meant for evaluating densities; draws go through
[`rand_coefficients`](@ref), which uses the Kronecker factor instead.
"""
coefficient_posterior(d::NaturalConjugate, Σ) =
    MvNormal(vec(d.β_μ), collect(Hermitian(kron(Σ, d.Ω))))

"""
    rand_coefficients([rng], d::NaturalConjugate, Σ) -> Vector

Draw `vec(B)` from the conditional posterior given `Σ`. With `Σ` fixed the factor
can be reused across draws — see [`coefficient_factor`](@ref).

The factor comes from [`kron_cholesky_factor`](@ref), which jitters both blocks by
`1e-5`, so draws are from `N(vec(B̄), (Σ + εI) ⊗ (Ω̄ + εI))`, a hair wider than
[`coefficient_posterior`](@ref).
"""
rand_coefficients(rng::AbstractRNG, d::NaturalConjugate, Σ) =
    rand_coefficients(rng, d, Σ, coefficient_factor(d, Σ))

rand_coefficients(rng::AbstractRNG, d::NaturalConjugate, Σ, L) =
    vec(d.β_μ) + L * randn(rng, size(L, 1))

rand_coefficients(d::NaturalConjugate, args...) =
    rand_coefficients(Random.default_rng(), d, args...)

"""
    coefficient_factor(d::NaturalConjugate, Σ) -> L

Lower-triangular factor with `L * L' ≈ Σ ⊗ Ω̄`, to pass to
[`rand_coefficients`](@ref) when drawing repeatedly for the same `Σ`.
"""
coefficient_factor(d::NaturalConjugate, Σ) = kron_cholesky_factor(Σ, d.Ω)

function Base.rand(rng::AbstractRNG, d::NaturalConjugate)
    Σ = rand(rng, covariance_posterior(d))
    return rand_coefficients(rng, d, Σ), Σ
end

Base.rand(d::NaturalConjugate) = rand(Random.default_rng(), d)

"""
    logpdf(d::NaturalConjugate, β, Σ)

Joint posterior log density `log p(β | Σ, Y) + log p(Σ | Y)`. `β` may be the `k × n`
matrix or its `vec`.
"""
Distributions.logpdf(d::NaturalConjugate, β, Σ) =
    logpdf(coefficient_posterior(d, Σ), vec(β)) + logpdf(covariance_posterior(d), Σ)
