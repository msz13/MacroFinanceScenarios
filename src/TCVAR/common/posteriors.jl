# Model-agnostic posterior distributions and the draws taken from them.
#
# Two kinds of function live here, and the split is what makes the layer testable:
#
#   *_posterior(...)  pure, no RNG — returns a Distributions.jl object or posterior
#                     moments, so it can be checked against closed-form conjugate
#                     results without seeding anything.
#   draw_*(...)       consumes RNG, returns a draw. Thin: build the posterior, `rand` it.
#
# Nothing here knows about trends, cycles or the Minnesota prior; the model- and
# VAR-specific scales are assembled by the callers in `var/` and `models/`.

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
    random_walk_covariance_posterior(states, scale_prior, df_posterior) -> InverseWishart

Covariance posterior of a random-walk state, `xₜ = xₜ₋₁ + εₜ` with `εₜ ~ N(0, Σ)`.

The innovations are the first differences of the sampled state path, so this is
[`inverse_wishart_posterior`](@ref) applied to `diff(states, dims = 1)`. `states`
is `T × n` and must include the pre-sample point (a path of `T` points yields
`T − 1` innovations, which is what `df_posterior` should account for).
"""
random_walk_covariance_posterior(states, scale_prior, df_posterior) =
    inverse_wishart_posterior(diff(states, dims = 1), scale_prior, df_posterior)

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
    draw_from_factor(mean, L) -> Vector

Draw from `N(vec(mean), L L')` given a precomputed lower-triangular factor `L`
(e.g. from [`kron_cholesky_factor`](@ref)). `mean` may be a matrix; it is
vectorised column-wise to match the Kronecker layout.
"""
draw_from_factor(mean, L) = vec(mean) + L * randn(size(L, 1))

"""
    alp_posterior(Y, X, A, B0, h, alp0, Valp)
 
Full-conditional posteriors of the VAR coefficients in a BVAR with stochastic
volatility, built equation by equation as in the MATLAB routine, and returned as
a `product_distribution` of `n` `k`-variate normals (one per equation) instead
of a draw.
 
Model: `(Y - X*A) * B0' = E`, with `E[t, j] ~ N(0, exp(h[t, j]))` and `B0` unit
lower triangular. Column `ii` of `A` therefore shows up in the structural errors
of equations `ii:n`, which is what the `ii:n` slices below select.
 
Arguments
- `Y`    : `T×n` observations
- `X`    : `T×k` regressors (lags + deterministics)
- `A`    : `k×n` current draw of the coefficients; each factor conditions on the
           columns other than its own
- `B0`   : `n×n` unit lower triangular structural matrix
- `h`    : `T×n` log-volatilities
- `alp0` : `nk` prior mean, stacked equation by equation (`vec` of a `k×n` matrix)
- `Valp` : `nk` prior variances, same stacking (e.g. the second output of a
           Minnesota prior, recomputed each sweep if the shrinkage
           hyperparameters are themselves sampled)
 
The returned object is a matrix-variate product distribution, so `rand(d)` gives
a `k×n` matrix laid out exactly like `A`.
"""
function alp_posterior(Y::AbstractMatrix, X::AbstractMatrix, A::AbstractMatrix,
                       B0::AbstractMatrix, h::AbstractMatrix,
                       alp0::AbstractVector, Valp::AbstractVector)
    T, n = size(Y)
    k = size(X, 2)
    @assert size(A) == (k, n)
    @assert size(X, 1) == T && size(h) == (T, n)
    @assert length(alp0) == length(Valp) == n * k
 
    sqrt_exph = exp.(h ./ 2)
    A = copy(A)                                     # never mutate the caller's A
 
    dists = map(1:n) do ii
        aii = A[:, ii]
        A[:, ii] .= 0                               # drop equation ii's own fit
 
        Λ  = vec(sqrt_exph[:, ii:n])                # T(n-ii+1)
        yi = vec((Y - X * A) * B0[ii:n, :]') ./ Λ   # T(n-ii+1)
        Wi = kron(B0[ii:n, ii], X) ./ Λ             # T(n-ii+1) × k
 
        A[:, ii] .= aii                             # restore for the next equation
 
        idx    = (ii - 1) * k + 1 : ii * k
        iValpi = Diagonal(1 ./ view(Valp, idx))
        alpi0  = view(alp0, idx)
 
        Kalpi = Symmetric(iValpi + Wi' * Wi)        # posterior precision
        bi    = iValpi * alpi0 + Wi' * yi          # potential vector
        MvNormalCanon(bi, Kalpi)                    # mean = Kalpi \ bi, cov = inv(Kalpi)
    end
 
    return product_distribution(dists)
end
