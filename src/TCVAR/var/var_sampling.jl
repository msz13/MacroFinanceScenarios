# Normal–inverse-Wishart posterior for a VAR, and the draw taken from it.
# The generic pieces (the IW constructor, the conjugate coefficient mean and its
# conditional normal posterior) live in `common/posteriors.jl`; what is VAR-specific
# and stays here is the NIW *scale*, the pairing of the two blocks, and the
# stationarity-rejection loop.

"""
    var_covariance_posterior(Y, X, β_posterior_μ, posterior_df, variance_prior, β_prior_μ, Ω_inv)

Conditional covariance posterior of a conjugate VAR:

    Σ | Y, X  ~  IW(posterior_df, ε'ε + (β̂ − β₀)' Ω⁻¹ (β̂ − β₀) + variance_prior)

with residuals `ε = Y − X β̂`. The coefficient-shrinkage term is the NIW-specific
part of the scale; the distribution itself is built by
[`inverse_wishart_posterior`](@ref).

`Ω_inv` is the prior precision of the coefficients.
"""
function var_covariance_posterior(Y, X, β_posterior_μ, posterior_df, variance_prior, β_prior_μ, Ω_inv)

    ε = Y - X * β_posterior_μ

    β_diff = β_posterior_μ - β_prior_μ

    S = ε' * ε + β_diff' * Ω_inv * β_diff + variance_prior

    return inverse_wishart_posterior(S, posterior_df)

end


"""
    normal_inverse_wishart_posterior(Y, X, β_prior_μ, Ω_inv, S, df)
        -> (β = Σ -> MvNormal, Σ = InverseWishart)

Normal–inverse-Wishart posterior of a conjugate VAR, `p(β, Σ | Y) = p(β | Σ, Y) p(Σ | Y)`,
its two blocks keyed like a draw:

    β:  vec(β) | Σ, Y  ~  N(vec(β̂), Σ ⊗ (X'X + Ω⁻¹)⁻¹)
    Σ:  Σ | Y          ~  IW(df, ε'ε + (β̂ − β₀)' Ω⁻¹ (β̂ − β₀) + S)

built by [`normal_coefficient_posterior`](@ref) and [`var_covariance_posterior`](@ref),
with `β̂` from [`normal_coefficient_posterior_mean`](@ref) and `ε = Y − X β̂`. The
arguments are those of [`sample_var_params`](@ref) on prepared data (`Y`, `X` from
`prepare_var_data`).

The coefficient covariance depends on `Σ`, so the `β` block is the conditional posterior
as a function of it, sharing `β̂` with the `Σ` block. A joint draw takes `Σ` from `post.Σ`
first and the coefficients from `post.β(Σ)` at that draw, as [`sample_var_params`](@ref)
does; `logpdf(post.β(Σ), vec(β)) + logpdf(post.Σ, Σ)` is the joint NIW log density (up to
the jitter of [`kron_cholesky_factor`](@ref)).
"""
function normal_inverse_wishart_posterior(Y, X, β_prior_μ, Ω_inv, S, df)

    β_hat = normal_coefficient_posterior_mean(Y, X, β_prior_μ, Ω_inv)

    return (β = Σ -> normal_coefficient_posterior(β_hat, X, Σ, Ω_inv),
            Σ = var_covariance_posterior(Y, X, β_hat, df, S, β_prior_μ, Ω_inv))

end


"""
    sample_var_params(data,p, β_mean, Ω_inv)

    data: observations
    p: number of lags
    β_priormean: prior mean of beta coefficients
    Ω_inv: inversion prior variance of beta coefficients
    S: prior covariance scale
    df: posterior covariance distribution degrees of freedom
"""
function sample_var_params(data, p, β_prior_μ, Ω_inv, S, df; max_draws::Int = 100)

    Y, X = prepare_var_data(data, p)
    n = size(Y, 2)

    posterior = normal_inverse_wishart_posterior(Y, X, β_prior_μ, Ω_inv, S, df)

    # Σ from its marginal posterior, then β conditional on that draw.
    Σ = rand(posterior.Σ)

    # Σ and X are fixed across rejection draws, so the coefficient posterior — and the
    # factor of Σ ⊗ (X'X + Ω⁻¹)⁻¹ it carries — is built once and reused.
    β_posterior = posterior.β(Σ)
    β = rand(β_posterior)

    # Companion bottom block A = B' (n × n*p) in oldest-lag-first ordering.
    var_coeff(β) = collect(reshape(β, n * p, n)')

    draws = 1
    while !is_stationary(var_coeff(β), n, p) && draws < max_draws

        β = rand(β_posterior)
        draws += 1
    end

    return β, Σ

end
