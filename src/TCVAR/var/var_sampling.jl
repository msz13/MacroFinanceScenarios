# Normal–inverse-Wishart conditional posterior for a VAR, and the draw taken from it.
# The generic pieces (the IW constructor, the conjugate coefficient mean, the
# Kronecker factor and the draw) live in `common/posteriors.jl`; what is VAR-specific
# and stays here is the NIW *scale* and the stationarity-rejection loop.

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

    β_hat = normal_coefficient_posterior_mean(Y, X, β_prior_μ, Ω_inv)

    Σ = rand(var_covariance_posterior(Y, X, β_hat, df, S, β_prior_μ, Ω_inv))

    # Σ and X are fixed across rejection draws, so factor the proposal covariance
    # Σ ⊗ (X'X + Ω⁻¹)⁻¹ once and reuse it.
    L = kron_cholesky_factor(Σ, inv(Symmetric(X'X + Ω_inv)))
    β = draw_from_factor(β_hat, L)

    # Companion bottom block A = B' (n × n*p) in oldest-lag-first ordering.
    var_coeff(β) = collect(reshape(β, n * p, n)')

    draws = 1
    while !is_stationary(var_coeff(β), n, p) && draws < max_draws

        β = draw_from_factor(β_hat, L)
        draws += 1
    end

    return β, Σ

end
