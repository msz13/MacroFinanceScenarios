posterior_beta_coefficient_mean(Y, X, beta_mean, Ω_inv) = inv(X'X + Ω_inv)*(X'Y + Ω_inv*beta_mean)


#posterior_beta_coefficient_var(X, Σ, Ω_inv) = kron(Σ, inv(X'X + Ω_inv)) do usuniecia

#Ω_inv prior of beta coefficient variance

function covariance_posterior_dist(Y, X, β_posterior_μ, posterior_df, variance_prior, β_prior_μ, Ω_inv)

    ε =  Y - X * β_posterior_μ

    β_diff = β_posterior_μ - β_prior_μ

    S = ε' * ε + β_diff' * Ω_inv * β_diff  + variance_prior

    return InverseWishart(posterior_df, collect(Hermitian(S)))

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

    β_hat = posterior_beta_coefficient_mean(Y, X, β_prior_μ, Ω_inv)

    Σ = rand(covariance_posterior_dist(Y, X, β_hat, df, S, β_prior_μ, Ω_inv))

    # Σ and X are fixed across rejection draws, so factor the proposal covariance
    # once and reuse it.
    L = beta_cholesky_factor(X, Σ, Ω_inv)
    β = draw_beta(β_hat, L)

    # Companion bottom block A = B' (n × n*p) in oldest-lag-first ordering.
    var_coeff(β) = collect(reshape(β, n * p, n)')

    draws = 1
    while !is_stationary(var_coeff(β), n, p) && draws < max_draws

        β = draw_beta(β_hat, L)
        draws += 1
    end

    return β, Σ

end
