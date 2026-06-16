
function prepare_var_data(Y::Matrix{Float64}, p::Int, X::Union{Matrix{Float64},Vector{Float64}} = Matrix{Float64}(undef, 0, 0), add_intercept::Bool = false)
    T, n = size(Y)
    Y_lagged = zeros(T - p, n * p)
    for t in (p + 1):T
        Y_lagged[t - p, :] = Y[t-p:t-1, :]'
    end

    predictors = Y_lagged

    if !isempty(X)
        if size(X, 1) != T
            error("The number of rows in X must be equal to the number of rows in Y.")
        end
        X_subset = X[p+1:end, :]
        predictors = hcat(predictors, X_subset)
    end

    if add_intercept
        intercept = ones(T - p, 1)
        predictors = hcat(intercept, predictors)
    end

    return Y[p+1:end, :], predictors
end


posterior_beta_coefficient_mean(Y, X, beta_mean, Ω_inv) = inv(X'X + Ω_inv)*(X'Y + Ω_inv*beta_mean)


#posterior_beta_coefficient_var(X, Σ, Ω_inv) = kron(Σ, inv(X'X + Ω_inv)) do usuniecia

#Ω_inv prior of beta coefficient variance

function draw_beta(X, Σ, beta_posterior, Ω_inv)

    n = size(Σ, 1)      # number of equations
    k = size(X, 2)      # number of predictors (n * p)
    m = n * k           # length of vec(β)

    beta_var = kron(Σ, inv(X'X + Ω_inv)) + I(m) * 1e-5

    F = eigen(Hermitian(beta_var))
    λ = max.(F.values, 0.0)
    L = F.vectors * Diagonal(sqrt.(λ))

    return vec(beta_posterior) + L * randn(m)

end


function covariance_posterior_dist(Y, X, β_posterior_μ, posterior_df, variance_prior, β_prior_μ, Ω_inv)

    ε =  Y - X * β_posterior_μ

    β_diff = β_posterior_μ - β_prior_μ

    S = ε' * ε + β_diff' * Ω_inv * β_diff  + variance_prior

    return InverseWishart(posterior_df, collect(Hermitian(S)))

end



"""
    is_stationary(var_coeff, n, p)

Check VAR(p) stationarity via the companion matrix eigenvalues.
`var_coeff` is the n × n*p companion bottom block in the oldest-lag-first ordering
used by [`tc_var`](@ref) (i.e. `B'` for the regression `Y = X·B`).
Returns true if all eigenvalues of the companion matrix have modulus < 1.
"""
function is_stationary(var_coeff::AbstractMatrix, n::Int, p::Int)
    if p == 1
        return all(abs.(eigvals(var_coeff)) .< 1.0)
    end
    companion = vcat(
        hcat(zeros(n * (p - 1), n), I(n * (p - 1))),
        var_coeff)
    return all(abs.(eigvals(companion)) .< 1.0)
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

    β = draw_beta(X, Σ, β_hat, Ω_inv)

    # Companion bottom block A = B' (n × n*p) in oldest-lag-first ordering.
    var_coeff(β) = collect(reshape(β, n * p, n)')

    draws = 1
    while !is_stationary(var_coeff(β), n, p) && draws < max_draws
        β = draw_beta(X, Σ, β_hat, Ω_inv)
        draws += 1
    end

    return β, Σ

end
  
 