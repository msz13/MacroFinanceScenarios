# Draw of the VAR parameters inside the Gibbs sampler: the posterior itself is
# `NaturalConjugate` (var/natural_conjugate.jl); what stays here is the
# stationarity-rejection loop.

"""
    sample_var_params(data, p, β_prior_μ, Ω_inv, sigma_prior; max_draws = 100)

Draw `(β, Σ)` from the [`NaturalConjugate`](@ref) posterior of a `p`-lag VAR on
`data`, rejecting non-stationary coefficient draws.

    data: observations, including the `p` pre-sample rows
    p: number of lags
    β_prior_μ: k × n prior mean of the coefficients
    Ω_inv: k × k prior row precision of the coefficients
    sigma_prior: InverseWishart prior on the innovation covariance

`Σ` is drawn once; `β` is redrawn from its conditional given that `Σ` until it is
stationary or `max_draws` is reached (the last draw is returned either way).
"""
function sample_var_params(data, p, β_prior_μ, Ω_inv, sigma_prior::InverseWishart; max_draws::Int = 100)

    Y, X = prepare_var_data(data, p)
    n = size(Y, 2)

    posterior = NaturalConjugate(Y, X, β_prior_μ, Ω_inv, sigma_prior)

    Σ = rand(covariance_posterior(posterior))

    # Σ and X are fixed across rejection draws, so factor the proposal covariance
    # Σ ⊗ (X'X + Ω⁻¹)⁻¹ once and reuse it.
    L = coefficient_factor(posterior, Σ)
    β = rand_coefficients(posterior, Σ, L)

    # Companion bottom block A = B' (n × n*p) in oldest-lag-first ordering.
    var_coeff(β) = collect(reshape(β, n * p, n)')

    draws = 1
    while !is_stationary(var_coeff(β), n, p) && draws < max_draws

        β = rand_coefficients(posterior, Σ, L)
        draws += 1
    end

    return β, Σ

end
