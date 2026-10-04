# Draw of the VAR parameters inside the Gibbs sampler: the posterior itself is
# `NaturalConjugate` (var/natural_conjugate.jl); what stays here is the
# stationarity-rejection loop.

"""
    sample_var_params(data, β_prior, sigma_prior; max_draws = 100)

Draw `(β, Σ)` from the [`NaturalConjugate`](@ref) posterior of a VAR on
`data`, rejecting non-stationary draws.

    data: observations, including the `p` pre-sample rows
    β_prior: MinnesotaPrior on the coefficients; the lag order `p` is read off it
    sigma_prior: InverseWishart prior on the innovation covariance

`(β, Σ)` is drawn jointly and the pair is redrawn until `β` is stationary or
`max_draws` is reached (the last draw is returned either way).
"""
function sample_var_params(data, β_prior::MinnesotaPrior, sigma_prior::InverseWishart; max_draws::Int = 100)

    p = β_prior.p
    Y, X = prepare_var_data(data, p)
    n = size(Y, 2)

    posterior = NaturalConjugate(Y, X, β_prior, sigma_prior)

    β, Σ = rand(posterior)

    draws = 1
    while !is_stationary(var_coeff(β, n, p), n, p) && draws < max_draws

        β, Σ = rand(posterior)
        draws += 1
    end

    return β, Σ

end
