using LinearAlgebra
using Random

"""
    covariance_draw(z, df0, mS0) -> Matrix

Draw from the posterior Inverse-Wishart distribution for the
trend innovation covariance.

The prior mode is `mS0` (n × n matrix or scalar times identity)
with `df0` prior degrees of freedom.  Posterior is:
    Sigma ~ IW(z'z + Sc0,  T + df0)
where Sc0 = mS0 * (df0 + n + 1) converts the modal prior to the
scale matrix convention.

Arguments:
  - `z`   : T × n matrix of first-differences of trend draws
  - `df0` : prior degrees of freedom
  - `mS0` : n × n prior mode for Sigma (typically diagonal)
"""
function covariance_draw(z::AbstractMatrix, df0::Int, mS0::AbstractMatrix)
    n   = size(z, 2)
    T   = size(z, 1)
    Sc0 = mS0 * (df0 + n + 1)   # modal prior → scale matrix
    S   = z' * z + Sc0

    F = eigen(Symmetric(S))
    # L s.t. rows of randn * L' ~ N(0, S^{-1})
    L = F.vectors * Diagonal(1.0 ./ sqrt.(abs.(F.values)))
    eta = randn(T + df0, n) * L'
    return inv(eta' * eta)
end

"""
    simulate_inv_wishart_prior(SC0tr, df0, n_sim) -> Array{Float64,3}

Simulate `n_sim` draws from the prior Inverse-Wishart for the
trend covariance (used to visualise prior vs posterior).

Returns an n × n × n_sim array.
"""
function simulate_inv_wishart_prior(SC0tr::AbstractVector, df0::Int, n_sim::Int)
    n    = length(SC0tr)
    sims = Array{Float64,3}(undef, n, n, n_sim)

    S = Diagonal(SC0tr) * (df0 + n + 1)   # scale matrix (diagonal)
    F = eigen(Symmetric(Matrix(S)))
    L = F.vectors * Diagonal(1.0 ./ sqrt.(abs.(F.values)))

    for jm in 1:n_sim
        eta = randn(df0, n) * L'
        sims[:, :, jm] = inv(eta' * eta)
        if mod(jm, 100_000) == 0
            @info "SimulInvWishart: $jm / $n_sim"
        end
    end
    return sims
end
