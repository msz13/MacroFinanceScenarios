# Model-agnostic posterior distributions and the draws taken from them.

function covariance_posterior(data, scale_prior, d_posterior)

    res = diff(data, dims=1)
    posterior_mean = res' * res .+ scale_prior

    return InverseWishart(d_posterior, posterior_mean)

end

"""
    beta_cholesky_factor(X, Σ, Ω_inv) -> L

Lower-triangular Cholesky factor `L` of the coefficient posterior covariance
`Σ ⊗ (X'X + Ω_inv)⁻¹` (so `L * L' ≈` that covariance).

Uses the identity that the Cholesky factor of a Kronecker product is the
Kronecker product of the factors:

    chol(A ⊗ B) = chol(A) ⊗ chol(B)

so the `m × m` factor (`m = n·k`) is assembled from the small `n × n` and
`k × k` blocks instead of decomposing the full matrix. Because this factor does
not depend on the proposed `β`, it is computed once and reused across the
stationarity-rejection draws in [`sample_var_params`](@ref).
"""
function beta_cholesky_factor(X, Σ, Ω_inv)
    jitter = 1e-5
    Σ_L = cholesky(Symmetric(Σ) + jitter * I).L                 # n × n
    V_L = cholesky(Symmetric(inv(Symmetric(X'X + Ω_inv))) + jitter * I).L  # k × k
    return kron(Σ_L, V_L)
end

"""
    draw_beta(beta_posterior, L) -> vec(β)

Draw `vec(β) ~ N(vec(beta_posterior), L L')` from the precomputed Cholesky
factor `L` returned by [`beta_cholesky_factor`](@ref).
"""
draw_beta(beta_posterior, L) = vec(beta_posterior) + L * randn(size(L, 1))
