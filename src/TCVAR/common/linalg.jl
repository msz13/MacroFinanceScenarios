using LinearAlgebra
using Random

"""
    chol_psd(A) -> Cholesky

Cholesky factorization of the symmetric, positive-semidefinite matrix `A`, used
in place of `pinv` for the Kalman/Carter-Kohn linear solves. `A` is symmetrized
first; if the bare factorization fails (the observation noise is `H = eps()*I`,
which leaves the innovation covariance numerically near-singular) a small
trace-scaled jitter is added so the factorization stays robust.

Use `A_inv_solve = B / chol_psd(A)` for `B*A⁻¹` and `chol_psd(A).L` as a matrix
square root for sampling `mean + chol_psd(cov).L * randn(n)`.
"""
function chol_psd(A::AbstractMatrix)
    A_sym = Symmetric((A + A') / 2)
    F = cholesky(A_sym; check = false)
    issuccess(F) && return F
    jitter = sqrt(eps()) * (tr(A_sym) / size(A_sym, 1) + 1.0)
    return cholesky(A_sym + jitter * I; check = false)
end

"""Draw one sample from N(`mean`, `cov`) via a robust Cholesky factor (see
[`chol_psd`](@ref))."""
sample_mvn(mean::AbstractVector, cov::AbstractMatrix) =
    mean + chol_psd(cov).L * randn(length(mean))
