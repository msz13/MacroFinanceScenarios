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

"""
    lyapunov_covariance(F, Q) -> Matrix

Stationary covariance of the VAR(1) `x_t = F x_{t-1} + u_t`, `u_t ~ N(0, Q)`: the
solution of the discrete Lyapunov equation `P = F P F' + Q`, obtained in vectorised
form as `vec(P) = (I − F⊗F)⁻¹ vec(Q)`.

`F` must be stable — every eigenvalue inside the unit circle — otherwise `I − F⊗F`
is singular and the solve fails or returns garbage. Callers that can be handed a
unit root (a random-walk trend block, a random-walk log-volatility) check
[`is_stationary`](@ref) first and supply a diffuse covariance instead.
"""
function lyapunov_covariance(F::AbstractMatrix, Q::AbstractMatrix)
    n = size(F, 1)
    return reshape((I - kron(F, F)) \ vec(Q), n, n)
end
