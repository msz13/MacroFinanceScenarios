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

"""
    psd_factor(A) -> Matrix

Lower-triangular (or, in the singular case, merely square) factor `L` with
`L*L' ≈ A` for a symmetric positive-*semi*definite `A`, so that
`L * randn(n)` draws from `N(0, A)`.

Unlike [`chol_psd`](@ref) this adds no jitter: a rank-deficient `A` keeps its exact
null space, and `A = 0` gives `L = 0` (a degenerate, deterministic "draw"). That is
what a *simulator* needs — jitter turns a deliberately switched-off noise block into
draws with a standard deviation of `sqrt(sqrt(eps())) ≈ 1e-4`, which is the whole
point of switching it off. [`chol_psd`](@ref) stays the right tool inside the filter,
where the jitter guards a linear solve.

The Cholesky factor is used when it exists; otherwise the symmetric eigendecomposition
gives `L = V √Λ`, with negative eigenvalues (rounding around a zero) clamped to zero.
"""
function psd_factor(A::AbstractMatrix)
    A_sym = Symmetric((A + A') / 2)
    F = cholesky(A_sym; check = false)
    issuccess(F) && return Matrix(F.L)
    values, vectors = eigen(A_sym)
    return vectors * Diagonal(sqrt.(max.(values, 0.0)))
end
