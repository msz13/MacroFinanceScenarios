using LinearAlgebra

"""
    cholred(S; tol=1e-12) -> Matrix

Reduced matrix square root via eigendecomposition.
Returns C (upper-triangular-like) such that C' * C ≈ S,
clipping negative eigenvalues to zero.

Usage: `mu + cholred(S)' * randn(n)` draws from N(mu, S).
"""
function cholred(S::AbstractMatrix; tol::Float64 = 1e-12)
    F = eigen(Symmetric((S + S') / 2))
    dd = max.(F.values, 0.0)
    return Diagonal(sqrt.(dd)) * F.vectors'
end

"""
    lag_matrix(y, n) -> Matrix

Return y shifted down by n rows, padded with zeros at the top.
Matches MATLAB's lag(y, n) with default v=0.
"""
function lag_matrix(y::AbstractMatrix, n::Int)
    T, k = size(y)
    z = zeros(T, k)
    if n >= 1 && n < T
        z[n+1:end, :] = y[1:end-n, :]
    end
    return z
end
