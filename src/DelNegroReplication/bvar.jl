using LinearAlgebra
using Random

"""
    bvar(y, lags, b, PSI, lambda, draw) -> (beta, sigma)

Bayesian VAR with Minnesota prior.

Arguments:
  - `y`      : (T+lags) × n data matrix (full history including pre-sample)
  - `lags`   : number of VAR lags
  - `b`      : n*lags × n prior mean for coefficients (typically zeros)
  - `PSI`    : length-n vector of residual variance scale (equation-specific)
  - `lambda` : overall prior tightness (Minnesota λ, e.g. 0.2)
  - `draw`   : if true draw from posterior, else return posterior mode

Returns:
  - `beta`  : n*lags × n coefficient matrix
  - `sigma` : n × n residual covariance matrix
"""
function bvar(
    y      :: AbstractMatrix,
    lags   :: Int,
    b      :: AbstractMatrix,
    PSI    :: AbstractVector,
    lambda :: Float64,
    draw   :: Bool,
)
    TT, n = size(y)
    k     = n * lags          # total regressors per equation

    # Build regressor matrix X (lag matrix)
    X_full = zeros(TT, k)
    for i in 1:lags
        X_full[:, (i-1)*n+1:i*n] = lag_matrix(y, i)
    end
    X = X_full[lags+1:end, :]
    Y = y[lags+1:end, :]
    T = size(Y, 1)

    alpha = 2.0
    d     = n + 2      # degrees of freedom for covariance prior

    # Minnesota prior variance (diagonal Ω)
    omega = zeros(k)
    for i in 1:lags
        omega[(i-1)*n+1:i*n] = (d - n - 1) * (lambda^2) * (1 / i^alpha) ./ PSI
    end
    Omega_inv = Diagonal(1.0 ./ omega)

    # Posterior mode for beta
    betahat = (X'X + Omega_inv) \ (X'Y + Omega_inv * b)

    # Posterior mode for sigma
    epshat   = Y - X * betahat
    sigmahat = (epshat'epshat + Diagonal(PSI) +
                (betahat - b)' * Omega_inv * (betahat - b)) / (T + d + n + 1)

    if !draw
        return betahat, sigmahat
    end

    # Draw from posterior (rejection-sampled for stationarity)
    beta  = copy(betahat)
    sigma = copy(sigmahat)
    stationary = false

    while !stationary
        # Draw sigma ~ IW posterior
        F    = eigen(Symmetric(sigmahat * (T + d + n + 1)))
        # L s.t. rows of randn * L' ~ N(0, S^{-1}) where S = sigmahat*(T+d+n+1)
        L    = F.vectors * Diagonal(1.0 ./ sqrt.(abs.(F.values)))
        eta  = randn(T + d, n) * L'
        sigma = inv(eta' * eta)

        # Draw beta | sigma ~ MatrixNormal
        cholSIGMA = cholred((sigma + sigma') / 2)
        cholZZinv = cholred(inv(X'X + Omega_inv))
        beta = betahat + cholZZinv' * randn(k, n) * cholSIGMA

        # Check stationarity of implied companion VAR
        AA_comp = zeros(n * lags, n * lags)
        AA_comp[1:n, 1:n*lags] = beta'
        if lags > 1
            AA_comp[n+1:end, 1:n*(lags-1)] = I(n * (lags - 1))
        end
        stationary = all(abs.(eigvals(AA_comp)) .<= 1.0)
    end

    return beta, sigma
end
