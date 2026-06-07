using LinearAlgebra

"""
Result type returned by `kalman_filter`.

Fields use MATLAB KF.m naming:
  - `loglik` : scalar log-likelihood (innovations form)
  - `S`  : T × ns  filtered states  x_{t|t}
  - `P`  : ns × ns × T  filtered covariances P_{t|t}
  - `S0` : ns prior mean (passed through unchanged)
  - `P0` : ns × ns prior covariance (passed through)
  - `Sf` : T × ns  one-step-ahead predicted states  x_{t|t-1}
  - `Pf` : ns × ns × T  predicted covariances P_{t|t-1}
  - `A`, `Q`, `C`, `R` : model matrices (stored for downstream use)
"""
struct KFResult
    loglik :: Float64
    S  :: Matrix{Float64}
    P  :: Array{Float64,3}
    S0 :: Vector{Float64}
    P0 :: Matrix{Float64}
    Sf :: Matrix{Float64}
    Pf :: Array{Float64,3}
    A  :: Matrix{Float64}
    Q  :: Matrix{Float64}
    C  :: Matrix{Float64}
    R  :: Matrix{Float64}
end

"""
    kalman_filter(y, C, R, A, Q, S0, P0) -> KFResult

Kalman filter for:
    s_t  = A * s_{t-1} + w_t,  w_t ~ N(0, Q)
    y_t  = C * s_t    + v_t,  v_t ~ N(0, R)

Arguments:
  - `y`  : T × n observation matrix (NaN entries treated as missing)
  - `C`  : n × ns observation matrix
  - `R`  : n × n observation-noise covariance
  - `A`  : ns × ns state-transition matrix
  - `Q`  : ns × ns process-noise covariance
  - `S0` : ns prior mean for x_{1|0}
  - `P0` : ns × ns prior covariance for P_{1|0}

Log-likelihood uses the innovations decomposition.  The normalisation
constant matches MATLAB KF.m exactly (-0.5 * 2π per observation step).
"""
function kalman_filter(
    y  :: AbstractMatrix,
    C  :: AbstractMatrix,
    R  :: AbstractMatrix,
    A  :: AbstractMatrix,
    Q  :: AbstractMatrix,
    S0 :: AbstractVector,
    P0 :: AbstractMatrix,
)
    T, _n = size(y)
    ns = size(C, 2)

    S  = fill(NaN, T, ns)
    P  = fill(NaN, ns, ns, T)
    Sf = fill(NaN, T, ns)
    Pf = fill(NaN, ns, ns, T)

    Sprev  = Vector{Float64}(S0)
    Pprev  = Matrix{Float64}(P0)
    loglik = 0.0

    for t in 1:T
        # Predict
        Sft = A * Sprev
        Pft = A * Pprev * A' + Q

        # Select non-missing observations
        yt      = y[t, :]
        obs_idx = .!isnan.(yt)
        yt_obs  = yt[obs_idx]
        Ct      = C[obs_idx, :]
        Rt      = R[obs_idx, obs_idx]

        # Update
        yf   = Ct * Sft
        innov = yt_obs - yf
        iV   = inv(Ct * Pft * Ct' + Rt)
        Gain = Pft * Ct' * iV
        St   = Sft + Gain * innov
        Pt   = Pft - Gain * Ct * Pft

        # Log-likelihood contribution (faithful to MATLAB: uses -0.5*(2π) as constant)
        loglik += 0.5 * log(det(iV)) - 0.5 * innov' * iV * innov - 0.5 * (2π)

        S[t, :]      = St
        P[:, :, t]   = Pt
        Sf[t, :]     = Sft
        Pf[:, :, t]  = Pft

        Sprev = St
        Pprev = Pt
    end

    return KFResult(loglik, S, P, S0, P0, Sf, Pf,
                    Matrix{Float64}(A), Matrix{Float64}(Q),
                    Matrix{Float64}(C), Matrix{Float64}(R))
end
