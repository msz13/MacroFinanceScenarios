using LinearAlgebra
using Random

"""
Result type returned by `carter_kohn`.

  - `S`  : T × ns  drawn state trajectory
  - `S0` : ns      drawn initial state (period 0)
"""
struct KCResult
    S  :: Matrix{Float64}
    S0 :: Vector{Float64}
end

"""
    carter_kohn(kf::KFResult) -> KCResult

Carter–Kohn (1994) backward simulation smoother.
Draws a joint trajectory {s_0, s_1, …, s_T} from the smoothing
distribution p(s_{0:T} | y_{1:T}).

The algorithm walks backward from t = T, using the filtered
means/covariances stored in `kf` together with the predicted
covariances to compute the backward conditional distribution at
each step.
"""
function carter_kohn(kf::KFResult)
    T, ns = size(kf.S)
    A     = kf.A

    drS = fill(NaN, T, ns)

    # Terminal draw: s_T | y_{1:T}
    drS[T, :] = kf.S[T, :] + cholred(kf.P[:, :, T])' * randn(ns)

    # Backward pass: t = T-1 down to 1
    for t in T-1:-1:1
        iPf   = pinv(kf.Pf[:, :, t+1])
        Pt    = kf.P[:, :, t]
        mu    = kf.S[t, :] + Pt * A' * iPf * (drS[t+1, :] - kf.Sf[t+1, :])
        sigma = Pt - Pt * A' * iPf * A * Pt
        drS[t, :] = mu + cholred(sigma)' * randn(ns)
    end

    # Initial state draw: s_0 | s_1, y_{1:T}
    iPf   = inv(kf.Pf[:, :, 1])
    mu    = kf.S0 + kf.P0 * A' * iPf * (drS[1, :] - kf.Sf[1, :])
    sigma = kf.P0 - kf.P0 * A' * iPf * A * kf.P0
    drS0  = mu + cholred(sigma)' * randn(ns)

    return KCResult(drS, drS0)
end
