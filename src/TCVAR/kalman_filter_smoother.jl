using LinearAlgebra
using Random

"""
    kalman_filter(Y, A, H, Q, R, x0, P0) -> NamedTuple

Kalman filter for the linear-Gaussian state-space model

    xₜ = A xₜ₋₁ + wₜ,   wₜ ~ N(0, Q)        (state, dimension n)
    yₜ = H xₜ   + vₜ,   vₜ ~ N(0, R)        (observation, dimension d)

Uses the **Joseph form** for the covariance update, so the filtered
covariances stay symmetric and positive semidefinite even with roundoff
or a suboptimal gain.

Arguments
- `Y`  :: d × T matrix — each column is one observation yₜ
- `A`  :: n × n state-transition matrix
- `H`  :: d × n observation matrix
- `Q`  :: n × n process-noise covariance
- `R`  :: d × d observation-noise covariance
- `x0` :: length-n prior mean for the first state, x_{1|0}
- `P0` :: n × n prior covariance for the first state, P_{1|0}

Returns a NamedTuple with
- `loglik` :: scalar log-likelihood (innovations decomposition)
- `xp`     :: n × T predicted means      x_{t|t-1}
- `Pp`     :: n × n × T predicted covs    P_{t|t-1}
- `xf`     :: n × T filtered means        x_{t|t}
- `Pf`     :: n × n × T filtered covs     P_{t|t}
"""
function kalman_filter(Y::AbstractMatrix, A, H, Q, R, x0, P0)
    d, T = size(Y)
    n    = length(x0)

    xp = zeros(n, T);    Pp = zeros(n, n, T)     # predicted  x_{t|t-1}, P_{t|t-1}
    xf = zeros(n, T);    Pf = zeros(n, n, T)     # filtered   x_{t|t},   P_{t|t}

    x  = Vector{Float64}(x0)
    P  = Matrix{Float64}(P0)
    ll = 0.0
    log2pi = log(2π)

    for t in 1:T
        # ---- Predict (t = 1 uses the prior x0, P0 directly) ----
        if t > 1
            x = A * x
            P = A * P * A' + Q
            P = (P + P') / 2
        end
        xp[:, t]    = x
        Pp[:, :, t] = P

        # ---- Innovation + log-likelihood contribution ----
        e = Y[:, t] .- H * x
        S = Symmetric(H * P * H' + R)
        C = cholesky(S)                          # S = L Lᵀ
        z = C.L \ e                              # zᵀz = eᵀ S⁻¹ e
        ll += -0.5 * (d * log2pi + logdet(C) + dot(z, z))

        # ---- Update (Joseph form) ----
        K = (P * H') / C                         # Kalman gain  P Hᵀ S⁻¹
        x = x + K * e
        M = I - K * H
        P = M * P * M' + K * R * K'              # symmetric & PSD by construction
        P = (P + P') / 2                         # harmless tidy-up

        xf[:, t]    = x
        Pf[:, :, t] = P
    end

    return (loglik = ll, xp = xp, Pp = Pp, xf = xf, Pf = Pf)
end

"""Convenience wrapper returning only the log-likelihood."""
kalman_loglik(Y, A, H, Q, R, x0, P0) = kalman_filter(Y, A, H, Q, R, x0, P0).loglik

# Robust draw from N(mean, cov): symmetrize, Cholesky, fall back to jitter.
function _sample_mvn(rng, mean::AbstractVector, cov::AbstractMatrix)
    Csym = Symmetric((cov + cov') / 2)
    F = cholesky(Csym; check = false)
    L = issuccess(F) ? F.L : cholesky(Csym + sqrt(eps()) * I).LS
    return mean .+ L * randn(rng, length(mean))
end

"""
    carter_kohn(filt, A; rng = Random.default_rng()) -> Matrix

Carter & Kohn (1994) forward-filter / backward-sample (FFBS) smoother.
Given the output `filt` of [`kalman_filter`](@ref), draws one trajectory
from the joint posterior p(x_{1:T} | y_{1:T}).

Backward recursion, for t = T-1, …, 1, using the smoothing gain
    Jₜ = P_{t|t} Aᵀ P_{t+1|t}⁻¹
    mean = x_{t|t} + Jₜ (x̃_{t+1} − x_{t+1|t})
    cov  = P_{t|t} − Jₜ P_{t+1|t} Jₜᵀ
and sampling x̃ₜ ~ N(mean, cov), seeded by x̃_T ~ N(x_{T|T}, P_{T|T}).

Returns an n × T matrix whose columns are the sampled states. Call it
repeatedly inside a Gibbs sampler to draw the latent path each sweep;
averaging many draws approximates the RTS smoothed mean.
"""
function carter_kohn(filt, A; rng = Random.default_rng())
    xf, Pf, xp, Pp = filt.xf, filt.Pf, filt.xp, filt.Pp
    n, T = size(xf)
    X = zeros(n, T)

    # ---- terminal draw ----
    X[:, T] = _sample_mvn(rng, xf[:, T], Pf[:, :, T])

    # ---- backward sampling ----
    for t in (T - 1):-1:1
        Cp = cholesky(Symmetric(Pp[:, :, t + 1]))      # P_{t+1|t}
        J  = (Pf[:, :, t] * A') / Cp                   # smoothing gain
        m  = xf[:, t] + J * (X[:, t + 1] - xp[:, t + 1])
        Σ  = Pf[:, :, t] - J * Pp[:, :, t + 1] * J'
        X[:, t] = _sample_mvn(rng, m, Σ)
    end

    return X
end

# ----------------------------------------------------------------------
# Demo: constant-velocity model (state = [position, velocity], observe position)
# ----------------------------------------------------------------------
function simulate(A, H, Q, R, x0, T; rng = Random.default_rng())
    n, d = length(x0), size(H, 1)
    Lq, Lr = cholesky(Symmetric(Q)).L, cholesky(Symmetric(R)).L
    x = Vector{Float64}(x0)
    Y = zeros(d, T)
    for t in 1:T
        x = A * x + Lq * randn(rng, n)
        Y[:, t] = H * x + Lr * randn(rng, d)
    end
    return Y
end

if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(1)
    dt = 1.0
    A  = [1.0 dt; 0.0 1.0]          # position-velocity transition
    H  = [1.0 0.0]                  # observe position only
    Q  = [1e-3 0.0; 0.0 1e-3]
    R  = reshape([0.1], 1, 1)
    x0 = [0.0, 1.0]
    P0 = Matrix(1.0I, 2, 2)

    Y    = simulate(A, H, Q, R, x0, 50)
    filt = kalman_filter(Y, A, H, Q, R, x0, P0)
    println("log-likelihood = ", filt.loglik)

    # one posterior draw of the entire latent path
    draw = carter_kohn(filt, A)
    println("sampled state x₁ = ", draw[:, 1])

    # averaging many FFBS draws ≈ RTS smoothed mean; compare to the filtered mean
    Ndraws = 2000
    acc = zeros(size(filt.xf))
    for _ in 1:Ndraws
        acc .+= carter_kohn(filt, A)
    end
    smooth_mean = acc ./ Ndraws
    println("filtered  x₁ = ", filt.xf[:, 1])
    println("smoothed≈ x₁ = ", smooth_mean[:, 1])
end
