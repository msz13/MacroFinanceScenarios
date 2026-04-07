"""
    DelleMonacheModel

Score-driven state-space model of the price-dividend ratio and long-run stock returns.

Reference:
  Delle Monache, Petrella & Venditti (2020/2021)
  "Price Dividend Ratio and Long-Run Stock Returns: A Score-Driven State Space Model"
  ECB Working Paper No. 2369 / Journal of Business & Economic Statistics, 39(4), 1054–1065

Model structure
───────────────
Observed variables:  y_t = (r_t, Δd_t, pd_t)'
State vector:        α_t = (1, g̃_t, μ̃_t, g̃_{t-1}, ε_{d,t}, ε_{g,t}, ε_{μ,t})'
Time-varying params: f_t = (ḡ_t, μ̄_t, σ̄_{d,t}, σ̄_{g,t}, σ̄_{μ,t}, ρ̄_{d,μ,t}, ρ̄_{g,μ,t})'

Key dynamics
────────────
Permanent components (score-driven martingales):
    ḡ_t   = ḡ_{t-1} + σ_ḡ · η_ḡ       long-run dividend growth
    μ̄_t   = μ̄_{t-1} + σ_μ̄ · η_μ̄       long-run expected return

Transitory components:
    g̃_t   = g̃_{t-1} + ε_{g,t}           random walk
    μ̃_t   = ρ_μ · μ̃_{t-1} + ε_{μ,t}    AR(1)

Dividend growth (expected + unexpected):
    Δd_t  = (ḡ_{t-1} + g̃_{t-1}) + ε_{d,t} + ε_{g,t}

Log price-dividend ratio (Campbell-Shiller present-value):
    pd_t  = κ + (ḡ_t + g̃_t − μ̄_t)/(1−ρ) − μ̃_t/(1−ρ·ρ_μ)

Log return (Campbell-Shiller identity):
    r_t   = Δd_t + log(1 + exp(pd_t)) − pd_{t-1}

Parameter calibration
─────────────────────
Parameters are fixed at values calibrated to quarterly US equity data (CRSP, 1926–2018),
consistent with the paper's ML estimates. Estimation algorithm is NOT implemented.

This module provides Monte Carlo simulation only.
"""
module DelleMonacheModel

using Random, Distributions, LinearAlgebra

export ModelParams, ModelState, SimulationResult, default_params,
       calibrate_kappa, simulate_path, extract_at_horizon,
       annualised_cumulative_return

# ─────────────────────────────────────────────
# 1. Parameter Struct
# ─────────────────────────────────────────────

"""
    ModelParams

All model parameters, calibrated to quarterly US equity data.
Consistent with the estimated values from Delle Monache et al. (JBES, 2021).

Parameters with (*) are held fixed for MC simulation; in the paper they are
estimated jointly via score-driven ML together with A and B matrices.
"""
Base.@kwdef struct ModelParams
    # ── Campbell-Shiller log-linearisation ──
    ρ::Float64    = 0.97            # quarterly constant ≈ exp(pd̄)/(1+exp(pd̄))

    # ── Transitory expected return: AR(1) ──
    ρ_μ::Float64  = 0.90            # quarterly persistence

    # ── Initial (estimated) permanent steady-state values ──
    ḡ₀::Float64   = 0.00375         # quarterly trend growth (1.5% pa)
    μ̄₀::Float64   = 0.0175          # quarterly long-run expected return (7% pa)

    # ── Structural shock standard deviations (*) ──
    σ_d::Float64   = 0.055          # dividend growth shock (idiosyncratic)
    σ_g::Float64   = 0.0005         # trend growth shock
    σ_μ::Float64   = 0.012          # expected return shock

    # ── Shock correlations (*) ──
    ρ_dμ::Float64  = -0.30          # dividend–return shock correlation
    ρ_gμ::Float64  =  0.10          # growth–return shock correlation

    # ── Permanent component diffusion (score-driven, small innovations) ──
    σ_ḡ::Float64   = 0.0002         # permanent growth random walk std
    σ_μ̄::Float64   = 0.0003         # permanent expected return random walk std

    # ── Steady-state pd ratio (used to calibrate κ) ──
    pd_ss::Float64 = 3.5
end

# ─────────────────────────────────────────────
# 2. State Struct
# ─────────────────────────────────────────────

"""
    ModelState

Full state at time t.
When `step!` is called on state_t, the fields hold values from period t.
After `step!` returns, the fields hold values from period t+1.

  ḡ, μ̄   — permanent (score-driven) components
  g̃, μ̃   — transitory components
  d       — log dividends
  pd      — log price-dividend ratio
  cum_log_r — cumulative log return from t=0
"""
mutable struct ModelState
    ḡ::Float64
    μ̄::Float64
    g̃::Float64
    μ̃::Float64
    d::Float64
    pd::Float64
    cum_log_r::Float64
end

# ─────────────────────────────────────────────
# 3. Present-Value Helpers
# ─────────────────────────────────────────────

"""
    calibrate_kappa(p::ModelParams) → Float64

Calibrate the Campbell-Shiller constant κ so that
    pd_ss = κ + (ḡ₀ − μ̄₀)/(1−ρ)
holds at the initial steady state (g̃=0, μ̃=0).
"""
function calibrate_kappa(p::ModelParams)
    return p.pd_ss - (p.ḡ₀ - p.μ̄₀) / (1.0 - p.ρ)
end

"""
    compute_pd(ḡ, g̃, μ̄, μ̃, κ, p) → Float64

Log price-dividend ratio from Campbell-Shiller present-value decomposition:

    pd_t = κ + (ḡ_t + g̃_t − μ̄_t)/(1−ρ) − μ̃_t/(1−ρ·ρ_μ)

Derivation (j indexes horizons, permanent components are martingales, g̃ is RW):
    pd_t = κ + ∑_{j≥1} ρ^{j-1} [E_t(Δd_{t+j}) − E_t(r_{t+j})]
         = κ + (g_t − μ̄_t)/(1−ρ) − μ̃_t/(1−ρ·ρ_μ)

Signs: higher expected growth ↑pd, higher expected return ↓pd.
"""
function compute_pd(ḡ::Float64, g̃::Float64, μ̄::Float64, μ̃::Float64,
                    κ::Float64, p::ModelParams)
    g_total  = ḡ + g̃
    denom_dr = 1.0 - p.ρ           # discount factor denominator
    denom_μ  = 1.0 - p.ρ * p.ρ_μ  # AR(1) accumulation denominator
    return κ + (g_total - μ̄) / denom_dr - μ̃ / denom_μ
end

# ─────────────────────────────────────────────
# 4. Shock Covariance
# ─────────────────────────────────────────────

"""
    shock_cholesky(p::ModelParams)

Cholesky factor L of the structural shock covariance matrix for
(ε_d, ε_g, ε_μ) with given standard deviations and correlations
ρ_dμ, ρ_gμ (ε_d and ε_g are assumed uncorrelated).

Usage: ε = L * z,  z ~ N(0, I₃)
"""
function shock_cholesky(p::ModelParams)
    Σ = [p.σ_d^2                      0.0                  p.ρ_dμ * p.σ_d * p.σ_μ;
         0.0                           p.σ_g^2              p.ρ_gμ * p.σ_g * p.σ_μ;
         p.ρ_dμ * p.σ_d * p.σ_μ       p.ρ_gμ * p.σ_g * p.σ_μ     p.σ_μ^2        ]
    return cholesky(Symmetric(Σ)).L
end

# ─────────────────────────────────────────────
# 5. Single-Step Simulation
# ─────────────────────────────────────────────

"""
    step!(state, κ, L, p, rng)

Advance the state by one quarter, mutating `state` in place.
Returns (Δd, log_r): realized dividend growth and log return.

Transition equations
────────────────────
    ḡ_t  = ḡ_{t-1} + σ_ḡ · η_ḡ              (score-driven martingale, small noise)
    μ̄_t  = μ̄_{t-1} + σ_μ̄ · η_μ̄              (score-driven martingale, small noise)
    g̃_t  = g̃_{t-1} + ε_g                     (random walk)
    μ̃_t  = ρ_μ · μ̃_{t-1} + ε_μ              (AR(1))
    Δd_t = (ḡ_{t-1} + g̃_{t-1}) + ε_d + ε_g  (expected + unexpected growth)
    pd_t = κ + (ḡ_t + g̃_t − μ̄_t)/(1−ρ) − μ̃_t/(1−ρ·ρ_μ)
    r_t  = Δd_t + log(1+exp(pd_t)) − pd_{t-1}
"""
function step!(state::ModelState, κ::Float64,
               L::LowerTriangular{Float64, Matrix{Float64}},
               p::ModelParams, rng::AbstractRNG)

    # ── Draw correlated structural shocks ──
    z    = randn(rng, 3)
    ε    = L * z                  # (ε_d, ε_g, ε_μ)
    ε_d, ε_g, ε_μ = ε[1], ε[2], ε[3]

    # ── Draw permanent-component innovations (small score-driven diffusion) ──
    η_ḡ = p.σ_ḡ * randn(rng)
    η_μ̄ = p.σ_μ̄ * randn(rng)

    # ── Update permanent components (martingales) ──
    ḡ_new = state.ḡ + η_ḡ
    μ̄_new = state.μ̄ + η_μ̄

    # ── Dividend growth ──
    # Expected component = (ḡ_{t-1} + g̃_{t-1}); unexpected = ε_d + ε_g
    Δd      = (state.ḡ + state.g̃_lag) + ε_d + ε_g
    d_new   = state.d + Δd

    # ── Update transitory components ──
    g̃_old  = state.g̃                               # capture before overwrite
    g̃_new  = state.g̃ + ε_g                        # random walk
    μ̃_new  = p.ρ_μ * state.μ̃ + ε_μ               # AR(1)

    # ── Log price-dividend ratio ──
    pd_new  = compute_pd(ḡ_new, g̃_new, μ̄_new, μ̃_new, κ, p)

    # ── Log return (Campbell-Shiller identity) ──
    log_r           = Δd + log(1.0 + exp(pd_new)) - state.pd
    cum_log_r_new   = state.cum_log_r + log_r

    # ── Mutate state ──
    state.ḡ         = ḡ_new
    state.μ̄         = μ̄_new
    state.g̃         = g̃_new
    state.μ̃         = μ̃_new
    state.g̃_lag     = g̃_old     # store g̃_{t-1} = pre-update g̃ for next step
    state.d         = d_new
    state.pd        = pd_new
    state.cum_log_r = cum_log_r_new

    return Δd, log_r
end

# ─────────────────────────────────────────────
# 6. Full Path Simulation
# ─────────────────────────────────────────────

"""
    SimulationResult

Full time-series from a single MC path.
All arrays have length T+1 (index 1 = initial state at t=0).
"""
struct SimulationResult
    T::Int
    d::Vector{Float64}          # log dividends
    ḡ::Vector{Float64}          # permanent growth (stored as τ in CSV)
    g::Vector{Float64}          # total expected growth g_t = ḡ_t + g̃_t
    g̃::Vector{Float64}          # transitory growth (cycle-like)
    μ::Vector{Float64}          # total expected return μ_t = μ̄_t + μ̃_t
    pd::Vector{Float64}         # log price-dividend ratio
    cum_log_r::Vector{Float64}  # cumulative log return (length T+1, first = 0)
end

"""
    default_params()

Return ModelParams with calibrated baseline values.
"""
function default_params()
    return ModelParams()
end

"""
    simulate_path(p::ModelParams, T_quarters::Int; seed=nothing)

Simulate the model for T_quarters periods.  Returns a SimulationResult.
"""
function simulate_path(p::ModelParams, T_quarters::Int;
                       seed::Union{Nothing, Int} = nothing)

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    κ = calibrate_kappa(p)
    L = shock_cholesky(p)

    # ── Initial state at t = 0 ──
    pd0    = compute_pd(p.ḡ₀, 0.0, p.μ̄₀, 0.0, κ, p)
    state  = ModelState(p.ḡ₀, p.μ̄₀, 0.0, 0.0, 0.0, 0.0, pd0, 0.0)

    n = T_quarters + 1
    d_arr     = Vector{Float64}(undef, n)
    ḡ_arr     = Vector{Float64}(undef, n)
    g_arr     = Vector{Float64}(undef, n)
    g̃_arr     = Vector{Float64}(undef, n)
    μ_arr     = Vector{Float64}(undef, n)
    pd_arr    = Vector{Float64}(undef, n)
    clr_arr   = Vector{Float64}(undef, n)

    # Store initial state
    d_arr[1]   = state.d
    ḡ_arr[1]   = state.ḡ
    g_arr[1]   = state.ḡ + state.g̃
    g̃_arr[1]   = state.g̃
    μ_arr[1]   = state.μ̄ + state.μ̃
    pd_arr[1]  = state.pd
    clr_arr[1] = 0.0

    for t in 2:n
        step!(state, κ, L, p, rng)
        d_arr[t]   = state.d
        ḡ_arr[t]   = state.ḡ
        g_arr[t]   = state.ḡ + state.g̃
        g̃_arr[t]   = state.g̃
        μ_arr[t]   = state.μ̄ + state.μ̃
        pd_arr[t]  = state.pd
        clr_arr[t] = state.cum_log_r
    end

    return SimulationResult(T_quarters, d_arr, ḡ_arr, g_arr, g̃_arr,
                            μ_arr, pd_arr, clr_arr)
end

# ─────────────────────────────────────────────
# 7. Return Analysis Helpers
# ─────────────────────────────────────────────

"""
    extract_at_horizon(sim::SimulationResult, horizon_quarters::Int)

Extract state values at a specific future quarter (index = horizon + 1).
"""
function extract_at_horizon(sim::SimulationResult, horizon_quarters::Int)
    idx = horizon_quarters + 1
    return (
        d         = sim.d[idx],
        g         = sim.g[idx],
        g̃         = sim.g̃[idx],
        μ         = sim.μ[idx],
        pd        = sim.pd[idx],
        cum_log_r = sim.cum_log_r[idx]
    )
end

"""
    annualised_cumulative_return(cum_log_r, horizon_years)

Convert a cumulative log return to an annualised (continuously compounded) return.
"""
function annualised_cumulative_return(cum_log_r::Float64, horizon_years::Float64)
    return cum_log_r / horizon_years
end

end # module DelleMonacheModel
