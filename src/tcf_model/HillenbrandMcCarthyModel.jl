"""
    HillenbrandMcCarthyModel

Implements the trend-cycle present-value model from:
  Hillenbrand & McCarthy (2026) "Expected Returns with Cash Flow Trends and Cycles"

State vector: s_t = (τ_t, g_t, c_t, ε_c_t, μ_t, 1)ᵀ
Observables:  x_t = (d_t, pd_t)ᵀ

The module provides:
  - Parameter struct with paper estimates as defaults
  - State-space matrix constructors (A, C, G_t)
  - Single-step simulation
  - Full path simulation (cash flows, pd ratio, returns)
  - Present-value helper functions
"""
module HillenbrandMcCarthyModel

using Random, Distributions, LinearAlgebra

export ModelParams, ModelState, default_params,
       steady_state_pd, simulate_path, compute_log_returns,
       compute_cumulative_returns, annualised_cumulative_return

# ─────────────────────────────────────────────
# 1. Parameter & State Structs
# ─────────────────────────────────────────────

"""
    ModelParams

All estimated (quarterly) parameters from Table 2 of the paper.
"""
Base.@kwdef struct ModelParams
    # Cash-flow cycle ARMA(1,1)
    ρ_c::Float64  = 0.9710       # AR(1) persistence of cycle level
    θ_c::Float64  = 0.4927       # MA(1) coefficient
    σ_c::Float64  = 0.0273       # std dev of cycle shock

    # Trend growth random walk
    σ_g::Float64  = 0.000226     # std dev of trend growth shock (0.0226 / 100)

    # Expected returns AR(1)
    ρ_μ::Float64  = 0.9556       # quarterly persistence
    σ_μ::Float64  = 0.0048       # std dev of expected return shock
    μ_bar::Float64 = 0.0215      # quarterly steady-state expected return

    # Campbell-Shiller linearisation constant
    ρ::Float64    = 0.99         # quarterly discount factor (≈ 0.96 annual)
end

"""
    ModelState

Full state at time t.  Tracks both the latent state vector
and the observable log-dividends / pd ratio.
"""
mutable struct ModelState
    τ::Float64      # trend level
    g::Float64      # trend growth rate
    c::Float64      # cycle level
    ε_c::Float64    # lagged cycle shock (for MA(1) term)
    μ::Float64      # expected return

    d::Float64      # log dividends  (= τ + c)
    pd::Float64     # log price-dividend ratio
end

# ─────────────────────────────────────────────
# 2. Present-Value Helpers
# ─────────────────────────────────────────────

"""
    steady_state_pd(g, μ_bar)

Compute the steady-state log price-dividend ratio:
    p̄d(g) = g − log(exp(μ̄) − exp(g))
Requires g < μ̄.
"""
function steady_state_pd(g::Float64, μ_bar::Float64)
    @assert g < μ_bar "Trend growth g=$g must be below μ̄=$μ_bar for finite valuations"
    return g - log(exp(μ_bar) - exp(g))
end

"""
    jensen_delta(p::ModelParams)

Jensen's inequality constant: δ = ρ / (2(1-ρ)²) · σ²_g
"""
function jensen_delta(p::ModelParams)
    return p.ρ * p.σ_g^2 / (2.0 * (1.0 - p.ρ)^2)
end

"""
    M_t(μ, p::ModelParams)

Expected-return component of the pd ratio:
    M_t = −1/(1 − ρ·ρ_μ) · (μ_t − μ̄)
"""
function M_component(μ::Float64, p::ModelParams)
    return -1.0 / (1.0 - p.ρ * p.ρ_μ) * (μ - p.μ_bar)
end

"""
    C_component(c, ε_c, p::ModelParams)

Cash-flow cycle component of the pd ratio:
    C_t = −(1−ρ_c)/(1−ρ·ρ_c) · c_t + (1−ρ)θ_c/(1−ρ·ρ_c) · ε_{c,t}
"""
function C_component(c::Float64, ε_c::Float64, p::ModelParams)
    denom = 1.0 - p.ρ * p.ρ_c
    β_c = -(1.0 - p.ρ_c) / denom
    β_ε = p.θ_c * (1.0 - p.ρ) / denom
    return β_c * c + β_ε * ε_c
end

"""
    compute_pd(state::ModelState, p::ModelParams)

Full model-implied log price-dividend ratio:
    pd_t = δ + p̄d(g_t) + M_t + C_t
"""
function compute_pd(g, c, ε_c, μ, p::ModelParams)
    δ  = jensen_delta(p)
    pd_bar = steady_state_pd(g, p.μ_bar)
    M  = M_component(μ, p)
    C  = C_component(c, ε_c, p)
    return δ + pd_bar + M + C
end

# ─────────────────────────────────────────────
# 3. Single-Step State Transition
# ─────────────────────────────────────────────

"""
    step!(state::ModelState, p::ModelParams, rng::AbstractRNG)

Advance the state by one quarter.
Returns the three structural shocks drawn this period.
"""
function step!(state::ModelState, p::ModelParams, rng::AbstractRNG)
    # Draw i.i.d. standard normal shocks
    z_g = randn(rng)
    z_c = randn(rng)
    z_μ = randn(rng)

    # Structural shocks
    ε_g = p.σ_g * z_g
    ε_c_new = p.σ_c * z_c
    ε_μ = p.σ_μ * z_μ

    # ── Transition equations ──
    # Trend growth (random walk)
    g_new = state.g + ε_g

    # Trend level (deterministic given g)
    τ_new = state.τ + state.g          # uses previous g (τ_t = τ_{t-1} + g_{t-1})

    # Cycle (ARMA(1,1))
    c_new = p.ρ_c * state.c + p.θ_c * state.ε_c + ε_c_new

    # Expected returns (AR(1) around μ̄)
    μ_new = p.μ_bar + p.ρ_μ * (state.μ - p.μ_bar) + ε_μ

    # ── Update state ──
    state.τ   = τ_new
    state.g   = g_new
    state.c   = c_new
    state.ε_c = ε_c_new
    state.μ   = μ_new

    # ── Observables ──
    state.d  = τ_new + c_new
    state.pd = compute_pd(g_new, c_new, ε_c_new, μ_new, p)

    return (ε_g, ε_c_new, ε_μ)
end

# ─────────────────────────────────────────────
# 4. Full Path Simulation
# ─────────────────────────────────────────────

"""
    default_initial_state(p::ModelParams)

Initialise state at approximate 1871 starting values
(trend growth ≈ 2% annual ≈ 0.5% quarterly, μ at steady state).
"""
function default_initial_state(p::ModelParams)
    g0 = 0.005                        # ≈ 2% annual trend growth
    τ0 = 0.0                          # normalise initial trend to 0
    c0 = 0.0
    ε_c0 = 0.0
    μ0 = p.μ_bar

    d0  = τ0 + c0
    pd0 = compute_pd(g0, c0, ε_c0, μ0, p)
    return ModelState(τ0, g0, c0, ε_c0, μ0, d0, pd0)
end

"""
    SimulationResult

Stores full time-series output of a single simulation path.
All arrays are of length T+1 (including initial state at index 1).
"""
struct SimulationResult
    T::Int                    # number of quarters simulated
    d::Vector{Float64}        # log dividends
    τ::Vector{Float64}        # trend level
    g::Vector{Float64}        # trend growth
    c::Vector{Float64}        # cycle
    μ::Vector{Float64}        # expected return
    pd::Vector{Float64}       # log price-dividend ratio
    log_r::Vector{Float64}    # quarterly log returns (length T)
    cum_log_r::Vector{Float64}# cumulative log returns from t=0 (length T+1, first = 0)
end

"""
    simulate_path(p::ModelParams, T_quarters::Int;
                  seed=nothing, initial_state=nothing)

Simulate the model for T_quarters periods.
Returns a `SimulationResult`.
"""
function simulate_path(p::ModelParams, T_quarters::Int;
                       seed::Union{Nothing,Int}=nothing,
                       initial_state::Union{Nothing,ModelState}=nothing)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    state = isnothing(initial_state) ? default_initial_state(p) : deepcopy(initial_state)

    # Pre-allocate arrays (index 1 = initial state)
    n = T_quarters + 1
    d_arr  = Vector{Float64}(undef, n)
    τ_arr  = Vector{Float64}(undef, n)
    g_arr  = Vector{Float64}(undef, n)
    c_arr  = Vector{Float64}(undef, n)
    μ_arr  = Vector{Float64}(undef, n)
    pd_arr = Vector{Float64}(undef, n)

    # Store initial state
    d_arr[1]  = state.d
    τ_arr[1]  = state.τ
    g_arr[1]  = state.g
    c_arr[1]  = state.c
    μ_arr[1]  = state.μ
    pd_arr[1] = state.pd

    for t in 2:n
        step!(state, p, rng)
        d_arr[t]  = state.d
        τ_arr[t]  = state.τ
        g_arr[t]  = state.g
        c_arr[t]  = state.c
        μ_arr[t]  = state.μ
        pd_arr[t] = state.pd
    end

    # ── Compute quarterly log returns ──
    # From the Campbell-Shiller identity:
    #   r_{t,t+1} = Δd_{t+1} + log(1 + exp(pd_{t+1})) − pd_t
    log_r = Vector{Float64}(undef, T_quarters)
    for t in 1:T_quarters
        Δd = d_arr[t+1] - d_arr[t]
        log_r[t] = Δd + log(1.0 + exp(pd_arr[t+1])) - pd_arr[t]
    end

    # Cumulative log returns
    cum_log_r = Vector{Float64}(undef, n)
    cum_log_r[1] = 0.0
    for t in 2:n
        cum_log_r[t] = cum_log_r[t-1] + log_r[t-1]
    end

    return SimulationResult(T_quarters, d_arr, τ_arr, g_arr, c_arr, μ_arr,
                            pd_arr, log_r, cum_log_r)
end

# ─────────────────────────────────────────────
# 5. Return Analysis Helpers
# ─────────────────────────────────────────────

"""
    compute_cumulative_returns(sim::SimulationResult, horizon_quarters::Int)

Extract all non-overlapping cumulative log returns over a given horizon
from a simulation result.
"""
function compute_cumulative_returns(sim::SimulationResult, horizon_quarters::Int)
    n_windows = div(sim.T, horizon_quarters)
    cum_rets = Vector{Float64}(undef, n_windows)
    for i in 1:n_windows
        t_start = (i - 1) * horizon_quarters + 1
        t_end   = i * horizon_quarters + 1
        cum_rets[i] = sim.cum_log_r[t_end] - sim.cum_log_r[t_start]
    end
    return cum_rets
end

"""
    annualised_cumulative_return(cum_log_r::Float64, horizon_years::Float64)

Convert a cumulative log return to an annualised (continuously compounded) return.
"""
function annualised_cumulative_return(cum_log_r::Float64, horizon_years::Float64)
    return cum_log_r / horizon_years
end

"""
    extract_at_horizon(sim::SimulationResult, horizon_quarters::Int)

Extract the state values at a specific future quarter.
Returns a NamedTuple with (:d, :g, :c, :μ, :pd, :cum_log_r).
"""
function extract_at_horizon(sim::SimulationResult, horizon_quarters::Int)
    idx = horizon_quarters + 1   # +1 because index 1 = time 0
    return (
        d = sim.d[idx],
        g = sim.g[idx],
        c = sim.c[idx],
        μ = sim.μ[idx],
        pd = sim.pd[idx],
        cum_log_r = sim.cum_log_r[idx]
    )
end

"""
    default_params()

Return ModelParams with the paper's baseline estimates.
"""
function default_params()
    return ModelParams()
end

end # module
