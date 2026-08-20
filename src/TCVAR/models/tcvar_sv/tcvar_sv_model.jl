# The trend-cycle VAR with multivariate stochastic volatility: the state-space skeleton,
# the model struct, the in-place per-draw update, and the forward simulator.
#
# Everything here is the TCVAR skeleton with one change — the cycle innovation covariance
# `Σ_t = A₀⁻¹ H_t A₀⁻ᵀ` moves with `t`, so the state-noise covariance `Q` is an
# `n_time × n_states × n_states` array instead of a matrix and the model is a
# [`TimeVaryingStateSpaceModel`](@ref). The constant blocks (`T`'s identity/shift
# structure, `R`, `Z`, `H`) are taken from [`tc_var`](@ref) itself rather than rebuilt, so
# the two models cannot drift apart.

"""
    tc_var_sv(trend_mapping, n_time; p = 1) -> TimeVaryingStateSpaceModel

State-space *skeleton* of a trend-cycle model with stochastic volatility, over a sample of
`n_time` periods.

Identical to [`tc_var`](@ref) — same companion ordering (oldest-lag-first
`ξ_t = [c_{t-p+1}; …; c_t]`), same `R`, `Z` and `H` — except that the state-noise
covariance is time varying:

    Q_t = blockdiag(Στ, 0_{n(p-1)}, Σ_t),     Σ_t = A₀⁻¹ H_t A₀⁻ᵀ

so `Q` is an `n_time × n_states × n_states` array, allocated here once and filled in place
by [`update_tc_var_sv!`](@ref) on every sweep. As in [`tc_var`](@ref) the draw-dependent
blocks — the VAR companion bottom block of `T`, the trend and contemporaneous-cycle blocks
of every `Q_t` — start at zero, and the initial state distribution is the caller's, not the
model's.

`n_time` is the number of observed periods `T` (the length of the data the model will be
estimated on), not the length of the state path: the pre-sample state at `t = 0` carries no
`Q_t` of its own, since it comes from the initial distribution.
"""
function tc_var_sv(trend_mapping, n_time::Int; p::Int = 1)
    n_time ≥ 1 || throw(ArgumentError("n_time must be ≥ 1, got $n_time"))

    skeleton = tc_var(trend_mapping; p = p)
    n_states = size(skeleton.T, 1)

    return TimeVaryingStateSpaceModel(skeleton.T, skeleton.R, skeleton.Z,
                                      zeros(n_time, n_states, n_states), skeleton.H)
end

"""
    TCVARSV

Trend-cycle VAR with multivariate stochastic volatility (Cogley–Sargent 2005): the cycle
innovations are `ε_t ~ N(0, Σ_t)` with

    Σ_t = A₀⁻¹ H_t A₀⁻ᵀ,          H_t = diag(exp(h_{1t}), …, exp(h_{nt}))
    h_t = μ + Φ (h_{t-1} − μ) + ν_t,   ν_t ~ N(0, Ω)

`A₀` is unit lower triangular and constant; only the variances move. Bundles the skeleton
built by [`tc_var_sv`](@ref) with the priors and the variable / trend names, so the sampler
receives one ready-made model.

# Fields
- `ssm::TimeVaryingStateSpaceModel` : skeleton with a 3-D `Q`; the draw-dependent blocks
  start at zero and are filled by [`update_tc_var_sv!`](@ref) during sampling
- `priors::TCVARSVPriors` : the nine priors, validated by [`tcvar_sv_priors`](@ref)
- `variable_names::Vector{String}` : names of the observed variables
- `trend_names::Vector{String}` : names of the trend states
- `ar_structure::Symbol` : `:diagonal` (default — one AR(1) persistence per series, as in
  Clark–Ravazzolo) or `:full` (an unrestricted `Φ`). It fixes the shape of the
  `volatility_ar` prior, which is re-checked here against the tuple it was built with.

!!! warning "The identification is ordering-dependent"
    `A₀` is a Cholesky-type factorisation of `Σ_t`: reordering the columns of the data
    gives a different `A₀` and different volatilities, exactly as in a recursive SVAR.
    Order the variables deliberately.

!!! note "There is no `Σc`"
    Under stochastic volatility the cycle innovation covariance is `Σ_t`, not a free
    parameter, so the posterior chain carries `A₀`, `μ`, `Φ`, `Ω` and the path `h` instead.
    `priors.cycle_covariance` survives only as the prior mean `Σ̄` — see
    [`TCVARSVPriors`](@ref).
"""
struct TCVARSV
    ssm::TimeVaryingStateSpaceModel
    priors::TCVARSVPriors
    variable_names::Vector{String}
    trend_names::Vector{String}
    ar_structure::Symbol
end

"""
    TCVARSV(trend_mapping, priors, n_time; variable_names, trend_names, ar_structure = :diagonal)

Build a [`TCVARSV`](@ref) from an `n_obs × n_trends` trend-to-observation mapping, a
validated [`TCVARSVPriors`](@ref) tuple and the sample length `n_time`. The number of lags
`p` and the number of variables `n` come from `priors.cycle_β`.

`n_time` is required — unlike TCVAR, whose `Q` is one matrix, the skeleton here carries one
`Q_t` per observed period, so the model is built for a sample of a given length and the
sweep checks the data against it.

Only the checks that need `trend_mapping` are made here; everything internal to the prior
tuple was already checked by [`tcvar_sv_priors`](@ref). `variable_names` defaults to
`["y1", …]` and `trend_names` to `["τ1", …]`.

```julia
Σc_prior, β_prior, c₀_prior = var_priors(0.2, 1, [0.5, 1.0, 2.0] .^ 2; δ = zeros(3))

priors = tcvar_sv_priors((initial_trend    = MvNormal(zeros(3), Matrix(1.0I, 3, 3)),
                          initial_cycle    = c₀_prior,
                          trend_covariance = InverseWishart(20.0, Matrix(0.3I, 3, 3)),
                          cycle_covariance = Σc_prior,
                          cycle_β          = β_prior),
                         sv_priors(3))

model = TCVARSV(Matrix(1.0I, 3, 3), priors, 400)
```
"""
function TCVARSV(trend_mapping, priors::TCVARSVPriors, n_time::Int;
                 variable_names = default_variable_names(size(trend_mapping, 1)),
                 trend_names = default_trend_names(size(trend_mapping, 2)),
                 ar_structure::Symbol = :diagonal)

    n_obs, n_trends = size(trend_mapping)
    β_prior = priors.cycle_β
    p = β_prior.p

    β_prior.n == n_obs || throw(DimensionMismatch(
        "priors.cycle_β is built for n = $(β_prior.n) variables, trend_mapping has $n_obs"))
    length(priors.initial_trend) == n_trends || throw(DimensionMismatch(
        "priors.initial_trend must have length $n_trends"))
    size(priors.trend_covariance) == (n_trends, n_trends) || throw(DimensionMismatch(
        "priors.trend_covariance must be $n_trends × $n_trends"))
    # Repeated from tcvar_sv_priors: the tuple does not carry ar_structure, so this is what
    # ties the field to the prior the tuple was actually built with.
    length(priors.volatility_ar) == volatility_ar_length(n_obs, ar_structure) ||
        throw(DimensionMismatch(
            "priors.volatility_ar has length $(length(priors.volatility_ar)), but " *
            "ar_structure = :$ar_structure needs " *
            "$(volatility_ar_length(n_obs, ar_structure))"))
    length(variable_names) == n_obs ||
        throw(DimensionMismatch("variable_names must have length $n_obs"))
    length(trend_names) == n_trends ||
        throw(DimensionMismatch("trend_names must have length $n_trends"))

    return TCVARSV(tc_var_sv(trend_mapping, n_time; p = p),
                   priors,
                   collect(String, variable_names),
                   collect(String, trend_names),
                   ar_structure)
end

"""
    update_tc_var_sv!(model, var_coeff, trend_cov, cycle_covariances, n_trends, n_variables, p)

Overwrite, in place, only the blocks of the skeleton that change between Gibbs draws — the
time-varying counterpart of [`update_tc_var!`](@ref):

* the VAR companion bottom block `[A_p … A_1]` of the transition `T` (`var_coeff`, size
  `n_variables × n_variables*p`),
* the trend block of every `Q_t` (`trend_cov`, the same matrix in every period — the trends
  are homoskedastic),
* the contemporaneous-cycle block of every `Q_t` (`cycle_covariances[t, :, :]` = `Σ_t`, an
  `n_time × n_variables × n_variables` array).

The `n_time` writes go straight into the array the model already owns, so no
`n_time × n_states × n_states` allocation happens between draws. Every other block built by
[`tc_var_sv`](@ref) is constant and left untouched, and the initial state distribution lives
outside the model, as it does for TCVAR.
"""
function update_tc_var_sv!(model::TimeVaryingStateSpaceModel, var_coeff, trend_cov,
                           cycle_covariances, n_trends, n_variables, p)

    Q = model.Q
    Q isa Array{Float64,3} || throw(ArgumentError(
        "update_tc_var_sv! needs the time-varying Q built by tc_var_sv; this model " *
        "carries a constant $(size(Q)) matrix"))

    n_cycle_states = n_variables * p
    n_states = n_trends + n_cycle_states
    n_time = size(Q, 1)

    size(Q, 2) == n_states || throw(DimensionMismatch(
        "model has $(size(Q, 2)) states but n_trends + n_variables*p = $n_states"))
    size(cycle_covariances) == (n_time, n_variables, n_variables) || throw(DimensionMismatch(
        "cycle_covariances must be $((n_time, n_variables, n_variables)) " *
        "(one Σ_t per period), got $(size(cycle_covariances))"))

    # Companion bottom block (oldest-lag-first), exactly as in update_tc_var!.
    row0 = n_trends + n_variables * (p - 1)
    model.T[row0+1:end, n_trends+1:end] = var_coeff

    cycle0 = n_states - n_variables      # offset of the contemporaneous cycle block
    @inbounds for t in 1:n_time
        for j in 1:n_trends, i in 1:n_trends
            Q[t, i, j] = trend_cov[i, j]
        end
        for j in 1:n_variables, i in 1:n_variables
            Q[t, cycle0+i, cycle0+j] = cycle_covariances[t, i, j]
        end
    end

    return model
end

"""
    simulate_tcvar_sv(model::TCVARSV, params, initial_state, n_steps; initial_volatility)
        -> (states, observations, volatilities)

    simulate_tcvar_sv(model::TCVARSV, params, initial_state, n_scenarios, n_steps;
                      initial_volatility) -> (states, observations, volatilities)

Simulate a TCVAR-SV forward from explicit parameters — the stochastic-volatility analogue
of [`simulate_scenarios`](@ref), and the generator of the simulated data the recovery
scripts estimate.

`params` is a NamedTuple `(Στ, β, A₀, μ, Φ, Ω)`; `β` is `k × n_obs` with the predictors
stacked oldest-lag-first (so the companion bottom block is `β'`) and `A₀` is the unit
lower-triangular simultaneity matrix. `initial_state = [τ₀; ξ₀]` stacks the trend block and
the cycle companion `ξ₀ = [c_{-p+1}; …; c_0]`, matching the state layout of
[`tc_var_sv`](@ref); `initial_volatility` is `h₀` and defaults to `params.μ`.

Each step draws, in this order,

    h_t = μ + Φ (h_{t-1} − μ) + ν_t,   ν_t ~ N(0, Ω)
    ε_t = A₀⁻¹ (exp(h_t/2) ⊙ η_t),     η_t ~ N(0, I)      so ε_t ~ N(0, A₀⁻¹H_tA₀⁻ᵀ)
    c_t = β' ξ_{t-1} + ε_t,   τ_t = τ_{t-1} + u_t,  u_t ~ N(0, Στ)
    y_t = Λ τ_t + c_t

The observation is taken from the model's own `Z`, and *without* observation noise: the
skeleton's `H = eps()·I` says the observation is an exact identity, so adding noise here
would simulate a different model than the one that will be estimated.

Returns `(states, observations, volatilities)` — for the single-path form,
`n_steps × n_states`, `n_steps × n_obs` and `(n_steps+1) × n_obs` (the volatility path
carries `h_0` in its first row, so it lines up with the trend/cycle pre-sample); for the
`n_scenarios` form, the same with a leading scenario axis.

Noise factors come from [`psd_factor`](@ref), so a switched-off block is exactly switched
off: with `Ω = 0` the volatility path stays at `h₀` and the simulation is an ordinary
homoskedastic TCVAR at `Σ = A₀⁻¹diag(exp(h₀))A₀⁻ᵀ`.
"""
function simulate_tcvar_sv(model::TCVARSV, params::NamedTuple, initial_state::AbstractVector,
                           n_steps::Int; initial_volatility = params.μ)

    n_obs    = length(model.variable_names)
    n_trends = length(model.trend_names)
    n_states = size(model.ssm.Z, 2)
    k        = n_states - n_trends            # n_obs * p
    p        = k ÷ n_obs

    n_steps ≥ 1 || throw(ArgumentError("n_steps must be ≥ 1, got $n_steps"))
    size(params.β) == (k, n_obs) || throw(DimensionMismatch(
        "params.β must be $((k, n_obs)) (k × n_obs, oldest-lag-first), got $(size(params.β))"))
    size(params.Στ) == (n_trends, n_trends) || throw(DimensionMismatch(
        "params.Στ must be $n_trends × $n_trends, got $(size(params.Στ))"))
    size(params.A₀) == (n_obs, n_obs) || throw(DimensionMismatch(
        "params.A₀ must be $n_obs × $n_obs, got $(size(params.A₀))"))
    size(params.Φ) == (n_obs, n_obs) || throw(DimensionMismatch(
        "params.Φ must be $n_obs × $n_obs, got $(size(params.Φ))"))
    size(params.Ω) == (n_obs, n_obs) || throw(DimensionMismatch(
        "params.Ω must be $n_obs × $n_obs, got $(size(params.Ω))"))
    length(params.μ) == n_obs || throw(DimensionMismatch(
        "params.μ must have length $n_obs, got $(length(params.μ))"))
    length(initial_state) == n_states || throw(DimensionMismatch(
        "initial_state must have length n_trends + n_obs*p = $n_states, " *
        "got $(length(initial_state))"))
    length(initial_volatility) == n_obs || throw(DimensionMismatch(
        "initial_volatility must have length $n_obs, got $(length(initial_volatility))"))

    μ = collect(float.(params.μ))
    Φ = Matrix(float.(params.Φ))
    # A₀ is unit lower triangular, so ε = A₀ \ u is an O(n²) forward substitution.
    A₀ = LowerTriangular(Matrix(float.(params.A₀)))
    var_coeff = collect(float.(params.β)')          # n_obs × k, companion bottom block
    Z = model.ssm.Z

    trend_factor = psd_factor(params.Στ)
    Ω_factor     = psd_factor(params.Ω)

    states       = zeros(n_steps, n_states)
    observations = zeros(n_steps, n_obs)
    volatilities = zeros(n_steps + 1, n_obs)

    state = collect(float.(initial_state))
    h     = collect(float.(initial_volatility))
    volatilities[1, :] = h

    for t in 1:n_steps
        h = μ + Φ * (h - μ) + Ω_factor * randn(n_obs)

        innovation = A₀ \ (exp.(h ./ 2) .* randn(n_obs))
        cycle = var_coeff * state[n_trends+1:end] + innovation
        trend = state[1:n_trends] + trend_factor * randn(n_trends)

        # Shift the companion: ξ_t = [ξ_{t-1}[n_obs+1:end]; c_t].
        state = [trend; state[n_trends+n_obs+1:end]; cycle]

        states[t, :]       = state
        observations[t, :] = Z * state
        volatilities[t+1, :] = h
    end

    return states, observations, volatilities
end

function simulate_tcvar_sv(model::TCVARSV, params::NamedTuple, initial_state::AbstractVector,
                           n_scenarios::Int, n_steps::Int; initial_volatility = params.μ)

    n_obs    = length(model.variable_names)
    n_states = size(model.ssm.Z, 2)

    states       = zeros(n_scenarios, n_steps, n_states)
    observations = zeros(n_scenarios, n_steps, n_obs)
    volatilities = zeros(n_scenarios, n_steps + 1, n_obs)

    for s in 1:n_scenarios
        states[s, :, :], observations[s, :, :], volatilities[s, :, :] =
            simulate_tcvar_sv(model, params, initial_state, n_steps;
                              initial_volatility = initial_volatility)
    end

    return states, observations, volatilities
end
