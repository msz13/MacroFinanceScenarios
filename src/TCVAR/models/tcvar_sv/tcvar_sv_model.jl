# The trend-cycle VAR with multivariate stochastic volatility: the state-space skeleton,
# the model struct and the in-place per-draw update. Forward simulation lives with the
# result, in `tcvar_sv_result.jl`, exactly as it does for TCVAR.
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
