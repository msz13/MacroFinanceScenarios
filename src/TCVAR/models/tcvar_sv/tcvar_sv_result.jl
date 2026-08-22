"""
    TCVarSVResult

Posterior sampling result of a [`TCVARSV`](@ref) model.

Follows [`TCVarResult`](@ref) exactly — one `FlexiChain` of parameter draws plus the state
arrays — with two differences: `Σc` is gone (under stochastic volatility the cycle
innovation covariance is `Σ_t = A₀⁻¹H_tA₀⁻ᵀ`, not a parameter) and the log-volatility path
`h` is added as a third state array.

# Fields
- `model::TCVARSV` : the sampled model, with the draw-dependent state-space blocks reset to
  their empty (zero) skeleton values
- `params::FlexiChain{VarName}` : posterior parameter draws —
  - `Στ` : trend innovation covariance, `n_trends × n_trends`
  - `β`  : cycle VAR coefficients, `n_obs*p × n_obs`, predictors oldest-lag-first
  - `A₀` : simultaneity matrix, `n_obs × n_obs`, unit lower triangular
  - `μ`  : unconditional log-volatility level, a length-`n_obs` **vector**
  - `Φ`  : log-volatility AR matrix, `n_obs × n_obs` (diagonal when
    `model.ar_structure === :diagonal`, stored full either way)
  - `Ω`  : log-volatility innovation covariance, `n_obs × n_obs`
- `trend_states` : `n_kept × (T+1) × n_trends`, includes the initial state at `t = 0`
- `cycle_states` : `n_kept × (T+p) × n_obs`, includes the `p` pre-sample periods
- `volatilities` : `n_kept × (T+1) × n_obs`, the log volatilities `h_{0:T}` — on the **log**
  scale, so report `exp(h/2)` for a standard deviation

`h` goes in the state arrays rather than in the chain for the same reason the trend and
cycle paths do: it is a `T`-length path, not a fixed-size parameter block, and
[`compute_posterior_statistics`](@ref) already works on that shape.

`L = A₀⁻¹` — the "correlation" factor with `Σ_t = L H_t Lᵀ` — is one `inv` away from the
stored `A₀`; `A₀` is what is stored because it is the parameterisation the sampler draws.
"""
struct TCVarSVResult
    model::TCVARSV
    params::FlexiChain{VarName}
    trend_states::Array{Float64,3}
    cycle_states::Array{Float64,3}
    volatilities::Array{Float64,3}
end

"""
    build_result(model::TCVARSV, trend_states, cycle_states, volatilities,
                 trend_covariance, betas, simultaneity, volatility_mean, volatility_ar,
                 volatility_covariance, burnin, thin)

Transform the raw Gibbs draws into a [`TCVarSVResult`](@ref): drop the burn-in, thin,
concatenate the flattened parameter draws into a single `FlexiChain` with the parameters
`Στ`, `β`, `A₀`, `μ`, `Φ`, `Ω`, and reset the model's draw-dependent state-space blocks to
zero so the returned model carries the empty skeleton.

Every raw draw array has the draws on its first axis: `trend_covariance` is
`n_draws × n_trends × n_trends`, `betas` is `n_draws × n_obs*k` (each row `vec(β)`,
column-major), `simultaneity` / `volatility_ar` / `volatility_covariance` are
`n_draws × n_obs × n_obs` and `volatility_mean` is `n_draws × n_obs`.
"""
function build_result(model::TCVARSV, trend_states, cycle_states, volatilities,
                      trend_covariance, betas, simultaneity, volatility_mean,
                      volatility_ar, volatility_covariance, burnin, thin)

    n_trends = size(trend_covariance, 2)
    n_obs = size(simultaneity, 2)
    k = size(betas, 2) ÷ n_obs        # VAR coefficients per equation (lags stacked)
    p = k ÷ n_obs

    kept = burnin+1:thin:size(betas, 1)
    n_kept = length(kept)

    # One (iters × chains × params) array: each sample's matrix is flattened column-major,
    # so the FlexiChain key spec below reshapes it back exactly. The order of the hcat and
    # the order of the key spec have to match.
    params_array = reshape(
        hcat(reshape(trend_covariance[kept, :, :], n_kept, :),
             betas[kept, :],
             reshape(simultaneity[kept, :, :], n_kept, :),
             volatility_mean[kept, :],
             reshape(volatility_ar[kept, :, :], n_kept, :),
             reshape(volatility_covariance[kept, :, :], n_kept, :)),
        n_kept, 1, :)

    params = FlexiChain{VarName}(params_array, (
        Parameter(@varname(Στ)) => (n_trends, n_trends),
        Parameter(@varname(β))  => (k, n_obs),
        Parameter(@varname(A₀)) => (n_obs, n_obs),
        Parameter(@varname(μ))  => (n_obs,),
        Parameter(@varname(Φ))  => (n_obs, n_obs),
        Parameter(@varname(Ω))  => (n_obs, n_obs)))

    # Return the model with empty params: zero the draw-dependent blocks the sampler filled
    # in, restoring the skeleton built by tc_var_sv.
    n_time = size(model.ssm.Q, 1)
    update_tc_var_sv!(model.ssm, zeros(n_obs, k), zeros(n_trends, n_trends),
                      zeros(n_time, n_obs, n_obs), n_trends, n_obs, p)

    return TCVarSVResult(model, params,
                         trend_states[kept, :, :],
                         cycle_states[kept, :, :],
                         volatilities[kept, :, :])
end

"""
    posterior_mean(result::TCVarSVResult)

Posterior-mean parameters averaged over the retained Gibbs draws, as a NamedTuple
`(Στ, β, A₀, μ, Φ, Ω)` — the same keys [`simulate_scenarios`](@ref) takes.

Each draw of a matrix-valued parameter is a matrix, so `mean` over the draws returns the
posterior-mean matrix directly.

The mean of `A₀` is still unit lower triangular (averaging preserves both), but note it is
*not* the simultaneity matrix of the mean `Σ_t`: the map from `A₀` to `Σ_t` is nonlinear.
Use it as a point summary, not as a plug-in identity.
"""
posterior_mean(result::TCVarSVResult) = (
    Στ = mean(result.params[@varname(Στ)]),
    β  = mean(result.params[@varname(β)]),
    A₀ = mean(result.params[@varname(A₀)]),
    μ  = mean(result.params[@varname(μ)]),
    Φ  = mean(result.params[@varname(Φ)]),
    Ω  = mean(result.params[@varname(Ω)]),
)

"""
    posterior_volatilities(result::TCVarSVResult; credible_level = 0.90)

Posterior mean and credible band of the volatility path on the **standard-deviation**
scale `exp(h/2)`, as three `(T+1) × n_obs` matrices `(mean, lower, upper)`.

The quantiles are taken of `exp(h/2)` itself rather than exponentiated after the fact —
`exp` is monotone, so the band is the same either way, but the mean is not: this returns
`E[exp(h/2)]`, not `exp(E[h]/2)`.
"""
posterior_volatilities(result::TCVarSVResult; credible_level::Float64 = 0.90) =
    compute_posterior_statistics(exp.(result.volatilities ./ 2);
                                 credible_level = credible_level)

"""
    simulate_volatility_path(μ, Φ, Ω, initial_volatility, n_steps) -> Matrix

One `n_steps × n_obs` forward path of the log volatilities

    h_t = μ + Φ (h_{t-1} − μ) + ν_t,   ν_t ~ N(0, Ω)

started at `h_1 = initial_volatility`, so row `t` lines up with period `t` of the state
path simulated by [`sample`](@ref) (whose first row is likewise the starting state).

The innovation factor comes from [`psd_factor`](@ref) rather than [`chol_psd`](@ref): with
`Ω = 0` the path stays at `initial_volatility` exactly, instead of picking up jitter draws.
"""
function simulate_volatility_path(μ, Φ, Ω, initial_volatility, n_steps::Int)

    n_obs = length(μ)
    Ω_factor = psd_factor(Ω)

    h = zeros(n_steps, n_obs)
    h[1, :] = initial_volatility
    for t in 2:n_steps
        h[t, :] = μ + Φ * (h[t-1, :] - μ) + Ω_factor * randn(n_obs)
    end

    return h
end

"""
    cycle_covariance_path(A₀, h) -> Array{Float64,3}

The cycle innovation covariances `Σ_t = A₀⁻¹ H_t A₀⁻ᵀ`, `H_t = diag(exp(h_t))`, of a log
volatility path `h` (`n_steps × n_obs`), in the `n_steps × n_obs × n_obs` layout
[`update_tc_var_sv!`](@ref) writes into `Q`.

`Σ_t` is built as `L Lᵀ` with `L = A₀⁻¹ diag(exp(h_t/2))` — positive semidefinite by
construction — and symmetrised, since the two triangular solves need not agree to the last
bit off the diagonal and the simulator factorises `Q_t`.
"""
function cycle_covariance_path(A₀, h::AbstractMatrix)

    n_steps, n_obs = size(h)
    # A₀ is unit lower triangular, so A₀ \ X is an O(n²) forward substitution.
    A = LowerTriangular(Matrix(float.(A₀)))

    covariances = zeros(n_steps, n_obs, n_obs)
    for t in 1:n_steps
        L = A \ Diagonal(exp.(h[t, :] ./ 2))
        covariances[t, :, :] = Symmetric(L * L')
    end

    return covariances
end

"""
    simulate_scenarios(result::TCVarSVResult, n_scenarios, n_steps)

Draw `n_scenarios` forward simulations of length `n_steps` from the estimated TCVAR-SV in
`result`, at the **posterior-mean** parameters returned by [`posterior_mean`](@ref).

Every scenario starts from the common posterior-mean terminal state — the last trend state,
the last `p` cycle states flattened oldest-lag-first (`ξ = [c_{T-p+1}; …; c_T]`) and the
last log volatility `h_T` — and then evolves stochastically, volatilities included. The
model stored in `result` is left untouched.

Returns `(states, observations, volatilities)`, all three with the same period axis:
- `states::Array{Float64,3}`       : `n_scenarios × n_steps × (n_trends + n_obs*p)`
- `observations::Array{Float64,3}` : `n_scenarios × n_steps × n_obs`
- `volatilities::Array{Float64,3}` : `n_scenarios × n_steps × n_obs`, log scale, the first
  row being the common starting `h_T`
"""
function simulate_scenarios(result::TCVarSVResult, n_scenarios::Int, n_steps::Int)

    parameters = posterior_mean(result)
    n_obs = length(parameters.μ)
    p     = size(parameters.β, 1) ÷ n_obs

    trend_start = vec(mean(result.trend_states[:, end, :], dims = 1))
    cycle_mean  = dropdims(mean(result.cycle_states, dims = 1), dims = 1)  # (T+p) × n_obs
    cycle_start = vec(permutedims(cycle_mean[end-p+1:end, :]))
    volatility_start = vec(mean(result.volatilities[:, end, :], dims = 1))

    return simulate_scenarios(result.model, parameters, [trend_start; cycle_start],
                              n_scenarios, n_steps; initial_volatility = volatility_start)
end

"""
    simulate_scenarios(model::TCVARSV, params::NamedTuple, initial_state, n_scenarios,
                       n_steps; initial_volatility = params.μ)

Simulate from `model` at explicit parameters `params = (Στ, β, A₀, μ, Φ, Ω)` starting from
`initial_state = [τ₀; ξ₀]` (the trend block, then the cycle companion stacked
oldest-lag-first) and the log volatility `initial_volatility`. This is the TCVAR-SV
counterpart of `simulate_scenarios(::TCVAR, …)`, and the generator of the simulated data
the recovery scripts estimate.

Each scenario draws its own volatility path
([`simulate_volatility_path`](@ref)), turns it into the per-period cycle innovation
covariances `Σ_t = A₀⁻¹H_tA₀⁻ᵀ` ([`cycle_covariance_path`](@ref)), writes those into a
private copy of the skeleton via [`update_tc_var_sv!`](@ref) — one `Q_t` per simulated
period, so `model.ssm` (whose `Q` is sized for the estimation sample) is left untouched —
and hands the result to [`sample`](@ref).

Since the path is simulated by [`sample`](@ref), the first row of every array is the
starting state itself: `states[s, 1, :] == initial_state`, `volatilities[s, 1, :] ==
initial_volatility`, and the `n_steps - 1` rows after it are the simulated periods. The
observation is drawn with the skeleton's `H = eps()·I`, i.e. it is `y_t = Λ τ_t + c_t` up
to the jitter [`sample`](@ref) adds to both covariances.

Returns `(states, observations, volatilities)`:
- `states::Array{Float64,3}`       : `n_scenarios × n_steps × (n_trends + n_obs*p)`
- `observations::Array{Float64,3}` : `n_scenarios × n_steps × n_obs`
- `volatilities::Array{Float64,3}` : `n_scenarios × n_steps × n_obs`, on the log scale
"""
function simulate_scenarios(model::TCVARSV, params::NamedTuple, initial_state::AbstractVector,
                            n_scenarios::Int, n_steps::Int; initial_volatility = params.μ)

    n_obs    = length(model.variable_names)
    n_trends = length(model.trend_names)
    n_states = size(model.ssm.Z, 2)
    k        = n_states - n_trends            # n_obs * p
    p        = k ÷ n_obs

    n_steps ≥ 1 || throw(ArgumentError("n_steps must be ≥ 1, got $n_steps"))
    size(params.β) == (k, n_obs) || throw(DimensionMismatch(
        "params.β must be $((k, n_obs)) (k × n_obs, oldest-lag-first), got $(size(params.β))"))
    length(initial_state) == n_states || throw(DimensionMismatch(
        "initial_state must have length n_trends + n_obs*p = $n_states, " *
        "got $(length(initial_state))"))
    length(initial_volatility) == n_obs || throw(DimensionMismatch(
        "initial_volatility must have length $n_obs, got $(length(initial_volatility))"))

    # A private skeleton carrying one Q_t per *simulated* period: the model's own Q is
    # sized for the estimation sample, and must stay untouched anyway.
    skeleton = model.ssm
    ssm = TimeVaryingStateSpaceModel(copy(skeleton.T), copy(skeleton.R), copy(skeleton.Z),
                                     zeros(n_steps, n_states, n_states), copy(skeleton.H))
    var_coeff = collect(float.(params.β)')     # n_obs × k, the companion bottom block

    states       = zeros(n_scenarios, n_steps, n_states)
    observations = zeros(n_scenarios, n_steps, n_obs)
    volatilities = zeros(n_scenarios, n_steps, n_obs)

    for s in 1:n_scenarios
        h = simulate_volatility_path(params.μ, params.Φ, params.Ω, initial_volatility,
                                     n_steps)
        update_tc_var_sv!(ssm, var_coeff, params.Στ, cycle_covariance_path(params.A₀, h),
                          n_trends, n_obs, p)

        states[s, :, :], observations[s, :, :] = sample(ssm, initial_state, n_steps)
        volatilities[s, :, :] = h
    end

    return states, observations, volatilities
end
