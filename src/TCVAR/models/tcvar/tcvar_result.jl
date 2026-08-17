"""
    TCVarResult

Posterior sampling result returned by [`gibbs_sampler`](@ref).

# Fields
- `model::TCVAR` : the sampled model, with the draw-dependent state-space blocks
  reset to their empty (zero) skeleton values
- `params::FlexiChain{Symbol}` : posterior parameter draws — `Στ` (trend
  innovation covariance, `n_trends × n_trends`), `β` (cycle VAR coefficients,
  `n_obs*p × n_obs`, predictors stacked oldest-lag-first) and `Σc` (cycle
  innovation covariance, `n_obs × n_obs`), each stored as a matrix-valued
  parameter
- `trend_states` : `n_kept × (T+1) × n_trends` array of sampled trend states
  (includes the initial state at t = 0)
- `cycle_states` : `n_kept × (T+p) × n_obs` array of sampled cycle states
  (includes the p pre-sample periods)
"""
struct TCVarResult
    model::TCVAR
    params::FlexiChain{VarName}
    trend_states::Array{Float64,3}
    cycle_states::Array{Float64,3}
end

"""
    build_result(model, trend_states, cycle_states, trend_covariance, betas, sigmas, burnin, thin)

Transform the raw Gibbs draws into a [`TCVarResult`](@ref): drop the burn-in,
thin, concatenate the flattened parameter draws (`trend_covariance`, `betas`,
`sigmas`) into a single `FlexiChain` with matrix-valued parameters `Στ`, `β`
and `Σc`, and reset the model's draw-dependent state-space blocks to zero so
the returned model carries the empty skeleton.
"""
function build_result(model::TCVAR, trend_states, cycle_states, trend_covariance, betas, sigmas, burnin, thin)

    n_trends = size(trend_covariance, 2)
    n_obs = size(sigmas, 2)
    k = size(betas, 2) ÷ n_obs # number of var coefficients per equation (lags stacked)
    p = k ÷ n_obs

    kept = burnin+1:thin:size(betas, 1)
    n_kept = length(kept)

    # One (iters × chains × params) array: each sample's matrix is flattened
    # column-major, so the FlexiChain key spec below reshapes it back exactly.
    params_array = reshape(
        hcat(reshape(trend_covariance[kept, :, :], n_kept, :),
             betas[kept, :],
             reshape(sigmas[kept, :, :], n_kept, :)),
        n_kept, 1, :)

    params = FlexiChain{VarName}(params_array, (
        Parameter(@varname(Στ)) => (n_trends, n_trends),
        Parameter(@varname(β)) => (k, n_obs),
        Parameter(@varname(Σc)) => (n_obs, n_obs)))

    # Return the model with empty params: zero the draw-dependent blocks the
    # sampler filled in, restoring the skeleton built by tc_var.
    update_tc_var!(model.ssm, zeros(n_obs, k), zeros(n_trends, n_trends),
                   zeros(n_obs, n_obs), n_trends, n_obs, p)

    return TCVarResult(model, params, trend_states[kept, :, :], cycle_states[kept, :, :])

end

"""
    posterior_mean(result::TCVarResult)

Posterior-mean parameter matrices averaged over the retained Gibbs draws, as a
NamedTuple `(Στ, β, Σc)`:

- `Στ` : trend innovation covariance, `n_trends × n_trends`
- `β`  : cycle VAR coefficients, `n_obs*p × n_obs` (predictors oldest-lag-first)
- `Σc` : cycle innovation covariance, `n_obs × n_obs`

Each draw of a matrix-valued parameter is a matrix, so `mean` over the draws
returns the posterior-mean matrix directly.
"""
posterior_mean(result::TCVarResult) = (
    Στ = mean(result.params[@varname(Στ)]),
    β  = mean(result.params[@varname(β)]),
    Σc = mean(result.params[@varname(Σc)]),
)

"""
    simulate_scenarios(result::TCVarResult, n_scenarios, n_steps)

Draw `n_scenarios` forward simulations of length `n_steps` from the estimated
trend-cycle VAR in `result`, using the [`sample`](@ref) state-space simulator.

The model is instantiated at the **posterior-mean** parameters (`Στ`, `β`, `Σc`)
returned by [`posterior_mean`](@ref); the draw-dependent state-space blocks of a
private copy of `result.model.ssm` are filled via [`update_tc_var!`](@ref) so the
model stored in `result` is left untouched. Every scenario starts from the common
posterior-mean terminal state — the last smoothed trend state and the last `p`
cycle states flattened oldest-lag-first (`ξ = [c_{t-p+1}; …; c_t]`) to match the
cycle companion ordering — then evolves stochastically through `sample`.

Returns `(states, observations)`:
- `states::Array{Float64,3}`       : `n_scenarios × n_steps × n_states`
- `observations::Array{Float64,3}` : `n_scenarios × n_steps × n_obs`
"""
function simulate_scenarios(result::TCVarResult, n_scenarios::Int, n_steps::Int)

    parameters = posterior_mean(result)
    n_obs = size(parameters.Σc, 1)
    p     = size(parameters.β, 1) ÷ n_obs

    # Common start state: posterior-mean last trend state, then the posterior-mean
    # last p cycle states stacked oldest-lag-first to match the companion order.
    trend_start = vec(mean(result.trend_states[:, end, :], dims = 1))
    cycle_mean  = dropdims(mean(result.cycle_states, dims = 1), dims = 1)  # (T+p) × n_obs
    cycle_start = vec(permutedims(cycle_mean[end-p+1:end, :]))
    initial_state = [trend_start; cycle_start]

    return simulate_scenarios(result.model, parameters, initial_state, n_scenarios, n_steps)
end

"""
    simulate_scenarios(model::TCVAR, params::NamedTuple, initial_state, n_scenarios, n_steps)

Simulate from `model` at explicit parameters `params = (Στ, β, Σc)` starting from
`initial_state = [τ₀; ξ₀]` (the trend block, then the cycle companion stacked
oldest-lag-first). `model.ssm` is left untouched — the parameters are written into a
private copy via [`update_tc_var!`](@ref).

Returns `(states, observations)`:
- `states::Array{Float64,3}`       : `n_scenarios × n_steps × (n_trends + n_obs*p)`
- `observations::Array{Float64,3}` : `n_scenarios × n_steps × n_obs`
"""
function simulate_scenarios(model::TCVAR, params::NamedTuple, initial_state::AbstractVector,
                            n_scenarios::Int, n_steps::Int)

    n_trends = size(params.Στ, 1)
    n_obs    = size(params.Σc, 1)
    k        = size(params.β, 1)      # n_obs * p
    p        = k ÷ n_obs

    length(initial_state) == n_trends + k || throw(DimensionMismatch(
        "initial_state must have length n_trends + n_obs*p = $(n_trends + k)"))

    # Instantiate the state-space model at the given parameters on a private copy of
    # the skeleton (the model passed in keeps its zero blocks).
    # var_coeff is β' — the n_obs × k companion bottom block.
    ssm = deepcopy(model.ssm)
    update_tc_var!(ssm, collect(params.β'), params.Στ, params.Σc, n_trends, n_obs, p)

    states       = zeros(n_scenarios, n_steps, n_trends + k)
    observations = zeros(n_scenarios, n_steps, n_obs)

    for s in 1:n_scenarios
        states[s, :, :], observations[s, :, :] = sample(ssm, initial_state, n_steps)
    end

    return states, observations
end
