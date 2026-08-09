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
