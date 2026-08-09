function covariance_posterior(data, scale_prior, d_posterior)
    
    res = diff(data, dims=1)
    posterior_mean = res' * res .+ scale_prior

    return InverseWishart(d_posterior, posterior_mean)    

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

    params   = posterior_mean(result)
    n_trends = size(params.Στ, 1)
    n_obs    = size(params.Σc, 1)
    k        = size(params.β, 1)      # n_obs * p
    p        = k ÷ n_obs

    # Instantiate the state-space model at the posterior-mean parameters on a
    # private copy of the skeleton (the model in `result` keeps its zero blocks).
    # var_coeff is β' — the n_obs × k companion bottom block.
    ssm = deepcopy(result.model.ssm)
    update_tc_var!(ssm, collect(params.β'), params.Στ, params.Σc, n_trends, n_obs, p)

    # Common start state: posterior-mean last trend state, then the posterior-mean
    # last p cycle states stacked oldest-lag-first to match the companion order.
    trend_start = vec(mean(result.trend_states[:, end, :], dims = 1))
    cycle_mean  = dropdims(mean(result.cycle_states, dims = 1), dims = 1)  # (T+p) × n_obs
    cycle_start = vec(permutedims(cycle_mean[end-p+1:end, :]))
    initial_state = [trend_start; cycle_start]

    n_states = length(initial_state)
    states       = zeros(n_scenarios, n_steps, n_states)
    observations = zeros(n_scenarios, n_steps, n_obs)

    for s in 1:n_scenarios
        states[s, :, :], observations[s, :, :] = sample(ssm, initial_state, n_steps)
    end

    return states, observations
end

"""
    model::TCVAR: trend-cycle VAR model bundling the state-space skeleton, the
        trend / initial-state priors (with the cycle `MinnesotaPrior` under
        `priors.cycle`) and the variable / trend names (see [`TCVAR`](@ref)).
    data: T x m matrix of observations
    n_samples: number of retained samples
    burnin: number of samples to discard
    thin: keep every `thin`-th sample

Returns a [`TCVarResult`](@ref) bundling the (skeleton-reset) model, the
parameter draws as one `FlexiChain` and the sampled trend / cycle states.

The cycle is mean-zero, so the prior's intercept row is dropped. `MinnesotaPrior`
orders its regressors newest-lag-first (`[lag1 … lagp, const]`), whereas the
state-space sampler stacks predictors oldest-lag-first (see `prepare_var_data`),
so the lag blocks of `Φ₀` and `Ω` are reversed here. Following the
Giannone–Lenza–Primiceri parameterization `Σ ~ IW(Ψ, d)`, the prior IW scale is
`Ψ` directly.
"""

function gibbs_sampler(model::TCVAR, data; burnin = 1000, n_samples=1000, thin=1, logging=false)

    priors = model.priors
    cycle_prior = priors.cycle
    ssm = model.ssm

    n_time_steps, n_obs = size(data)
    n_trends = length(model.trend_names)
    p = cycle_prior.p
    k = n_obs * p #number of var coefficients per equation (lags stacked)

    # sample_states prepends the drawn pre-sample to the t = 1..T smoothed states.
    # Trends are a random walk, so only the single initial state (t = 0) is
    # prepended -> n_time_steps + 1 points. The cycle companion carries p
    # pre-sample periods (c_{-p+1}, ..., c_0), so the cycle path spans
    # n_time_steps + p points.
    n_trend_time_steps = n_time_steps + 1
    n_cycle_time_steps = n_time_steps + p

    n_obs == cycle_prior.n ||
        throw(DimensionMismatch("data has $n_obs variables but the model was built for $(cycle_prior.n)"))

    n_draws = burnin + n_samples

    #posterior degrees pf freedom for trend covariance matrix
    #(trend innovations = diff of the n_time_steps + 1 trend states = n_time_steps)
    dτ_post = n_trend_time_steps - 1 + priors.trend_covariance_df

    #posterior degrees pf freedom for cycle covariance matrix
    #(cycle regressions = n_cycle_time_steps - p = n_time_steps)
    dc_post = n_cycle_time_steps - p + cycle_prior.d

    #cycle VAR prior translated to the sampler's oldest-lag-first, no-intercept layout
    #(MinnesotaPrior stores regressors as [lag1 … lagp, const]; reverse the lag blocks)
    Ω = vec(reverse(reshape(diag(cycle_prior.Ω)[1:k], n_obs, p), dims=2))
    Ω_inv = inv(Diagonal(Ω))

    cycle_coeff_mean = reshape(reverse(reshape(cycle_prior.Φ₀[1:k, :], n_obs, p, n_obs), dims=2), k, n_obs)

    #MinnesotaPrior: Σ ~ IW(Ψ, d), so Ψ is the prior scale; its mean is Ψ/(d-n-1)
    cycle_covariance_scale = Matrix(cycle_prior.Ψ)
    cycle_covariance_mean = cycle_covariance_scale / (cycle_prior.d - n_obs - 1)

    trend_covariance_scale = priors.trend_covariance_mean * (priors.trend_covariance_df + n_trends + 1)

    # Initial state mean/covariance for the cycle companion (length / order n_obs*p)
    initial_cycle_mean = repeat(priors.initial_cycle_mean, p)
    initial_cycle_covariance = kron(Matrix(I, p, p), cycle_covariance_mean)

    # Full initial-state distribution (trend block then cycle companion). The mean
    # is constant across draws; the cycle block of the covariance is re-initialised
    # from the implied stationary distribution after each parameter update below.
    initial_state_mean = [priors.initial_trend_mean; initial_cycle_mean]
    initial_state_covariance = [priors.initial_trend_covariance   zeros(n_trends, n_obs*p)
                                zeros(n_obs*p, n_trends)           initial_cycle_covariance]

    # Storage for sampled states and variables (states include the pre-sample)
    trends_states = zeros(n_draws, n_trend_time_steps, n_trends)
    cycle_states = zeros(n_draws, n_cycle_time_steps, n_obs)

    trend_covariance = zeros(n_draws, n_trends, n_trends)
    betas = zeros(n_draws, n_obs*k)
    sigmas = zeros(n_draws, n_obs, n_obs)

    #set initial parameters values to prior values
    trend_covariance[1, :, :] = priors.trend_covariance_mean
    # identity dynamics on the most recent lag (last predictor block), zero elsewhere
    betas[1, :] = vec([zeros(n_obs*(p-1), n_obs); Matrix(I(n_obs))])
    sigmas[1, :, :] = cycle_covariance_mean

    # The model carries the state space skeleton (constant structure, zero
    # draw-dependent blocks); set the initial (prior) parameter values here, then
    # refresh the draw-dependent blocks with newly drawn parameters at the end of
    # every draw.
    update_tc_var!(
                ssm,
                collect(reshape(betas[1, :], k, n_obs)'),
                trend_covariance[1, :, :],
                sigmas[1, :, :],
                n_trends,
                n_obs,
                p)


    for s in 2:n_draws

        trends_states[s,:,:], cycle_states[s,:,:] = sample_states(ssm, data, initial_state_mean, initial_state_covariance, n_trends, n_obs; p = p)

        trend_covariance[s, :, :] = rand(covariance_posterior(trends_states[s,:,:], trend_covariance_scale, dτ_post))

        betas[s,:], sigmas[s, :, :] = sample_var_params(cycle_states[s,:,:], p, cycle_coeff_mean, Ω_inv, cycle_covariance_scale, dc_post)

        # Update the model with the newly drawn parameters for the next iteration,
        # mutating only the blocks that change (transition cycle block, the two
        # covariance blocks, and the re-initialised cycle initial covariance)
        # instead of rebuilding the whole StateSpaceModel.
        update_tc_var!(
                    ssm,
                    collect(reshape(betas[s, :], k, n_obs)'),
                    trend_covariance[s, :, :],
                    sigmas[s, :, :],
                    n_trends,
                    n_obs,
                    p)

        # Re-initialise the cycle block of the initial covariance from the implied
        # stationary distribution of the just-updated VAR dynamics (the trend block
        # is kept fixed at its prior value). Previously done inside update_tc_var!.
        initial_state_covariance[n_trends+1:end, n_trends+1:end] =
            stationary_cycle_covariance(ssm, n_trends)

        logging && s % 2000 == 0 && println("Gibbs sampler: draw $s of $n_draws")

    end

    return build_result(model, trends_states, cycle_states, trend_covariance, betas, sigmas, burnin, thin)

end