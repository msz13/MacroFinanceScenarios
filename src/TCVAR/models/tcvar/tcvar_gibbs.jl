"""
    model::TCVAR: trend-cycle VAR model bundling the state-space skeleton, the
        distribution-keyed priors NamedTuple (`initial_trend`, `initial_cycle`,
        `trend_covariance`, `cycle_covariance`, `cycle_β`) and the variable /
        trend names (see [`TCVAR`](@ref)).
    data: T x m matrix of observations
    n_samples: number of retained samples
    burnin: number of samples to discard
    thin: keep every `thin`-th sample

Returns a [`TCVarResult`](@ref) bundling the (skeleton-reset) model, the
parameter draws as one `FlexiChain` and the sampled trend / cycle states.

All prior quantities are read straight off the distributions in `model.priors`:
the two `InverseWishart` scales are taken at face value (`Σ ~ IW(Ψ, d)`, the
Giannone–Lenza–Primiceri parameterization), and the cycle coefficient prior is
re-ordered from the `MinnesotaPrior` newest-lag-first layout to the
oldest-lag-first one used by `prepare_var_data` by [`prior_var_coeff`](@ref) and
[`prior_row_covariance`](@ref), which also drop the intercept row (the cycle is
mean-zero).
"""

function gibbs_sampler(model::TCVAR, data; burnin = 1000, n_samples=1000, thin=1, logging=false)

    priors = model.priors
    β_prior = priors.cycle_β
    ssm = model.ssm

    n_time_steps, n_obs = size(data)
    n_trends = length(model.trend_names)
    p = β_prior.p
    k = n_obs * p #number of var coefficients per equation (lags stacked)

    # sample_states prepends the drawn pre-sample to the t = 1..T smoothed states.
    # Trends are a random walk, so only the single initial state (t = 0) is
    # prepended -> n_time_steps + 1 points. The cycle companion carries p
    # pre-sample periods (c_{-p+1}, ..., c_0), so the cycle path spans
    # n_time_steps + p points.
    n_trend_time_steps = n_time_steps + 1
    n_cycle_time_steps = n_time_steps + p

    n_obs == β_prior.n ||
        throw(DimensionMismatch("data has $n_obs variables but the model was built for $(β_prior.n)"))

    n_draws = burnin + n_samples

    # --- prior quantities read straight off the distributions ---
    ντ, Ψτ = params(priors.trend_covariance)   # (degrees of freedom, PDMat scale)
    νc, Ψc = params(priors.cycle_covariance)

    #posterior degrees pf freedom for trend covariance matrix
    #(trend innovations = diff of the n_time_steps + 1 trend states = n_time_steps)
    dτ_post = n_trend_time_steps - 1 + ντ

    #posterior degrees pf freedom for cycle covariance matrix
    #(cycle regressions = n_cycle_time_steps - p = n_time_steps)
    dc_post = n_cycle_time_steps - p + νc

    #IW scales are used as written; the cycle prior mean is Ψc/(νc-n-1)
    trend_covariance_scale = Matrix(Ψτ)
    cycle_covariance_scale = Matrix(Ψc)
    cycle_covariance_mean  = mean(priors.cycle_covariance)

    #cycle VAR prior in the sampler's oldest-lag-first, no-intercept layout
    Ω_inv            = inv(prior_row_covariance(β_prior))
    cycle_coeff_mean = collect(prior_var_coeff(β_prior)')   # k × n_obs

    # Initial state mean/covariance for the cycle companion (length / order n_obs*p)
    initial_cycle_mean = mean(priors.initial_cycle)
    initial_cycle_covariance = Matrix(cov(priors.initial_cycle))

    # Full initial-state distribution (trend block then cycle companion). The mean
    # is constant across draws; the cycle block of the covariance is re-initialised
    # from the implied stationary distribution after each parameter update below.
    initial_state_mean = [mean(priors.initial_trend); initial_cycle_mean]
    initial_state_covariance = [Matrix(cov(priors.initial_trend))  zeros(n_trends, k)
                                zeros(k, n_trends)                 initial_cycle_covariance]

    # Storage for sampled states and variables (states include the pre-sample)
    trends_states = zeros(n_draws, n_trend_time_steps, n_trends)
    cycle_states = zeros(n_draws, n_cycle_time_steps, n_obs)

    trend_covariance = zeros(n_draws, n_trends, n_trends)
    betas = zeros(n_draws, n_obs*k)
    sigmas = zeros(n_draws, n_obs, n_obs)

    #set initial parameters values to prior values
    trend_covariance[1, :, :] = mean(priors.trend_covariance)
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

        trend_covariance[s, :, :] = rand(random_walk_covariance_posterior(trends_states[s,:,:], trend_covariance_scale, dτ_post))

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