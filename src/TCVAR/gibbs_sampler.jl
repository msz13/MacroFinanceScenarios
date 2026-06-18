function covariance_posterior(data, scale_prior, d_posterior)
    
    res = diff(data, dims=1)
    posterior_mean = res' * res .+ scale_prior

    return InverseWishart(d_posterior, posterior_mean)    

end

coeff_names(n_coeffs) = ["β$i" for i in 1:n_coeffs]
cycle_covariance_names(n_variables) = ["Σc$i" for i in 1:n_variables*n_variables]
trend_covariance_names(n_variables) = ["Στ$i" for i in 1:n_variables*n_variables]



"""
    data: T x m matrix of observations
    trend_mapping: m x n_trends mapping of trends to observations
    priors: NamedTuple of trend / initial-state priors with fields
        `initial_trend_mean`, `initial_cycle_mean`, `initial_trend_covariance`,
        `trend_covariance_df`, `trend_covariance_mean`.
    cycle_prior::MinnesotaPrior: Minnesota prior for the cycle VAR. Its `n` must
        equal the number of observed variables and its `p` sets the number of
        cycle VAR lags.
    n_samples: number of retained samples
    burnin: number of samples to discard
    thin: keep every `thin`-th sample

The cycle is mean-zero, so the prior's intercept row is dropped. `MinnesotaPrior`
orders its regressors newest-lag-first (`[lag1 … lagp, const]`), whereas the
state-space sampler stacks predictors oldest-lag-first (see `prepare_var_data`),
so the lag blocks of `Φ₀` and `Ω` are reversed here. Following the
Giannone–Lenza–Primiceri parameterization `Σ ~ IW(Ψ, d)`, the prior IW scale is
`Ψ` directly.
"""

function gibbs_sampler(data, trend_mapping, priors, cycle_prior::MinnesotaPrior; burnin = 1000, n_samples=1000, thin=1, logging=false)

    n_time_steps, n_obs = size(data)
    n_trends = size(trend_mapping, 2)
    p = cycle_prior.p
    k = n_obs * p #number of var coefficients per equation (lags stacked)

    # sample_states prepends the drawn initial state (t = 0) to the t = 1..T
    # smoothed states, so the sampled state paths span n_time_steps + 1 points.
    n_states_time_steps = n_time_steps + 1

    n_obs == cycle_prior.n ||
        throw(DimensionMismatch("cycle_prior.n = $(cycle_prior.n) must match the number of observed variables $n_obs"))

    n_draws = burnin + n_samples

    #posterior degrees pf freedom for trend covariance matrix
    dτ_post = n_states_time_steps - p + priors.trend_covariance_df

    #posterior degrees pf freedom for cycle covariance matrix
    dc_post = n_states_time_steps - p + cycle_prior.d

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

    # Storage for sampled states and variables (states include t = 0)
    trends_states = zeros(n_draws, n_states_time_steps, n_trends)
    cycle_states = zeros(n_draws, n_states_time_steps, n_obs)

    trend_covariance = zeros(n_draws, n_trends, n_trends)
    betas = zeros(n_draws, n_obs*k)
    sigmas = zeros(n_draws, n_obs, n_obs)

    #set initial parameters values to prior values
    trend_covariance[1, :, :] = priors.trend_covariance_mean
    # identity dynamics on the most recent lag (last predictor block), zero elsewhere
    betas[1, :] = vec([zeros(n_obs*(p-1), n_obs); Matrix(I(n_obs))])
    sigmas[1, :, :] = cycle_covariance_mean


    for s in 2:n_draws

        trends_states[s,:,:], cycle_states[s,:,:] = sample_states(
                                       data,
                                       trend_mapping,
                                       collect(reshape(betas[s-1, :], k, n_obs)'),
                                       trend_covariance[s-1,:,:],
                                       sigmas[s-1,:,:],
                                       priors.initial_trend_mean,
                                       initial_cycle_mean,
                                       priors.initial_trend_covariance,
                                       initial_cycle_covariance;
                                       p = p)

        trend_covariance[s, :, :] = rand(covariance_posterior(trends_states[s,:,:], trend_covariance_scale, dτ_post))

        betas[s,:], sigmas[s, :, :] = sample_var_params(cycle_states[s,:,:], p, cycle_coeff_mean, Ω_inv, cycle_covariance_scale, dc_post)

        logging && s % 1000 == 0 && @info "Gibbs sampler: draw $s of $n_draws"

    end

    t_trends_states = trends_states[burnin+1:thin:end, :, :]
    t_cycle_states =  cycle_states[burnin+1:thin:end, :, :]
    t_trend_covariance = Chains(reshape(trend_covariance[burnin+1:thin:end,:,:], n_samples÷thin, n_trends*n_trends, 1), trend_covariance_names(n_trends))
    t_betas = Chains(betas[burnin+1:thin:end,:,:], coeff_names(n_obs*k))
    t_sigmas = Chains(reshape(sigmas[burnin+1:thin:end,:,:], n_samples÷thin, n_obs*n_obs, 1), cycle_covariance_names(n_obs))

    return t_trends_states, t_cycle_states, t_trend_covariance, t_betas, t_sigmas

end