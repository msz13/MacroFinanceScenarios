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
    priors: Named tuples of priors
    p: number of VAR lags for the cycle (default 1)
    n_samples: number of samples
    burnin: numper of samples to discart
    thin: skip nth sample

When `p > 1`, `priors.cycle_coeff_mean` must be sized (n_obs*p) x n_obs (the prior
mean of the regression coefficients `B` for `Y = X·B`, predictors stacked
oldest-lag-first). The coefficient prior variance is a Minnesota-style diagonal that
scales as λ²/(l²·σ_j²) for lag `l`, which reduces to the original λ²/σ_j² when p = 1.
"""

function gibbs_sampler(data, trend_mapping, priors; p::Int = 1, burnin = 1000, n_samples=1000, thin=1, logging=false)

    n_time_steps, n_obs = size(data)
    n_trends = size(trend_mapping, 2)
    k = n_obs * p #number of var coefficients per equation (lags stacked)

    n_draws = burnin + n_samples

    #posterior degrees pf freedom for trend covariance matrix
    dτ_post = n_time_steps - p + priors.trend_covariance_df

    #posterior degrees pf freedom for cycle covariance matrix
    dc_post = n_time_steps - p + priors.cycle_covariance_df

    #prior variance of cycle coefficients (Minnesota-style, with lag decay λ²/(l²σ²))
    #predictors are ordered oldest-lag-first, so block b corresponds to lag l = p-b+1
    λ = priors.cycle_coeff_shrinkage_param
    σ2 = diag(priors.cycle_covariance_mean)
    Ω = zeros(k)
    for b in 1:p
        l = p - b + 1
        Ω[(b-1)*n_obs+1 : b*n_obs] = λ^2 ./ (l^2 .* σ2)
    end
    Ω_inv = inv(Diagonal(Ω))

    trend_covariance_scale = priors.trend_covariance_mean * (priors.trend_covariance_df + n_trends + 1)
    cycle_covariance_scale = priors.cycle_covariance_mean * (priors.cycle_covariance_df + n_obs + 1)

    # Initial state mean/covariance for the cycle companion (length / order n_obs*p)
    initial_cycle_mean = repeat(priors.initial_cycle_mean, p)
    initial_cycle_covariance = kron(Matrix(I, p, p), Matrix(priors.cycle_covariance_mean))

    # Storage for sampled states and variables
    trends_states = zeros(n_draws, n_time_steps, n_trends)
    cycle_states = zeros(n_draws, n_time_steps, n_obs)

    trend_covariance = zeros(n_draws, n_trends, n_trends)
    betas = zeros(n_draws, n_obs*k)
    sigmas = zeros(n_draws, n_obs, n_obs)

    #set initial parameters values to prior values
    trend_covariance[1, :, :] = priors.trend_covariance_mean
    # identity dynamics on the most recent lag (last predictor block), zero elsewhere
    betas[1, :] = vec([zeros(n_obs*(p-1), n_obs); Matrix(I(n_obs))])
    sigmas[1, :, :] = priors.cycle_covariance_mean


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

        betas[s,:], sigmas[s, :, :] = sample_var_params(cycle_states[s,:,:], p, priors.cycle_coeff_mean, Ω_inv, cycle_covariance_scale, dc_post)

        logging && s % 1000 == 0 && @info "Gibbs sampler: draw $s of $n_draws"

    end

    t_trends_states = trends_states[burnin+1:thin:end, :, :]
    t_cycle_states =  cycle_states[burnin+1:thin:end, :, :]
    t_trend_covariance = Chains(reshape(trend_covariance[burnin+1:thin:end,:,:], n_samples÷thin, n_trends*n_trends, 1), trend_covariance_names(n_trends))
    t_betas = Chains(betas[burnin+1:thin:end,:,:], coeff_names(n_obs*k))
    t_sigmas = Chains(reshape(sigmas[burnin+1:thin:end,:,:], n_samples÷thin, n_obs*n_obs, 1), cycle_covariance_names(n_obs))

    return t_trends_states, t_cycle_states, t_trend_covariance, t_betas, t_sigmas

end