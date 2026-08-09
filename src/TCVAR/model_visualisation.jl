
"""
Compute posterior mean and credible intervals from Carter-Kohn samples
"""
function compute_posterior_statistics(state_smoothed_samples::Array{Float64,3}; credible_level::Float64=0.90)
    n_samples, n_time_steps, n_states = size(state_smoothed_samples)
    
    # Posterior means
    state_smoothed_mean = mean(state_smoothed_samples, dims=1)[1, :, :]
    
    # Credible intervals
    alpha = (1 - credible_level) / 2
    lower_quantile = alpha
    upper_quantile = 1 - alpha
    
    state_smoothed_lower = zeros(n_time_steps, n_states)
    state_smoothed_upper = zeros(n_time_steps, n_states)
    
    for t in 1:n_time_steps
        for s in 1:n_states
            state_samples_ts = state_smoothed_samples[:, t, s]
            state_smoothed_lower[t, s] = quantile(state_samples_ts, lower_quantile)
            state_smoothed_upper[t, s] = quantile(state_samples_ts, upper_quantile)
        end
    end
    
    return state_smoothed_mean, state_smoothed_lower, state_smoothed_upper
end


function plot_variable_states(observations, states, titles)

    data = hcat(observations, states)

    plot(data; layout=(3,1), size=(800, 600), title=titles)

end

"""
    plot_states(result::TCVarResult; credible_level = 0.95)

Plot every trend and cycle state of a fitted [`TCVarResult`](@ref) in its own
subplot, stacked vertically. Each trend subplot is titled with the corresponding
entry of `result.model.trend_names`; each cycle subplot with the corresponding
entry of `result.model.variable_names`. Every subplot shows the posterior-mean
path with a shaded `credible_level` credible band over the retained Gibbs draws.
"""
function plot_states(result::TCVarResult; credible_level = 0.90)

    trend_names = result.model.trend_names
    variable_names = result.model.variable_names

    trend_mean, trend_lower, trend_upper =
        compute_posterior_statistics(result.trend_states; credible_level)
    cycle_mean, cycle_lower, cycle_upper =
        compute_posterior_statistics(result.cycle_states; credible_level)

    plots = Plots.Plot[]

    for (i, name) in enumerate(trend_names)
        ribbon = (trend_mean[:, i] .- trend_lower[:, i], trend_upper[:, i] .- trend_mean[:, i])
        push!(plots, plot(trend_mean[:, i]; ribbon, title = "Trend $name", legend = false))
    end

    for (i, name) in enumerate(variable_names)
        ribbon = (cycle_mean[:, i] .- cycle_lower[:, i], cycle_upper[:, i] .- cycle_mean[:, i])
        push!(plots, plot(cycle_mean[:, i]; ribbon, title = "Cycle $name", legend = false))
    end

    plot(plots...; layout = (length(plots), 1), size = (800, 250 * length(plots)))

end
