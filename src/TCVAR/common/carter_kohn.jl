"""
Carter-Kohn Algorithm for sampling smoothed states
This algorithm samples from the joint posterior distribution of all states
given all observations using backward simulation
"""
function carter_kohn_sampler(model::StateSpaceModel, observations::Matrix{Union{Missing, Float64}}, initial_state_mean, initial_state_covariance)
    n_time_steps, n_obs = size(observations)
    n_states = size(model.T, 1)

    # Run Kalman filter forward pass
    state_filtered, covariance_filtered, state_predicted, covariance_predicted, _ =
        kalman_filter(model, observations, initial_state_mean, initial_state_covariance)

    state_smoothed_current = zeros(n_time_steps, n_states)

    # Sample final state from filtered distribution at T
    final_state_mean = state_filtered[end, :]
    final_state_covariance = covariance_filtered[end, :, :]
    state_smoothed_current[end, :] = sample_mvn(final_state_mean, final_state_covariance)

    # Backward pass: sample states from T-1 down to 1
    for t in (n_time_steps-1):-1:1
        # Get filtered estimates at time t
        state_filtered_t = state_filtered[t, :]
        covariance_filtered_t = covariance_filtered[t, :, :]

        # Get predicted estimates at time t+1
        state_predicted_t_plus_1 = state_predicted[t+1, :]
        covariance_predicted_t_plus_1 = covariance_predicted[t+1, :, :]

        # Compute smoothing gain matrix via a Cholesky solve instead of pinv.
        smoothing_gain = (covariance_filtered_t * model.T') / chol_psd(covariance_predicted_t_plus_1)

        # Conditional mean and covariance for state at time t given state at t+1
        state_smoothed_mean = state_filtered_t +
            smoothing_gain * (state_smoothed_current[t+1, :] - state_predicted_t_plus_1)

        covariance_smoothed = covariance_filtered_t - smoothing_gain * model.T * covariance_filtered_t

        # Sample state at time t
        state_smoothed_current[t, :] = sample_mvn(state_smoothed_mean, covariance_smoothed)
    end

    # Draw the initial state (t = 0) conditional on the sampled state at t = 1,
    # using the prior moments (initial_state_mean / covariance) as the "filtered"
    # estimate at t = 0 and the predicted moments at t = 1.
    initial_smoothing_gain = (initial_state_covariance * model.T') / chol_psd(covariance_predicted[1, :, :])

    initial_smoothed_mean = initial_state_mean +
        initial_smoothing_gain * (state_smoothed_current[1, :] - state_predicted[1, :])

    initial_smoothed_covariance = initial_state_covariance -
        initial_smoothing_gain * model.T * initial_state_covariance

    initial_state = sample_mvn(initial_smoothed_mean, initial_smoothed_covariance)

    return initial_state, state_smoothed_current

end
