"""
Kalman Filter implementation
Returns filtered states, covariances, predicted states, and predicted covariances
"""
function kalman_filter(model::StateSpaceModel, observations::Matrix{Union{Missing, Float64}}, initial_state_mean::Vector{Float64}, initial_state_covariance::Matrix{Float64})
    n_time_steps, n_obs = size(observations)
    n_states = size(model.T, 1)

    # Storage for results
    state_filtered = zeros(n_time_steps, n_states)
    covariance_filtered = zeros(n_time_steps, n_states, n_states)
    state_predicted = zeros(n_time_steps, n_states)
    covariance_predicted = zeros(n_time_steps, n_states, n_states)
    log_likelihood = 0.0

    # Initialize
    state_current = initial_state_mean
    covariance_current = initial_state_covariance

    # Additive process noise R*Q*R' is constant across time; compute it once.
    RQR = model.R * model.Q * model.R'

    for t in 1:n_time_steps
        # Prediction step
        if t == 1
            state_predicted_t = model.T * state_current
            covariance_predicted_t = model.T * covariance_current * model.T' + RQR
        else
            state_predicted_t = model.T * state_filtered[t-1, :]
            covariance_predicted_t = model.T * reshape(covariance_filtered[t-1, :, :], n_states, n_states) * model.T' + RQR
        end

        state_predicted[t, :] = state_predicted_t
        covariance_predicted[t, :, :] = covariance_predicted_t

        # Update step using only the observed (non-missing) series at time t
        y_t = observations[t, :]
        obs_idx = findall(!ismissing, y_t)
        if !isempty(obs_idx)
            # Subset observation equation to the available series
            y = Float64.(y_t[obs_idx])
            Z_t = model.Z[obs_idx, :]
            H_t = model.H[obs_idx, obs_idx]

            # Innovation
            innovation = y - Z_t * state_predicted_t
            innovation_covariance = Z_t * covariance_predicted_t * Z_t' + H_t

            # Kalman gain via a Cholesky solve (P Z' S⁻¹) instead of pinv(S).
            S = chol_psd(innovation_covariance)
            kalman_gain = (covariance_predicted_t * Z_t') / S

            # Filtered state and covariance (Joseph form for numerical stability)
            state_filtered[t, :] = state_predicted_t + kalman_gain * innovation
            IKZ = I - kalman_gain * Z_t
            covariance_filtered[t, :, :] = IKZ * covariance_predicted_t * IKZ' + kalman_gain * H_t * kalman_gain'

            # Log-likelihood contribution TODO protect negative values
            log_likelihood += 0. #-0.5 * (log(det(innovation_covariance)) + innovation' * inv(innovation_covariance) * innovation)
        else
            # No observation available this period
            state_filtered[t, :] = state_predicted_t
            covariance_filtered[t, :, :] = covariance_predicted_t
        end
    end

    return state_filtered, covariance_filtered, state_predicted, covariance_predicted, log_likelihood
end
