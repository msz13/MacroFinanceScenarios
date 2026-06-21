using LinearAlgebra
using Random

"""
Returns the matrix square root L of a symmetric matrix A via eigendecomposition,
such that L * L' ≈ A. Use as: mean + eigen_sqrt(cov) * randn(n)
"""
function eigen_sqrt(A::AbstractMatrix)
    A_sym = Hermitian((A + A') / 2)
    vals, vecs = eigen(A_sym)
    return vecs * Diagonal(sqrt.(max.(vals, 0.0)))
end

"""
Kalman Filter implementation
Returns filtered states, covariances, predicted states, and predicted covariances
"""
function kalman_filter(model::StateSpaceModel, observations::Matrix{Union{Missing, Float64}})
    n_time_steps, n_obs = size(observations)
    n_states = size(model.T, 1)
    
    # Storage for results
    state_filtered = zeros(n_time_steps, n_states)
    covariance_filtered = zeros(n_time_steps, n_states, n_states)
    state_predicted = zeros(n_time_steps, n_states)
    covariance_predicted = zeros(n_time_steps, n_states, n_states)
    log_likelihood = 0.0
    
    # Initialize
    state_current = model.initial_state_mean
    covariance_current = model.initial_state_covariance
    
    for t in 1:n_time_steps
        # Prediction step
        if t == 1
            state_predicted_t = model.T * state_current
            covariance_predicted_t = model.T * covariance_current * model.T' + model.R * model.Q * model.R'
        else
            state_predicted_t = model.T * state_filtered[t-1, :]
            covariance_predicted_t = model.T * reshape(covariance_filtered[t-1, :, :], n_states, n_states) * model.T' + model.R * model.Q * model.R'
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

            # Kalman gain
            kalman_gain = covariance_predicted_t * Z_t' * pinv(innovation_covariance)

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

"""
Carter-Kohn Algorithm for sampling smoothed states
This algorithm samples from the joint posterior distribution of all states
given all observations using backward simulation
"""
function carter_kohn_sampler2(model::StateSpaceModel, observations::Matrix{Union{Missing,Float64}}; n_samples::Int=1000)
    n_time_steps, n_obs = size(observations)
    n_states = size(model.T, 1)
    
    # Run Kalman filter forward pass
    state_filtered, covariance_filtered, state_predicted, covariance_predicted, _ = 
        kalman_filter(model, observations)
    
    # Storage for sampled states
    state_smoothed_samples = zeros(n_samples, n_time_steps, n_states)
    
    for sample_idx in 1:n_samples
        state_smoothed_current = zeros(n_time_steps, n_states)
        
        # Sample final state from filtered distribution at T
        final_state_mean = state_filtered[end, :]
        final_state_covariance = covariance_filtered[end, :, :]
        state_smoothed_current[end, :] = final_state_mean + eigen_sqrt(final_state_covariance) * randn(n_states)
        
        # Backward pass: sample states from T-1 down to 1
        for t in (n_time_steps-1):-1:1
            # Get filtered estimates at time t
            state_filtered_t = state_filtered[t, :]
            covariance_filtered_t = covariance_filtered[t, :, :]
            
            # Get predicted estimates at time t+1
            state_predicted_t_plus_1 = state_predicted[t+1, :]
            covariance_predicted_t_plus_1 = covariance_predicted[t+1, :, :]
            
            # Compute smoothing gain matrix
            smoothing_gain = covariance_filtered_t * model.T' * pinv(covariance_predicted_t_plus_1)
            
            # Conditional mean and covariance for state at time t given state at t+1
            state_smoothed_mean = state_filtered_t + 
                smoothing_gain * (state_smoothed_current[t+1, :] - state_predicted_t_plus_1)
            
            covariance_smoothed = covariance_filtered_t - 
                smoothing_gain * model.T*covariance_filtered_t
            
            # Ensure covariance is positive definite
            covariance_smoothed = (covariance_smoothed + covariance_smoothed') / 2
            covariance_smoothed += 1e-10 * I
            
            # Sample state at time t
            state_smoothed_current[t, :] = state_smoothed_mean + eigen_sqrt(covariance_smoothed) * randn(n_states)
        end

        state_smoothed_samples[sample_idx, :, :] = state_smoothed_current
    end
    
    return state_smoothed_samples
end

"""
Carter-Kohn Algorithm for sampling smoothed states
This algorithm samples from the joint posterior distribution of all states
given all observations using backward simulation
"""
function carter_kohn_sampler(model::StateSpaceModel, observations::Matrix{Union{Missing, Float64}})
    n_time_steps, n_obs = size(observations)
    n_states = size(model.T, 1)
    
    # Run Kalman filter forward pass
    state_filtered, covariance_filtered, state_predicted, covariance_predicted, _ = 
        kalman_filter(model, observations)
                
    state_smoothed_current = zeros(n_time_steps, n_states)
        
    # Sample final state from filtered distribution at T
    final_state_mean = state_filtered[end, :]
    final_state_covariance = covariance_filtered[end, :, :]
    state_smoothed_current[end, :] = final_state_mean + eigen_sqrt(final_state_covariance) * randn(n_states)
        
    # Backward pass: sample states from T-1 down to 1
    for t in (n_time_steps-1):-1:1
        # Get filtered estimates at time t
        state_filtered_t = state_filtered[t, :]
        covariance_filtered_t = covariance_filtered[t, :, :]
           
        # Get predicted estimates at time t+1
        state_predicted_t_plus_1 = state_predicted[t+1, :]
        covariance_predicted_t_plus_1 = covariance_predicted[t+1, :, :]
            
        # Compute smoothing gain matrix
        smoothing_gain = covariance_filtered_t * model.T' * pinv(covariance_predicted_t_plus_1)

        # Conditional mean and covariance for state at time t given state at t+1
        state_smoothed_mean = state_filtered_t +
            smoothing_gain * (state_smoothed_current[t+1, :] - state_predicted_t_plus_1)

        covariance_smoothed = covariance_filtered_t - smoothing_gain * model.T * covariance_filtered_t

        # Sample state at time t
        state_smoothed_current[t, :] = state_smoothed_mean + eigen_sqrt(covariance_smoothed) * randn(n_states)
    end

    # Draw the initial state (t = 0) conditional on the sampled state at t = 1,
    # using the prior moments (initial_state_mean / covariance) as the "filtered"
    # estimate at t = 0 and the predicted moments at t = 1.
    initial_smoothing_gain = model.initial_state_covariance * model.T' * pinv(covariance_predicted[1, :, :])

    initial_state_mean = model.initial_state_mean +
        initial_smoothing_gain * (state_smoothed_current[1, :] - state_predicted[1, :])

    initial_state_covariance = model.initial_state_covariance -
        initial_smoothing_gain * model.T * model.initial_state_covariance

    initial_state = initial_state_mean + eigen_sqrt(initial_state_covariance) * randn(n_states)

    return initial_state, state_smoothed_current

end



#TODO sprawdizc cze reshape beta jest dobre

function sample_states(model::StateSpaceModel, data, n_trends, n_observations; p::Int = 1)

        initial_state, state_smoothed_samples = carter_kohn_sampler(model, data)

        # Cycle companion is ordered oldest-lag-first; the contemporaneous cycle
        # c_t is the last block of size n_observations.
        cycle_start = n_trends + n_observations * (p - 1) + 1

        # The drawn initial state (t = 0) carries the full cycle companion, i.e. the
        # p initial cycle periods c_{-p+1}, ..., c_0 stacked oldest-lag-first. Reshape
        # them into p chronologically ordered rows so the whole pre-sample is prepended.
        initial_cycle = reshape(
            initial_state[n_trends+1:n_trends+n_observations*p],
            n_observations, p
        )'

        # Stack the drawn initial state (t = 0) on top of the sampled t = 1..T states.
        trends_states = vcat(
            initial_state[1:n_trends]',
            state_smoothed_samples[:, 1:n_trends]
        )
        cycle_states = vcat(
            initial_cycle,
            state_smoothed_samples[:, cycle_start:cycle_start+n_observations-1]
        )

        return trends_states, cycle_states

end

