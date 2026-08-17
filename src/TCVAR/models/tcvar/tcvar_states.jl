#TODO sprawdizc cze reshape beta jest dobre

function sample_states(model::StateSpaceModel, data, initial_state_mean, initial_state_covariance, n_trends, n_observations; p::Int = 1)

        initial_state, state_smoothed_samples = carter_kohn_sampler(model, data, initial_state_mean, initial_state_covariance)

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
