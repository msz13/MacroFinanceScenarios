"""
State Space Model structure
x_t = F * x_{t-1} + G * u_t    (state equation)
y_t = H * x_t + v_t            (observation equation)

where:
- x_t is the state vector at time t
- y_t is the observation vector at time t
- u_t ~ N(0, Q) is the state noise
- v_t ~ N(0, H) is the observation noise
- T is the state transition matrix
- R is the state noise coefficient matrix
- Z is the observation matrix
"""

struct StateSpaceModel
    T::Matrix{Float64}  # State transition matrix
    R::Matrix{Float64}  # State noise coefficient matrix
    Z::Matrix{Float64}  # Observation matrix
    Q::Matrix{Float64}  # State noise covariance
    H::Matrix{Float64}  # Observation noise covariance
end

function sample(model:: StateSpaceModel, initial_state_mean, initial_state_covariance, n_steps)

    initial_states = rand(MvNormal(initial_state_mean, initial_state_covariance))

    return sample(model, initial_states, n_steps)

end

function sample(model:: StateSpaceModel, initial_state, n_steps)

    n_variables, n_states = size(model.Z)
    states = zeros(n_steps, n_states)
    obs = zeros(n_steps, n_variables)


    states[1, :] = initial_state
    obs[1, :] = model.Z * states[1,:] .+ rand(MvNormal(zeros(n_variables), model.H))

    for t in 2:n_steps
        states[t,:] = model.T * states[t-1,:] + rand(MvNormal(zeros(n_states), model.Q + I(n_states) .* 1e-4))
        obs[t, :] = model.Z * states[t,:] + rand(MvNormal(zeros(n_variables), model.H + I(n_variables) .* 1e-4))
    end

    return states, obs

end
