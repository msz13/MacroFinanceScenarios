"""
    AbstractStateSpaceModel

Linear Gaussian state space model

    x_t = T * x_{t-1} + R * u_t    (state equation)
    y_t = Z * x_t + v_t            (observation equation)

where:
- `x_t` is the state vector at time t
- `y_t` is the observation vector at time t
- `u_t ~ N(0, Q)` is the state noise
- `v_t ~ N(0, H)` is the observation noise
- `T` is the state transition matrix
- `R` is the state noise coefficient matrix
- `Z` is the observation matrix

Two concrete models share this interface: [`StateSpaceModel`](@ref), whose noise
covariances are constant, and [`TimeVaryingStateSpaceModel`](@ref), whose `Q` and/or
`H` change every period. Filtering and smoothing reach the two covariances only
through [`process_noise`](@ref) and [`observation_noise`](@ref), so they are written
once against this abstract type.
"""
abstract type AbstractStateSpaceModel end

"""
    StateSpaceModel(T, R, Z, Q, H)

State space model with constant noise covariances (see
[`AbstractStateSpaceModel`](@ref) for the equations).
"""
struct StateSpaceModel <: AbstractStateSpaceModel
    T::Matrix{Float64}  # State transition matrix
    R::Matrix{Float64}  # State noise coefficient matrix
    Z::Matrix{Float64}  # Observation matrix
    Q::Matrix{Float64}  # State noise covariance
    H::Matrix{Float64}  # Observation noise covariance
end

"""
    NoiseCovariance

A noise covariance that is either constant (an `n × n` matrix) or time varying (an
`n_time × n × n` array, first axis indexed by period).
"""
const NoiseCovariance = Union{Matrix{Float64}, Array{Float64,3}}

"""
    TimeVaryingStateSpaceModel(T, R, Z, Q, H)

State space model whose noise covariances may vary with time (see
[`AbstractStateSpaceModel`](@ref) for the equations). `Q` and `H` are each given
either as a constant matrix or as an `n_time × n × n` array whose slice `[t, :, :]`
is the covariance used at period `t`, so a model needs a 3-D array only for the
covariance that actually moves:

* the TCVAR-SV state draw has a time-varying `Q_t = blockdiag(Στ, 0, Σ_t)` and the
  constant observation noise of the trend-cycle skeleton;
* the stochastic-volatility block has the constant `Q = Ω` of the log-volatility
  VAR(1) and a time-varying `H_t = diag(v²_{s_t})` from the mixture indicators.
"""
struct TimeVaryingStateSpaceModel <: AbstractStateSpaceModel
    T::Matrix{Float64}  # State transition matrix
    R::Matrix{Float64}  # State noise coefficient matrix
    Z::Matrix{Float64}  # Observation matrix
    Q::NoiseCovariance  # State noise covariance, constant or n_time × n_states × n_states
    H::NoiseCovariance  # Observation noise covariance, constant or n_time × n_obs × n_obs
end

noise_at(covariance::Matrix{Float64}, ::Int) = covariance
noise_at(covariance::Array{Float64,3}, t::Int) = @view covariance[t, :, :]

"""
    process_noise(model, t) -> AbstractMatrix

State-noise covariance `Q` used by the state equation at period `t`.
"""
process_noise(model::StateSpaceModel, ::Int) = model.Q
process_noise(model::TimeVaryingStateSpaceModel, t::Int) = noise_at(model.Q, t)

"""
    observation_noise(model, t) -> AbstractMatrix

Observation-noise covariance `H` used by the observation equation at period `t`.
"""
observation_noise(model::StateSpaceModel, ::Int) = model.H
observation_noise(model::TimeVaryingStateSpaceModel, t::Int) = noise_at(model.H, t)

"""
    constant_state_noise(model) -> Matrix or nothing

The additive process noise `R*Q*R'` when it does not depend on `t`, and `nothing`
when it does. [`kalman_filter`](@ref) calls this once before its loop, so a
constant-`Q` model keeps computing the product exactly once per filter pass (and
reuses the very same matrix in every period, leaving its arithmetic bit-for-bit
what it was before the time-varying seam existed).
"""
constant_state_noise(model::StateSpaceModel) = model.R * model.Q * model.R'
constant_state_noise(model::TimeVaryingStateSpaceModel) =
    model.Q isa Matrix{Float64} ? model.R * model.Q * model.R' : nothing

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
