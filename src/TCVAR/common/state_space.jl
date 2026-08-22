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

"""
    sample(model, initial_state_mean, initial_state_covariance, n_steps; jitter = 1e-4)
    sample(model, initial_state, n_steps; jitter = 1e-4) -> (states, obs)

Simulate `n_steps` periods forward from the state space model, drawing the starting
state from `N(initial_state_mean, initial_state_covariance)` in the first form and
taking it as given in the second.

The first row is the starting state itself — `states[1, :] = initial_state`, with only
the observation drawn — so the path contains `n_steps - 1` transitions. Both noise
covariances are reached through [`process_noise`](@ref) / [`observation_noise`](@ref),
so a [`TimeVaryingStateSpaceModel`](@ref) simulates with its own `Q_t` and `H_t` at
every period; `states[t, :]` is drawn with `Q_t`, which leaves `Q_1` unused.

Noise is drawn as `psd_factor(Σ + jitter*I) * randn(n)`. The default `jitter = 1e-4` is
added to both covariances from the second period on, so that a singular `Q` — the
trend-cycle companion is full of exactly-zero rows — still factors; it also means a
deliberately switched-off noise block still draws with a standard deviation of `1e-2`.
Pass `jitter = 0` when "off" has to mean exactly off: [`psd_factor`](@ref) keeps the exact
null space of a rank-deficient covariance and turns a zero covariance into a deterministic
zero draw, which is what [`simulate_volatility_path`](@ref) needs.
"""
function sample(model::AbstractStateSpaceModel, initial_state_mean, initial_state_covariance,
                n_steps; jitter::Real = 1e-4)

    initial_states = rand(MvNormal(initial_state_mean, initial_state_covariance))

    return sample(model, initial_states, n_steps; jitter = jitter)

end

"""Draw one `N(0, covariance + jitter*I)` noise vector of length `n`. For a
positive-definite covariance this is the Cholesky factor applied to a `randn` vector, i.e.
exactly what `rand(MvNormal(covariance))` draws."""
noise_draw(covariance::AbstractMatrix, n::Int, jitter::Real) =
    psd_factor(covariance + jitter * I) * randn(n)

function sample(model::AbstractStateSpaceModel, initial_state, n_steps; jitter::Real = 1e-4)

    n_variables, n_states = size(model.Z)
    states = zeros(n_steps, n_states)
    obs = zeros(n_steps, n_variables)


    states[1, :] = initial_state
    obs[1, :] = model.Z * states[1,:] + noise_draw(observation_noise(model, 1), n_variables, 0)

    for t in 2:n_steps
        states[t,:] = model.T * states[t-1,:] + noise_draw(process_noise(model, t), n_states, jitter)
        obs[t, :] = model.Z * states[t,:] + noise_draw(observation_noise(model, t), n_variables, jitter)
    end

    return states, obs

end
