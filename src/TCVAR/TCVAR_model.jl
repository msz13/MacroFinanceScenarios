


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
    initial_state_mean::Vector{Float64}
    initial_state_covariance::Matrix{Float64}


end


"""
    tc_var(trend_mapping, var_coeff, trend_cov, cycle_cov, ...; p = 1)

Build the state space representation of a trend-cycle model whose cycle follows a
VAR(`p`). The cycle is represented in companion form, ordered oldest-lag-first:

    ξ_t = [c_{t-p+1}; …; c_{t-1}; c_t]   (length n_variables * p)

so that `var_coeff` (size n_variables × n_variables*p) is the companion bottom block
[A_p … A_1] matching that ordering, i.e. the rows of the regression `Y = X·B` with
predictors stacked oldest-lag-first (`var_coeff == B'`). The contemporaneous cycle
`c_t` is the last block of ξ_t.

`initial_cycle_mean` / `initial_cycle_covariance` must be sized for the full
companion state (length / order n_variables * p). For `p == 1` this reduces to the
original VAR(1) model.
"""
function tc_var(trend_mapping, var_coeff, trend_cov, cycle_cov, initial_trend_mean, initial_cycle_mean, initial_trend_covariance, initial_cycle_covariance; p::Int = 1)

    n_variables, n_trends = size(trend_mapping)
    n_cycle_states = n_variables * p          # cycle companion state dimension
    n_states = n_trends + n_cycle_states

    # Companion transition for the cycle VAR(p), oldest-lag-first ordering.
    if p == 1
        cycle_transition = var_coeff
    else
        cycle_transition = vcat(
            hcat(zeros(n_variables * (p - 1), n_variables), I(n_variables * (p - 1))),
            var_coeff)
    end

    T = [I(n_trends)                      zeros(n_trends, n_cycle_states) # Transition matrix
         zeros(n_cycle_states, n_trends)  cycle_transition]

    # Only the contemporaneous cycle block (last block) carries noise.
    cycle_Q = zeros(n_cycle_states, n_cycle_states)
    cycle_Q[end-n_variables+1:end, end-n_variables+1:end] = cycle_cov
    Q = [trend_cov                        zeros(n_trends, n_cycle_states) #State noise covariance
         zeros(n_cycle_states, n_trends)  cycle_Q]

    R = Matrix(I, n_states, n_states)  # State noise coefficient matrix
    # Observation maps trends and the contemporaneous cycle (last companion block).
    Z = hcat(trend_mapping, zeros(n_variables, n_variables * (p - 1)), I(n_variables))


     H = Matrix(I, n_variables, n_variables) * eps()  # Observation noise covariance

     initial_state_mean = [initial_trend_mean; initial_cycle_mean]

     initial_state_covariance = [initial_trend_covariance zeros(n_trends, n_cycle_states)
                                 zeros(n_cycle_states, n_trends) initial_cycle_covariance]


    return StateSpaceModel(T, R, Z, Q, H, initial_state_mean, initial_state_covariance)


end

"""
    stationary_cycle_covariance(model::StateSpaceModel, n_trends)

Stationary (unconditional) state covariance implied by the full transition matrix
`model.T` and state-noise covariance `model.Q`, restricted to the cycle block.

Solves the discrete Lyapunov equation `P = T·P·T' + Q` in vectorised form,
`vec(P) = (I − T⊗T)⁻¹ vec(Q)`, restricted to the **cycle** block only.

The trend block of `model.T` is a decoupled random walk, so the full `I − T⊗T` is
singular and the trend initialisation is kept at its prior value anyway. The cycle
companion block, by contrast, is guaranteed stable (enforced by the stationarity
rejection in `sample_var_params`), so `I − Tc⊗Tc` is invertible and an ordinary
solve replaces the Moore–Penrose pseudoinverse of the full `n_states²` system.
This drops the dimension from `n_states²` to `n_cycle_states²` and the SVD to an LU
solve.
"""
function stationary_cycle_covariance(model::StateSpaceModel, n_trends)
    Tc = model.T[n_trends+1:end, n_trends+1:end]
    Qc = model.Q[n_trends+1:end, n_trends+1:end]
    n_cycle_states = size(Tc, 1)
    vecP = (I - kron(Tc, Tc)) \ vec(Qc)
    return reshape(vecP, n_cycle_states, n_cycle_states)
end

"""
    update_tc_var!(model, var_coeff, trend_cov, cycle_cov, n_trends, n_variables, p)

Overwrite, in place, only the blocks of `model` that change between Gibbs draws:

* the VAR companion bottom block `[A_p … A_1]` of the transition `T`
  (`var_coeff`, size `n_variables × n_variables*p`),
* the trend block of the state-noise covariance `Q` (`trend_cov`),
* the contemporaneous-cycle block of `Q` (`cycle_cov`, the last `n_variables`
  states), and
* the cycle block of `initial_state_covariance`, re-initialised from the implied
  stationary distribution of the just-updated VAR dynamics
  (see [`stationary_cycle_covariance`](@ref); the trend block is kept fixed at
  its prior value).

Every other block built by [`tc_var`](@ref) — `R`, `Z`, `H`, the initial means,
the identity/shift structure of `T`, and the zero off-diagonal blocks of `Q` — is
constant across draws and left untouched. This avoids reconstructing the whole
`StateSpaceModel` (and reallocating several dense `n_states × n_states` matrices)
on every sweep.
"""
function update_tc_var!(model::StateSpaceModel, var_coeff, trend_cov, cycle_cov, n_trends, n_variables, p)
    n_cycle_states = n_variables * p
    n_states = n_trends + n_cycle_states

    # Companion bottom block (oldest-lag-first): the last n_variables rows of the
    # cycle transition, sitting at rows n_trends + n_variables*(p-1)+1 : end and
    # cols n_trends+1 : end. The shift block above it is constant.
    row0 = n_trends + n_variables * (p - 1)
    model.T[row0+1:end, n_trends+1:end] = var_coeff

    # Trend block of Q.
    model.Q[1:n_trends, 1:n_trends] = trend_cov

    # Contemporaneous cycle block of Q (last n_variables states).
    model.Q[n_states-n_variables+1:end, n_states-n_variables+1:end] = cycle_cov

    # Re-initialise the cycle block of the initial covariance from the implied
    # stationary distribution of the just-updated VAR dynamics.
    model.initial_state_covariance[n_trends+1:end, n_trends+1:end] =
        stationary_cycle_covariance(model, n_trends)

    return model
end

function sample(model:: StateSpaceModel,  n_steps)
        
    initial_states = rand(MvNormal(model.initial_state_mean, model.initial_state_covariance))    

    return sample(model, initial_states, n_steps)
   
end

function sample(model:: StateSpaceModel, initial_state, n_steps)

    n_variables, n_states = size(model.Z)
    states = zeros(n_steps, n_states)
    obs = zeros(n_steps, n_variables)

    
    states[1, :] = initial_state
    obs[1, :] = model.Z * states[1,:] .+ rand(MvNormal(zeros(n_variables), model.H))
    
    for t in 2:n_steps
        states[t,:] = model.T * states[t-1,:] + rand(MvNormal(zeros(n_states), model.Q))
        obs[t, :] = model.Z * states[t,:] + rand(MvNormal(zeros(n_variables), model.H))
    end

    return states, obs

end

