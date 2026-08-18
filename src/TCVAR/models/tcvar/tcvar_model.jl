
"""
    tc_var(trend_mapping; p = 1)

Build the state space *skeleton* of a trend-cycle model whose cycle follows a
VAR(`p`). The cycle is represented in companion form, ordered oldest-lag-first:

    ξ_t = [c_{t-p+1}; …; c_{t-1}; c_t]   (length n_variables * p)

so that the VAR companion bottom block `[A_p … A_1]` (size n_variables × n_variables*p)
matches that ordering, i.e. the rows of the regression `Y = X·B` with predictors
stacked oldest-lag-first (`var_coeff == B'`). The contemporaneous cycle `c_t` is the
last block of ξ_t.

Only the constant structure is built here — `R`, `Z`, `H`, the identity/shift
structure of `T`, and the zero off-diagonal blocks of `Q`. The draw-dependent blocks
(the VAR companion bottom block of `T`, the trend and contemporaneous-cycle blocks of
`Q`) are initialised to zero and filled in by [`update_tc_var!`](@ref).

The initial state distribution is not part of the model: the caller owns
`initial_state_mean` / `initial_state_covariance` and passes them to
[`kalman_filter`](@ref). For `p == 1` this reduces to the original VAR(1) model.
"""
function tc_var(trend_mapping; p::Int = 1)

    n_variables, n_trends = size(trend_mapping)
    n_cycle_states = n_variables * p          # cycle companion state dimension
    n_states = n_trends + n_cycle_states

    # Companion transition for the cycle VAR(p), oldest-lag-first ordering. The
    # bottom block [A_p … A_1] starts at zero (filled by update_tc_var!); the shift
    # block above it is the constant part of the companion form.
    if p == 1
        cycle_transition = zeros(n_variables, n_variables)
    else
        cycle_transition = vcat(
            hcat(zeros(n_variables * (p - 1), n_variables), I(n_variables * (p - 1))),
            zeros(n_variables, n_cycle_states))
    end

    T = [I(n_trends)                      zeros(n_trends, n_cycle_states) # Transition matrix
         zeros(n_cycle_states, n_trends)  cycle_transition]

    # State-noise covariance: trend block and contemporaneous cycle block start at
    # zero; only those blocks carry noise once filled by update_tc_var!.
    Q = zeros(n_states, n_states)

    R = Matrix(I, n_states, n_states)  # State noise coefficient matrix
    # Observation maps trends and the contemporaneous cycle (last companion block).
    Z = hcat(trend_mapping, zeros(n_variables, n_variables * (p - 1)), I(n_variables))


     H = Matrix(I, n_variables, n_variables) * eps()  # Observation noise covariance

    return StateSpaceModel(T, R, Z, Q, H)


end

default_variable_names(n_variables) = ["y$i" for i in 1:n_variables]
default_trend_names(n_trends) = ["τ$i" for i in 1:n_trends]

"""
    TCVAR

Trend-cycle VAR model: bundles the state-space skeleton built by [`tc_var`](@ref)
with the priors and the observation-variable / trend names, so the sampler
receives one ready-made model instead of building the skeleton itself.

# Fields
- `ssm::StateSpaceModel` : state-space skeleton (draw-dependent blocks start at
  zero and are filled by [`update_tc_var!`](@ref) during sampling)
- `priors::NamedTuple` : one distribution per parameter name, stored verbatim —
  - `initial_trend::MvNormal` (length `n_trends`) : initial trend state `τ₀`
  - `initial_cycle::MvNormal` (length `n_obs*p`) : initial cycle companion state
    `ξ₀ = [c_{-p+1}; …; c_0]`, oldest-lag-first (see [`initial_cycle_prior`](@ref))
  - `trend_covariance::InverseWishart` (`n_trends × n_trends`) : trend innovation
    covariance; its scale matrix is used as written, no rescaling
  - `cycle_covariance::InverseWishart` (`n_obs × n_obs`) : cycle innovation covariance
  - `cycle_β::MinnesotaPrior` : cycle VAR coefficients; `n` and `p` are read off it
- `variable_names::Vector{String}` : names of the observed variables
- `trend_names::Vector{String}` : names of the trend states
"""
struct TCVAR
    ssm::StateSpaceModel
    priors::NamedTuple
    variable_names::Vector{String}
    trend_names::Vector{String}
end

"""
    TCVAR(trend_mapping, priors; variable_names, trend_names)

Build a [`TCVAR`](@ref) model from an `n_obs × n_trends` trend-to-observation mapping
and the distribution-keyed `priors` NamedTuple (see [`TCVAR`](@ref) for the five
required keys). The number of cycle VAR lags `p` and the number of variables `n` come
from `priors.cycle_β`, so there is no separate cycle-prior argument; the tuple is
stored exactly as given. `variable_names` defaults to `["y1", …]` and `trend_names` to
`["τ1", …]`.

```julia
Σ_prior, β_prior, c0_prior = var_priors(0.2, 4, [2., 1., .1, 25., 1]; δ = zeros(5))

priors = (initial_trend    = MvNormal(τ₀_mean, τ₀_cov),
          initial_cycle    = c0_prior,
          trend_covariance = InverseWishart(dτ, Ψτ),
          cycle_covariance = Σ_prior,
          cycle_β          = β_prior)

model = TCVAR(trend_mapping, priors)
```
"""
function TCVAR(trend_mapping, priors::NamedTuple;
               variable_names = default_variable_names(size(trend_mapping, 1)),
               trend_names = default_trend_names(size(trend_mapping, 2)))

    n_obs, n_trends = size(trend_mapping)

    required = (:initial_trend, :initial_cycle, :trend_covariance, :cycle_covariance, :cycle_β)
    missing_keys = filter(key -> !haskey(priors, key), required)
    isempty(missing_keys) ||
        throw(ArgumentError("priors is missing key(s): $(join(missing_keys, ", "))"))

    β_prior = priors.cycle_β
    p = β_prior.p

    β_prior.n == n_obs || throw(DimensionMismatch(
        "priors.cycle_β is built for n = $(β_prior.n) variables, trend_mapping has $n_obs"))
    size(priors.cycle_covariance) == (n_obs, n_obs) || throw(DimensionMismatch(
        "priors.cycle_covariance must be $n_obs × $n_obs"))
    length(priors.initial_cycle) == n_obs * p || throw(DimensionMismatch(
        "priors.initial_cycle must have length n_obs*p = $(n_obs * p)"))
    length(priors.initial_trend) == n_trends || throw(DimensionMismatch(
        "priors.initial_trend must have length $n_trends"))
    size(priors.trend_covariance) == (n_trends, n_trends) || throw(DimensionMismatch(
        "priors.trend_covariance must be $n_trends × $n_trends"))
    length(variable_names) == n_obs ||
        throw(DimensionMismatch("variable_names must have length $n_obs"))
    length(trend_names) == n_trends ||
        throw(DimensionMismatch("trend_names must have length $n_trends"))

    return TCVAR(tc_var(trend_mapping; p = p),
                 priors,
                 collect(String, variable_names),
                 collect(String, trend_names))
end

"""
    stationary_cycle_covariance(model::StateSpaceModel, n_trends)

Stationary (unconditional) state covariance implied by the full transition matrix
`model.T` and state-noise covariance `model.Q`, restricted to the cycle block.

Delegates to [`lyapunov_covariance`](@ref), restricted to the **cycle** block only.

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
    return lyapunov_covariance(Tc, Qc)
end

"""
    update_tc_var!(model, var_coeff, trend_cov, cycle_cov, n_trends, n_variables, p)

Overwrite, in place, only the blocks of `model` that change between Gibbs draws:

* the VAR companion bottom block `[A_p … A_1]` of the transition `T`
  (`var_coeff`, size `n_variables × n_variables*p`),
* the trend block of the state-noise covariance `Q` (`trend_cov`),
* the contemporaneous-cycle block of `Q` (`cycle_cov`, the last `n_variables`
  states).

The initial state distribution lives outside the model now; the caller is
responsible for re-initialising the cycle block of its `initial_state_covariance`
from the implied stationary distribution of the just-updated VAR dynamics
(see [`stationary_cycle_covariance`](@ref)).

Every other block built by [`tc_var`](@ref) — `R`, `Z`, `H`,
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

    return model
end
