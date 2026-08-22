# The multivariate stochastic-volatility Gibbs block: given residuals whose conditional
# covariance is `diag(exp(h_t))`, draw the whole log-volatility path.
#
#     y*_t = log(e_t² + c̄)  =  h_t + z_t,        z_t ~ log χ²₁      (measurement)
#     h_t  = μ + Φ (h_{t-1} − μ) + ν_t,          ν_t ~ N(0, Ω)      (state)
#
# Two steps, in this order: draw the KSC mixture labels given `h` (`ksc_mixture.jl`), then
# draw `h` given the labels — conditional on them the system above is linear Gaussian, so
# the entire path comes out of one Carter–Kohn sweep.
#
# Nothing here knows about trends, cycles or a VAR: the caller supplies orthogonalised
# residuals and the three volatility parameters, which is what makes this block reusable
# by any model with a stochastic-volatility layer.

"""
    stationary_volatility_covariance(Φ, Ω) -> Matrix

Unconditional covariance of the log-volatility VAR(1) `h̃_t = Φ h̃_{t-1} + ν_t`, used as
the prior covariance of `h̃_0` when the caller does not supply one.

Throws when `Φ` is not stable — a random-walk volatility (`Φ = I`) has no stationary
distribution, and such a model has to pass an explicit diffuse `h0_covariance` instead.
"""
function stationary_volatility_covariance(Φ::AbstractMatrix, Ω::AbstractMatrix)
    maximum(abs.(eigvals(Matrix(float.(Φ))))) < 1 || throw(ArgumentError(
        "Φ has an eigenvalue on or outside the unit circle, so h has no stationary " *
        "distribution; pass an explicit diffuse `h0_covariance` (a random-walk " *
        "volatility, Φ = I, always needs one)"))
    return lyapunov_covariance(Matrix(float.(Φ)), Matrix(float.(Ω)))
end

"""
    volatility_state_space(Φ, Ω, observation_covariance = zeros(n, n))
        -> TimeVaryingStateSpaceModel

The log-volatility VAR(1) `h_t = μ + Φ (h_{t-1} − μ) + ν_t`, `ν_t ~ N(0, Ω)`, as a state
space model in the demeaned volatility `h̃_t = h_t − μ`:

    state:        h̃_t = Φ h̃_{t-1} + ν_t     (T = Φ, R = I, Q = Ω)
    observation:  ỹ_t = h̃_t + z_t            (Z = I, H = observation_covariance)

`μ` is not part of the model — a state space model has no intercept — so the caller demeans
what it feeds in and adds `μ` back to the path that comes out. Both users of the volatility
VAR(1) do exactly that, and differ only in the measurement they attach to it:

* [`draw_log_volatilities`](@ref) passes the `n_time × n × n` KSC mixture variances
  `H_t = diag(v²_{s_t})` and filters `ỹ_t = y*_t − m_{s_t} − μ` through them — a
  covariance that moves every period, which is what
  [`TimeVaryingStateSpaceModel`](@ref) exists for;
* [`simulate_volatility_path`](@ref) leaves `H` at its zero default — a simulator wants
  the state path itself, so its observation equation is a formality.
"""
function volatility_state_space(Φ::AbstractMatrix, Ω::AbstractMatrix,
                                observation_covariance = zeros(size(Φ, 1), size(Φ, 1)))
    n = size(Φ, 1)
    identity_n = Matrix(1.0I, n, n)

    return TimeVaryingStateSpaceModel(Matrix(float.(Φ)), identity_n, identity_n,
                                      Matrix(float.(Ω)), observation_covariance)
end

"""
    draw_log_volatilities(y_star, indicators, params; h0_covariance = nothing) -> Matrix

Draw the log-volatility path conditional on the mixture labels.

Given `s`, component `s_it` contributes a known mean `m_{s_it}` and a known variance
`v²_{s_it}`, so subtracting the mean (and the unconditional level `μ`) leaves a linear
Gaussian state space in `h̃_t = h_t − μ`:

    measurement:  ỹ_t = h̃_t + z_t,        ỹ_t = y*_t − m_{s_t} − μ,  z_t ~ N(0, diag(v²_{s_t}))
    state:        h̃_t = Φ h̃_{t-1} + ν_t,  ν_t ~ N(0, Ω)

i.e. the state space model of [`volatility_state_space`](@ref) carrying a *time-varying*
observation covariance. `carter_kohn_sampler` returns `h̃_0` alongside `h̃_{1:T}`, so the
`h_0` the volatility-parameter blocks need comes out of the same sweep.

`y_star` and `indicators` are `T × n`; `params` is a NamedTuple `(μ, Φ, Ω)`. Returns the
`(T+1) × n` path `h_{0:T}` with `μ` added back, row `t+1` being period `t`.
"""
function draw_log_volatilities(y_star::AbstractMatrix{<:Real}, indicators::AbstractMatrix{<:Integer},
                               params; h0_covariance = nothing)
    size(y_star) == size(indicators) || throw(DimensionMismatch(
        "y_star is $(size(y_star)) but the indicators are $(size(indicators))"))

    μ = collect(float.(params.μ))
    Φ = Matrix(float.(params.Φ))
    Ω = Matrix(float.(params.Ω))

    n_time, n = size(y_star)
    length(μ) == n && size(Φ) == (n, n) && size(Ω) == (n, n) || throw(DimensionMismatch(
        "y_star has $n series but params carry μ of length $(length(μ)), Φ of size " *
        "$(size(Φ)) and Ω of size $(size(Ω))"))

    means, variances = KSC_MIXTURE.means, KSC_MIXTURE.variances

    # Measurement equation: de-mean by the drawn component and by μ, and collect the
    # component variances into the n_time × n × n diagonal observation covariances.
    demeaned = Matrix{Float64}(undef, n_time, n)
    observation_covariances = zeros(n_time, n, n)
    @inbounds for i in 1:n, t in 1:n_time
        component = indicators[t, i]
        demeaned[t, i] = y_star[t, i] - means[component] - μ[i]
        observation_covariances[t, i, i] = variances[component]
    end

    model = volatility_state_space(Φ, Ω, observation_covariances)

    initial_covariance = isnothing(h0_covariance) ? stationary_volatility_covariance(Φ, Ω) :
                                                    Matrix(float.(h0_covariance))

    h0_demeaned, h_demeaned = carter_kohn_sampler(model, demeaned, zeros(n), initial_covariance)

    return vcat(h0_demeaned', h_demeaned) .+ μ'
end

"""
    draw_stochastic_volatility(residuals, h, params; h0_covariance = nothing, offset = 1e-3)
        -> (h_new, indicators)

One Gibbs draw of a multivariate stochastic-volatility path, via the KSC mixture
approximation.

* `residuals` — `T × n` orthogonalised residuals `e_t`, i.e. residuals already rotated so
  that their conditional covariance is the diagonal `diag(exp(h_t))`. In TCVAR-SV these
  are `e_t = A₀ ε_t` (as a `T × n` matrix, `ε * A₀'`).
* `h` — the `(T+1) × n` current path `h_{0:T}`, row `t+1` being period `t`. Only the
  `h_{1:T}` rows enter, as the conditioning value of the indicator draw.
* `params` — NamedTuple `(μ, Φ, Ω)`: unconditional level, AR matrix and innovation
  covariance of the log-volatility VAR(1).

# Keywords
- `h0_covariance` — prior covariance of `h̃_0 = h_0 − μ`. Defaults to the stationary
  covariance implied by `(Φ, Ω)`, which requires a stable `Φ`; a random-walk volatility
  must pass a diffuse matrix.
- `offset` — the KSC offset `c̄` in `log(e² + c̄)`, which keeps the log finite when a
  residual is (numerically) zero. `1e-3` is the value KSC use; it biases `y*` upwards
  for residuals much smaller than `√c̄`, so scale it down if the data are.

Returns the new `(T+1) × n` path and the `T × n` mixture labels (the latter are not
needed by the other blocks, but they are what makes the draw reproducible and are worth
inspecting when a volatility path misbehaves).
"""
function draw_stochastic_volatility(residuals::AbstractMatrix{<:Real}, h::AbstractMatrix{<:Real},
                                    params; h0_covariance = nothing, offset::Real = 1e-4)
    n_time, n = size(residuals)
    size(h) == (n_time + 1, n) || throw(DimensionMismatch(
        "residuals are $(size(residuals)) (T × n), so h must be $((n_time + 1, n)) " *
        "(rows t = 0 … T) but is $(size(h))"))
    offset > 0 || throw(ArgumentError("the KSC offset c̄ must be > 0, got $offset"))

    y_star = log.(residuals .^ 2 .+ offset)
    indicators = draw_mixture_indicators(y_star, @view h[2:end, :])
    h_new = draw_log_volatilities(y_star, indicators, params; h0_covariance = h0_covariance)

    return h_new, indicators
end
