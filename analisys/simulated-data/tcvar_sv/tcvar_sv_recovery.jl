# The full TCVAR-SV on simulated data — the simulation half.
#
#     y_t = Λ τ_t + c_t
#     τ_t = τ_{t-1} + u_t,                          u_t ~ N(0, Στ)
#     c_t = A₁ c_{t-1} + … + A_p c_{t-p} + ε_t,     ε_t ~ N(0, Σ_t)
#     Σ_t = A₀⁻¹ H_t A₀⁻ᵀ,                          H_t = diag(exp(h_t))
#     h_t = μ + Φ (h_{t-1} − μ) + ν_t,              ν_t ~ N(0, Ω)
#
# One TCVAR-SV path of T = 400 periods, drawn forward from known (Στ, β, A₀, μ, Φ, Ω) by
# `simulate_scenarios(::TCVARSV, params, …)` — the generator the recovery run of §5.3 of
# `tcvar_sv_plan.md` estimates. **This script stops at the data.** The estimation half —
# `gibbs_sampler(::TCVARSV, data; burnin, n_samples)`, the truth-vs-posterior table and the
# state/volatility plots against it — lands with stage 6 of the plan, once the sweep
# exists; nothing here anticipates it.
#
# What it is good for on its own: the simulator is the one piece of the recovery run that
# cannot be checked *by* the recovery run — an error in it makes the estimate look wrong
# with no way to tell which side is at fault. So the path is measured against the
# parameters that generated it (§ "the path is the process it claims to be" below), and
# the figure shows what the sampler will be handed: the trends against the observations,
# the cycles, and the volatilities whose movement is the entire point of the model.
#
# Run as:  julia --project analisys/simulated-data/tcvar_sv/tcvar_sv_recovery.jl

using Distributions
using LinearAlgebra
using Plots
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "..", "..", "..", "src", "TCVAR", "TCVAR.jl"))

# `using Statistics` loses to a variable of the same name already bound in `Main`, and a
# live REPL / IDE worker accumulates those: `analisys/temp.jl`, for one, assigns `std` and
# `cov` as data. The script then dies 200 lines below, inside a @printf, with "objects of
# type Vector{Float64} are not callable" and no hint of where that came from. Say it here.
for name in (:mean, :std, :cov, :cor)
    shadowing = isdefined(Main, name) ? getfield(Main, name) : nothing
    shadowing isa Function || shadowing === nothing || error(
        "Main.$name is a $(typeof(shadowing)), not the Statistics function: something in " *
        "this session bound `$name` as a variable and every `$name(x)` below would fail. " *
        "Restart the session, or rebind it with `$name = Statistics.$name`.")
end

const OUTPUT_DIR = joinpath(@__DIR__, "output")

# ---------------------------------------------------------------------------- settings

const SEED     = 20250821
const N_SERIES = 3
const N_TRENDS = 3
const N_LAGS   = 1
const N_TIME   = 400

# One trend per observed series: y_t = τ_t + c_t.
const TREND_MAPPING = Matrix(1.0I, N_SERIES, N_TRENDS)
const VARIABLE_NAMES = ["y1", "y2", "y3"]
const TREND_NAMES    = ["τ1", "τ2", "τ3"]

# ------------------------------------------------------------------------------- truth

# Trend innovations: random-walk steps of 0.10 / 0.07 / 0.15 per period. Kept well clear
# of the 1e-4 jitter `sample` adds to every state covariance (see the trend check below),
# and small against the cycle so the decomposition is not degenerate.
const Στ_TRUE = diagm([0.10, 0.07, 0.15] .^ 2)

# Cycle VAR(1) in its natural form c_t = A₁ c_{t-1} + ε_t, moderately persistent with two
# cross-effects so β is not diagonal. `β` is its transpose: the codebase writes the cycle
# as c_t = βᵀ x_t with x_t = [c_{t-p}; …; c_{t-1}] stacked oldest-lag-first, so β is
# k × n and the companion bottom block the sampler sees is βᵀ = [A_p … A₁].
const A1_TRUE = [0.70  0.10  0.00
                 0.00  0.60  0.20
                 0.10  0.00  0.50]
const Β_TRUE = Matrix(A1_TRUE')

# Unit lower triangular; the free elements below the diagonal are what step 4 of the sweep
# draws. Non-trivial in both signs, so a dropped negation cannot pass unnoticed.
const A₀_TRUE = [ 1.0  0.0  0.0
                  0.4  1.0  0.0
                 -0.2  0.3  1.0]

# The three series sit at genuinely different volatility levels (innovation sd 0.5, 1.0,
# 2.0 at the unconditional mean), Φ is the diagonal default at high persistence, and Ω is
# deliberately correlated across series.
const MU_TRUE = log.([0.5, 1.0, 2.0] .^ 2)
const Φ_TRUE = Matrix(0.95I, N_SERIES, N_SERIES)
const Ω_TRUE = 0.02 * [1.0 0.5 0.2
                       0.5 1.0 0.3
                       0.2 0.3 1.0]

# Where the trends start. Levels only — they carry no information about the dynamics, and
# the sampler estimates them as states.
const TREND_START = [2.0, 1.0, 4.0]

# ------------------------------------------------------------------------------ helpers

"""Unconditional cycle innovation covariance `Σ̄ = A₀⁻¹ diag(exp(μ)) A₀⁻ᵀ` — the covariance
`Σ_t` fluctuates around, and the scale every prior below is set on."""
unconditional_cycle_covariance(A₀, μ) = Symmetric(A₀ \ diagm(exp.(μ)) / A₀')

"""Priors of the model the simulated data will be estimated with.

Nothing in this script reads them — `simulate_scenarios` takes the true parameters
directly — but [`TCVARSV`](@ref) is the object that carries the state-space skeleton, and
it does not exist without a validated prior tuple. They are set on the *scale* of the
truth (the usual practice for a recovery exercise: a prior on the wrong scale tests the
prior, not the sampler) and are loose about everything else — the Minnesota prior is
centred at white noise (`δ = 0`), which is the standard choice for a cycle."""
function simulation_priors(Σ̄)
    Σc_prior, β_prior, c₀_prior = TCVAR.var_priors(0.2, N_LAGS, diag(Σ̄); δ = zeros(N_SERIES))

    return TCVAR.tcvar_sv_priors(
        (initial_trend    = MvNormal(TREND_START, Matrix(1.0I, N_TRENDS, N_TRENDS)),
         initial_cycle    = c₀_prior,
         trend_covariance = InverseWishart(N_TRENDS + 10.0,
                                           Matrix((N_TRENDS + 10.0 - N_TRENDS - 1) * Στ_TRUE)),
         cycle_covariance = Σc_prior,
         cycle_β          = β_prior),
        TCVAR.sv_priors(N_SERIES; volatility_level = 1.0))
end

"""Largest deviation of a sample covariance from its target, in units of the target's own
scale: `max |S_ij − Σ_ij| / sqrt(Σ_ii Σ_jj)`.

A plain relative deviation is useless here — every target below has exactly-zero
off-diagonal entries — and an absolute one is unreadable across blocks whose variances
differ by two orders of magnitude. Dividing by `sqrt(Σ_ii Σ_jj)` puts every entry on the
correlation scale, where the sampling error of a `T = 400` sample is a few hundredths and
a layout error is order one."""
scaled_covariance_deviation(sample, target) =
    maximum(abs.((sample .- target) ./ sqrt.(diag(target) * diag(target)')))

"""Starting state `[τ₀; ξ₀]` of the simulation: the trends at `TREND_START`, the cycle
companion drawn from its own stationary distribution at the unconditional volatility `Σ̄`.

Drawing `ξ₀` rather than starting it at zero is what keeps the first periods of the sample
from being a visible transient towards the cycle's own scale — the simulated data is meant
to look like the middle of a series, not like its beginning."""
function stationary_cycle_start(A1, Σ̄, n, p)
    F = p == 1 ? Matrix(float.(A1)) :
        vcat(hcat(zeros(n * (p - 1), n), Matrix(1.0I, n * (p - 1), n * (p - 1))), A1)

    Q = zeros(n * p, n * p)
    Q[end-n+1:end, end-n+1:end] = Σ̄

    P = TCVAR.lyapunov_covariance(F, Q)
    return TCVAR.psd_factor(Symmetric((P + P') / 2)) * randn(n * p)
end

"""Draw `h_0 ~ N(μ, P₀)` from the stationary distribution of the volatility VAR(1), so the
volatility path starts where it lives rather than at its mean."""
stationary_volatility_start(μ, Φ, Ω) =
    μ + TCVAR.psd_factor(TCVAR.stationary_volatility_covariance(Φ, Ω)) * randn(length(μ))

"""One simulated TCVAR-SV sample, as the pieces the sampler and the plots want:
`(observations, trends, cycles, volatilities)`, each `n_time × ·`.

`simulate_scenarios` returns the companion state `ξ_t = [c_{t-p+1}; …; c_t]`; the cycle
`c_t` is its **last** `n` entries, the contemporaneous block."""
function simulate_sample(model, params, initial_state, initial_volatility, n_time)
    states, observations, volatilities =
        TCVAR.simulate_scenarios(model, params, initial_state, 1, n_time;
                                 initial_volatility = initial_volatility)

    return (observations = observations[1, :, :],
            trends       = states[1, :, 1:N_TRENDS],
            cycles       = states[1, :, end-N_SERIES+1:end],
            volatilities = volatilities[1, :, :])
end

"""Three-panel-per-series figure of the simulated sample: the observation against its
trend, the cycle, and the volatility `exp(h_t/2)` that generated the cycle innovations.

The three columns are the three layers the model separates, in the order the model builds
them, so a path that does not look like a trend-cycle decomposition is visible as such."""
function plot_simulated_sample(data)
    periods = 1:size(data.observations, 1)

    panels = Plots.Plot[]
    for i in 1:N_SERIES
        observation = plot(periods, data.observations[:, i];
                           color = :grey, linewidth = 1, alpha = 0.8,
                           label = i == 1 ? "y_t" : "",
                           ylabel = VARIABLE_NAMES[i], legend = i == 1 ? :best : false,
                           title = i == 1 ? "observation and trend" : "")
        plot!(observation, periods, data.trends[:, i];
              color = :black, linewidth = 2, label = i == 1 ? "Λτ_t" : "")

        cycle = plot(periods, data.cycles[:, i];
                     color = :steelblue, linewidth = 1, legend = false,
                     title = i == 1 ? "cycle" : "")
        hline!(cycle, [0.0]; color = :black, linewidth = 0.5, linestyle = :dot)

        volatility = plot(periods, exp.(data.volatilities[:, i] ./ 2);
                          color = :firebrick, linewidth = 1.5, legend = i == 1 ? :best : false,
                          label = i == 1 ? "exp(h_t/2)" : "",
                          title = i == 1 ? "cycle innovation volatility" : "")
        hline!(volatility, [exp(MU_TRUE[i] / 2)];
               color = :black, linewidth = 1, linestyle = :dash,
               label = i == 1 ? "exp(μ/2)" : "")

        append!(panels, (observation, cycle, volatility))
        i == N_SERIES && foreach(panel -> xlabel!(panel, "t"), panels[end-2:end])
    end

    return plot(panels...; layout = (N_SERIES, 3), size = (1400, 250 * N_SERIES),
                plot_title = "TCVAR-SV simulated sample", left_margin = 5Plots.mm,
                bottom_margin = 5Plots.mm)
end

# --------------------------------------------------------------------------------- run

Random.seed!(SEED)

Σ̄_TRUE = unconditional_cycle_covariance(A₀_TRUE, MU_TRUE)

TCVAR.is_stationary(A1_TRUE, N_SERIES, N_LAGS) ||
    error("the cycle VAR is not stationary — the simulated cycle would diverge")
TCVAR.is_stationary(Φ_TRUE, N_SERIES, 1) ||
    error("the volatility VAR(1) is not stationary — h has no unconditional distribution")

params = (Στ = Στ_TRUE, β = Β_TRUE, A₀ = A₀_TRUE, μ = MU_TRUE, Φ = Φ_TRUE, Ω = Ω_TRUE)
model = TCVAR.TCVARSV(TREND_MAPPING, simulation_priors(Σ̄_TRUE), N_TIME;
                variable_names = VARIABLE_NAMES, trend_names = TREND_NAMES,
                ar_structure = :diagonal)

initial_state = [TREND_START; stationary_cycle_start(A1_TRUE, Σ̄_TRUE, N_SERIES, N_LAGS)]
initial_volatility = stationary_volatility_start(MU_TRUE, Φ_TRUE, Ω_TRUE)

@info "simulating" n_series = N_SERIES n_trends = N_TRENDS p = N_LAGS n_time = N_TIME
data = simulate_sample(model, params, initial_state, initial_volatility, N_TIME)

# ------------------------------------------------------------------------------ report

@printf("\nTCVAR-SV simulated sample — n = %d, n_trends = %d, p = %d, T = %d (seed %d)\n",
        N_SERIES, N_TRENDS, N_LAGS, N_TIME, SEED)
println("Data only: the parameters below generated it, nothing here estimates them.")

println("\nSimulated series:")
println("  series        mean          sd         min         max   |   cycle sd   trend sd")
for i in 1:N_SERIES
    y, τ, c = data.observations[:, i], data.trends[:, i], data.cycles[:, i]
    @printf("  %6s  %10.3f  %10.3f  %10.3f  %10.3f   |  %9.3f  %9.3f\n",
            VARIABLE_NAMES[i], mean(y), std(y), minimum(y), maximum(y), std(c), std(τ))
end

# The path against the process that generated it. Each block below recovers one moment of
# one true parameter from the simulated path alone. At T = 400 the sampling error on them
# is a few hundredths, so they catch a wrong layout — a transposed β, a mis-stacked
# companion, a dropped A₀ — rather than a small bias.
println("\nThe path is the process it claims to be")

# `sample` adds a 1e-4 jitter to every state covariance, so the trend steps carry
# Στ + 1e-4·I, not Στ. That is ~1% of the smallest trend variance here, and it is added
# rather than ignored so the comparison is against what was actually drawn.
trend_steps = diff(data.trends, dims = 1)
trend_target = Στ_TRUE + 1e-4I
@printf("  trend steps      cov(Δτ) vs Στ + 1e-4·I : max scaled dev = %.3f  (diag: %s vs %s)\n",
        scaled_covariance_deviation(cov(trend_steps), trend_target),
        join((@sprintf("%.4f", v) for v in diag(cov(trend_steps))), ", "),
        join((@sprintf("%.4f", v) for v in diag(trend_target)), ", "))

# The cycle innovations ε_t = c_t − A₁c_{t-1} are heteroskedastic by construction, so the
# comparison that is tight at T = 400 is against the covariance the *realised* volatility
# path implies, Σ̄_realised = A₀⁻¹ diag(mean_t exp(h_t)) A₀⁻ᵀ — that one is the simulator's
# job. The unconditional levels underneath are context, and they are deliberately three
# different numbers: `exp` is convex, so E[Σ_t] = A₀⁻¹diag(exp(μ + σ²_h/2))A₀⁻ᵀ sits above
# Σ̄ = A₀⁻¹diag(exp(μ))A₀⁻ᵀ, and at Φ = 0.95 the average volatility of any one 400-period
# path is itself an imprecise draw around E[Σ_t] — a gap there is the path's luck, not a
# defect.
cycle_innovations = data.cycles[2:end, :] - data.cycles[1:end-1, :] * A1_TRUE'
volatility_variance = diag(TCVAR.stationary_volatility_covariance(Φ_TRUE, Ω_TRUE))
Σ_realised = Symmetric(A₀_TRUE \ diagm(vec(mean(exp.(data.volatilities[2:end, :]), dims = 1))) /
                       A₀_TRUE')
Σ_expected = Symmetric(A₀_TRUE \ diagm(exp.(MU_TRUE .+ volatility_variance ./ 2)) / A₀_TRUE')
@printf("  cycle innov.     cov(ε) vs realised Σ̄  : max scaled dev = %.3f\n",
        scaled_covariance_deviation(cov(cycle_innovations), Σ_realised))
@printf("                   sd(ε)      = %s\n",
        join((@sprintf("%7.3f", v) for v in sqrt.(diag(cov(cycle_innovations)))), " "))
@printf("                   realised Σ̄ = %s   E[Σ_t] = %s   Σ̄ = %s\n",
        join((@sprintf("%7.3f", v) for v in sqrt.(diag(Σ_realised))), " "),
        join((@sprintf("%7.3f", v) for v in sqrt.(diag(Σ_expected))), " "),
        join((@sprintf("%7.3f", v) for v in sqrt.(diag(Σ̄_TRUE))), " "))

# A₀ is what makes the *orthogonalised* residuals e_t = A₀ε_t independent across series
# with variance exp(h_it): their sample correlation is the check that the factorisation in
# `cycle_covariance_path` is the one the model documents.
orthogonalised = cycle_innovations * A₀_TRUE'
scaled = orthogonalised ./ exp.(data.volatilities[2:end, :] ./ 2)
off_diagonal = cor(scaled) - I
@printf("  A₀               max |corr| of A₀ε_t / exp(h_t/2) off the diagonal = %.3f\n",
        maximum(abs.(off_diagonal)))
@printf("                   var of A₀ε_t / exp(h_t/2) (target 1) = %s\n",
        join((@sprintf("%6.3f", v) for v in diag(cov(scaled))), " "))

# And the volatility path is the stationary VAR(1) it was drawn from: the level is μ (up to
# the sampling error of a T = 400 path at Φ = 0.95, which is a stationary sd or so), and
# what is left after removing μ + Φ(h_{t-1} − μ) has covariance Ω.
volatility_innovations = data.volatilities[2:end, :] .- MU_TRUE' .-
                         (data.volatilities[1:end-1, :] .- MU_TRUE') * Φ_TRUE'
@printf("  volatility       cov(ν) vs Ω            : max scaled dev = %.3f\n",
        scaled_covariance_deviation(cov(volatility_innovations), Ω_TRUE))
println("\n  series          μ    mean h    sd of h   stationary sd")
for i in 1:N_SERIES
    @printf("  %6s  %9.4f  %8.4f  %9.4f  %14.4f\n",
            VARIABLE_NAMES[i], MU_TRUE[i], mean(data.volatilities[:, i]),
            std(data.volatilities[:, i]), sqrt(volatility_variance[i]))
end

# The observation equation has no noise of its own — the skeleton's H is eps()·I — so
# y_t − Λτ_t − c_t is nothing but the 1e-4 jitter `sample` adds to it, i.e. a standard
# deviation of 1e-2. Anything larger means the state layout was read wrongly.
observation_error = data.observations - data.trends * TREND_MAPPING' - data.cycles
@printf("\n  observation      max |y_t − Λτ_t − c_t| = %.4f  (jitter only, sd 1e-2)\n",
        maximum(abs, observation_error))

mkpath(OUTPUT_DIR)
figure_path = joinpath(OUTPUT_DIR, "tcvar_sv_simulated_sample.png")
savefig(plot_simulated_sample(data), figure_path)
println("\nfigure written to ", figure_path)

# Estimation — `gibbs_sampler(::TCVARSV, data.observations; burnin, n_samples)`, the
# truth-vs-posterior table, `plot_states` and `plot_volatilities` against the paths
# simulated above — is stage 6 of `tcvar_sv_plan.md` and is deliberately absent here: the
# sweep does not exist yet, and the data this script produces is what it will be run on.
