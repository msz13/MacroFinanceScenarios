# The stochastic-volatility block on its own, against a simulated stationary volatility.
#
#     h_t = μ + Φ (h_{t-1} − μ) + ν_t,     ν_t ~ N(0, Ω)     (correlated across series)
#     h_0 ~ N(μ, P₀),                      P₀ stationary
#     e_t ~ N(0, diag(exp(h_t)))
#
# One Gibbs block only — step 5 (`draw_stochastic_volatility`) of the TCVAR-SV sweep — with
# (μ, Φ, Ω) held at the values that generated the data. Only the *state* h is inferred: no
# volatility parameter is drawn, so nothing can absorb an error in the block, and the
# recovery reported below is attributable to this code and to nothing else. Recovering
# (μ, Φ, Ω) is the job of `sv_posterior_checks.jl`.
#
# The volatilities are simulated from the same stationary AR(1)-with-mean law the target
# model assumes, rather than from a random walk: that is the process the block will meet
# inside the sweep, and it fixes the level of h through μ, so a level error is a real
# defect here instead of a symptom of an unidentified level.
#
# This is the acceptance test for the block: it isolates the mixture constants, the
# -1.2704 shift, the log(e² + c̄) offset and the de-meaning from every other moving part,
# so a mistake in any of them shows up as a level or shape error against a known truth
# rather than being absorbed by the rest of a full sampler.
#
# Run as:  julia --project analisys/simulated-data/tcvar_sv/sv_block_recovery.jl

using Distributions
using LinearAlgebra
using Plots
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "..", "..", "..", "src", "TCVAR", "TCVAR.jl"))

const OUTPUT_DIR = joinpath(@__DIR__, "output")

# ---------------------------------------------------------------------------- settings

const SEED      = 20250819
const N_SERIES  = 3
const N_TIME    = 500
const N_BURNIN  = 2_000
const N_KEPT    = 3_000
const CREDIBLE  = 0.90

# The three series sit at genuinely different volatility levels (sd 0.5, 1.0, 2.0), so a
# mis-specified level — the failure mode of a wrong mixture shift or a missing de-meaning —
# cannot hide behind a common intercept.
const MU_TRUE = log.([0.5, 1.0, 2.0] .^ 2)
const Φ_TRUE = Matrix(0.95I, N_SERIES, N_SERIES)

# Deliberately correlated volatility innovations: the block draws the whole path jointly
# through a Carter–Kohn sweep with Q = Ω, so an implementation that quietly treated the
# series as n independent univariate SV models would still fit the marginals and only
# miss here.
const Ω_TRUE = 0.02 * [1.0 0.5 0.2;
                       0.5 1.0 0.3;
                       0.2 0.3 1.0]

# ------------------------------------------------------------------------------ helpers

"""Simulate `h_{0:T}` from the stationary VAR(1) `h_t = μ + Φ(h_{t-1} − μ) + ν_t` — started
from its own stationary distribution, so the path carries no burn-in transient — and the
residuals `e_t ~ N(0, diag(exp(h_t)))` they generate."""
function simulate_ar_volatility(n_time, μ, Φ, Ω)
    n = length(μ)
    Ω_L = cholesky(Symmetric(Ω)).L
    P0_L = cholesky(Symmetric(TCVAR.stationary_volatility_covariance(Φ, Ω))).L

    h = zeros(n_time + 1, n)
    h[1, :] = μ + P0_L * randn(n)
    for t in 2:(n_time + 1)
        h[t, :] = μ + Φ * (h[t-1, :] - μ) + Ω_L * randn(n)
    end

    residuals = [randn() * exp(h[t+1, i] / 2) for t in 1:n_time, i in 1:n]
    return h, residuals
end

"""Repeated draws of `h` from step 5 alone, with `(μ, Φ, Ω)` fixed at `params`.

Step 5 is itself a two-block sweep over the path and the mixture labels, so iterating it
is a valid Markov chain for `p(h | e, μ, Φ, Ω)` under the KSC approximation. `h` starts at
`μ` in every series; `h0_covariance` is left at its default, the stationary covariance
implied by `(Φ, Ω)` — the same prior the full sampler will use."""
function sample_volatility_path(residuals, params; n_burnin, n_kept)
    n_time, n = size(residuals)

    h = repeat(collect(float.(params.μ))', n_time + 1)
    h_kept = zeros(n_kept, n_time + 1, n)

    for iteration in 1:(n_burnin + n_kept)
        h, _ = TCVAR.draw_stochastic_volatility(residuals, h, params)

        if iteration > n_burnin
            h_kept[iteration - n_burnin, :, :] = h
        end
    end

    return h_kept
end

"""Per-series figure: the true volatility, the posterior mean and band, and the realised
`|e_t|` the inference actually saw.

The `|e_t|` scatter is there for context — it is the *only* thing the block sees — but a
single draw from `N(0, exp(h_t))` regularly lands three or four times above `exp(h_t/2)`,
so the axis is set from the volatility paths and the scatter is clipped to it. Otherwise
the panel is scaled by its largest outlier and the recovery it is meant to show becomes an
unreadable band near zero."""
function plot_volatility_recovery(h_true, residuals, h_mean, h_lower, h_upper)
    n_time, n = size(residuals)
    periods = 0:n_time

    panels = map(1:n) do i
        upper_limit = 1.1 * max(maximum(exp.(h_upper[:, i] ./ 2)),
                                maximum(exp.(h_true[:, i] ./ 2)))

        panel = plot(periods, exp.(h_mean[:, i] ./ 2);
                     ylims = (0, upper_limit),
                     ribbon = (exp.(h_mean[:, i] ./ 2) .- exp.(h_lower[:, i] ./ 2),
                               exp.(h_upper[:, i] ./ 2) .- exp.(h_mean[:, i] ./ 2)),
                     fillalpha = 0.2, linewidth = 2, color = :steelblue,
                     label = i == 1 ? "posterior mean ($(Int(100 * CREDIBLE))% band)" : "",
                     ylabel = "series $i", legend = i == 1 ? :topright : false)

        scatter!(panel, 1:n_time, abs.(residuals[:, i]);
                 markersize = 1.5, markerstrokewidth = 0, color = :grey, alpha = 0.45,
                 label = i == 1 ? "|e_t| (clipped)" : "")

        plot!(panel, periods, exp.(h_true[:, i] ./ 2);
              linewidth = 2, color = :black, linestyle = :dash,
              label = i == 1 ? "truth" : "")

        hline!(panel, [exp(MU_TRUE[i] / 2)];
               linewidth = 1, color = :firebrick, linestyle = :dot,
               label = i == 1 ? "exp(μ/2)" : "")

        i == n && xlabel!(panel, "t")
        panel
    end

    return plot(panels...; layout = (n, 1), size = (900, 250 * n),
                plot_title = "SV block recovery — exp(h_t / 2)", left_margin = 5Plots.mm)
end

# --------------------------------------------------------------------------------- run

Random.seed!(SEED)

params = (μ = MU_TRUE, Φ = Φ_TRUE, Ω = Ω_TRUE)
h_true, residuals = simulate_ar_volatility(N_TIME, MU_TRUE, Φ_TRUE, Ω_TRUE)

@info "sampling" n_series = N_SERIES n_time = N_TIME burnin = N_BURNIN kept = N_KEPT
elapsed = @elapsed h_kept = sample_volatility_path(residuals, params;
                                                   n_burnin = N_BURNIN, n_kept = N_KEPT)

h_mean, h_lower, h_upper = TCVAR.compute_posterior_statistics(h_kept; credible_level = CREDIBLE)

# ----------------------------------------------------------------------------- report

@printf("\nSV block recovery — n = %d, T = %d, %d draws kept (%.1f s)\n",
        N_SERIES, N_TIME, N_KEPT, elapsed)
println("h inferred alone; μ, Φ and Ω held at the truth.")

println("\nVolatility path, on the standard-deviation scale exp(h/2):")
println("  series      RMSE   mean truth    mean post   level ratio   band coverage")
for i in 1:N_SERIES
    true_sd = exp.(h_true[:, i] ./ 2)
    post_sd = exp.(h_mean[:, i] ./ 2)
    rmse = sqrt(mean((post_sd .- true_sd) .^ 2))
    coverage = mean(h_lower[:, i] .<= h_true[:, i] .<= h_upper[:, i])

    @printf("  %6d  %8.4f   %10.4f   %10.4f   %11.3f   %13.3f\n",
            i, rmse, mean(true_sd), mean(post_sd), mean(post_sd) / mean(true_sd), coverage)
end
@printf("      coverage over all series and periods: %.3f (target %.2f)\n",
        mean(h_lower .<= h_true .<= h_upper), CREDIBLE)

# The level check the mixture approximation is judged on. `mean(ĥ)` is compared with the
# realised `mean(h_true)` rather than with μ directly: at T = 500 and Φ = 0.95 the sample
# mean of a simulated path sits a stationary-sd or so away from μ, and that gap is the
# simulation's, not the block's. μ is printed alongside so both are visible. A wrong KSC
# shift or a dropped de-meaning moves `mean(ĥ) − mean(h_true)` by ≈ 1.27 or more; sampling
# noise moves it by a few hundredths.
println("\nLevel of h, on the log scale (the -1.2704 shift and the de-meaning):")
println("  series          μ   mean truth    mean post    post − truth")
for i in 1:N_SERIES
    @printf("  %6d  %9.4f   %10.4f   %10.4f   %13.4f\n",
            i, MU_TRUE[i], mean(h_true[:, i]), mean(h_mean[:, i]),
            mean(h_mean[:, i]) - mean(h_true[:, i]))
end

mkpath(OUTPUT_DIR)
figure_path = joinpath(OUTPUT_DIR, "sv_block_recovery.png")
savefig(plot_volatility_recovery(h_true, residuals, h_mean, h_lower, h_upper), figure_path)
println("\nfigure written to ", figure_path)
