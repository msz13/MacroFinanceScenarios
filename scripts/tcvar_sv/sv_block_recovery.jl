# The stochastic-volatility block on its own, against a simulated random-walk volatility.
#
#     h_t = h_{t-1} + ν_t,     ν_t ~ N(0, Ω)          (correlated across series)
#     e_t ~ N(0, diag(exp(h_t)))
#
# Two Gibbs blocks only — step 5 (`draw_stochastic_volatility`) and step 8 (the conjugate
# IW draw of Ω) of the TCVAR-SV sweep — with μ = 0 and Φ = I held fixed. Under a unit root
# μ is not identified, so it is not estimated here: the level of h is anchored by h_0
# alone, which is why the prior on h_0 has to be diffuse.
#
# This is the acceptance test for the block: it isolates the mixture constants, the
# -1.2704 shift, the log(e² + c̄) offset and the de-meaning from every other moving part,
# so a mistake in any of them shows up as a level or shape error against a known truth
# rather than being absorbed by the rest of a full sampler.
#
# Run as:  julia --project scripts/tcvar_sv/sv_block_recovery.jl

using Distributions
using LinearAlgebra
using Plots
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "..", "..", "src", "TCVAR", "TCVAR.jl"))

const OUTPUT_DIR = joinpath(@__DIR__, "output")

# ---------------------------------------------------------------------------- settings

const SEED      = 20250819
const N_SERIES  = 3
const N_TIME    = 500
const N_BURNIN  = 2_000
const N_KEPT    = 3_000
const CREDIBLE  = 0.90

# Deliberately correlated volatility innovations: the block draws the whole path jointly
# through a Carter–Kohn sweep with Q = Ω, so an implementation that quietly treated the
# series as n independent univariate SV models would still fit the marginals and only
# miss here.
const Ω_TRUE  = 0.02 * [1.0 0.5 0.2;
                        0.5 1.0 0.3;
                        0.2 0.3 1.0]
const H0_TRUE = log.([0.5, 1.0, 2.0] .^ 2)

# ------------------------------------------------------------------------------ helpers

"""Simulate `h_{0:T}` as a random walk with innovation covariance `Ω` and the residuals
`e_t ~ N(0, diag(exp(h_t)))` they generate."""
function simulate_random_walk_volatility(n_time, h0, Ω)
    n = length(h0)
    Ω_L = cholesky(Symmetric(Ω)).L

    h = zeros(n_time + 1, n)
    h[1, :] = h0
    for t in 2:(n_time + 1)
        h[t, :] = h[t-1, :] + Ω_L * randn(n)
    end

    residuals = [randn() * exp(h[t+1, i] / 2) for t in 1:n_time, i in 1:n]
    return h, residuals
end

"""Two-block Gibbs sampler over `(h, Ω)` with `μ = 0` and `Φ = I` fixed."""
function sample_volatility_block(residuals, Ω_prior; n_burnin, n_kept, h0_covariance)
    n_time, n = size(residuals)
    scale_prior = Matrix(Distributions.params(Ω_prior)[2])          # Ψ_Ω
    df_posterior = n_time + Distributions.params(Ω_prior)[1]        # T + ν_Ω

    unit_root = Matrix(1.0I, n, n)
    zero_mean = zeros(n)

    h = zeros(n_time + 1, n)
    Ω = mean(Ω_prior)

    h_kept = zeros(n_kept, n_time + 1, n)
    Ω_kept = zeros(n_kept, n, n)

    for iteration in 1:(n_burnin + n_kept)
        # Step 5 — the volatility path, given Ω.
        h, _ = TCVAR.draw_stochastic_volatility(residuals, h, (μ = zero_mean, Φ = unit_root, Ω = Ω);
                                                h0_covariance = h0_covariance)

        # Step 8 — the volatility covariance, given the path. With Φ = I and μ = 0 the
        # innovations are the first differences of h, which is exactly what
        # `random_walk_covariance_posterior` assembles.
        Ω = rand(TCVAR.random_walk_covariance_posterior(h, scale_prior, df_posterior))

        if iteration > n_burnin
            kept = iteration - n_burnin
            h_kept[kept, :, :] = h
            Ω_kept[kept, :, :] = Ω
        end
    end

    return h_kept, Ω_kept
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

        i == n && xlabel!(panel, "t")
        panel
    end

    return plot(panels...; layout = (n, 1), size = (900, 250 * n),
                plot_title = "SV block recovery — exp(h_t / 2)", left_margin = 5Plots.mm)
end

# --------------------------------------------------------------------------------- run

Random.seed!(SEED)

h_true, residuals = simulate_random_walk_volatility(N_TIME, H0_TRUE, Ω_TRUE)

# μ is not estimated, so the whole level of h has to come out of the data through h_0;
# the prior on it is correspondingly diffuse.
Ω_prior = TCVAR.sv_priors(N_SERIES).volatility_covariance
h0_covariance = Matrix(10.0I, N_SERIES, N_SERIES)

@info "sampling" n_series = N_SERIES n_time = N_TIME burnin = N_BURNIN kept = N_KEPT
elapsed = @elapsed h_kept, Ω_kept = sample_volatility_block(residuals, Ω_prior;
                                                            n_burnin = N_BURNIN,
                                                            n_kept = N_KEPT,
                                                            h0_covariance = h0_covariance)

h_mean, h_lower, h_upper = TCVAR.compute_posterior_statistics(h_kept; credible_level = CREDIBLE)

# ----------------------------------------------------------------------------- report

@printf("\nSV block recovery — n = %d, T = %d, %d draws kept (%.1f s)\n",
        N_SERIES, N_TIME, N_KEPT, elapsed)

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
@printf("  %s coverage over all series and periods: %.3f (target %.2f)\n",
        " " ^ 4, mean(h_lower .<= h_true .<= h_upper), CREDIBLE)

# `oracle` is the step-8 posterior mean conditional on the *true* volatility path: what
# the Ω draw would return if step 5 were replaced by an omniscient one. Printing it next
# to the estimate separates the two things a discrepancy could mean — an error in the
# block, or a T that simply does not identify Ω. At T = 500 the off-diagonals are the
# second: each h_it is observed through a single squared residual, i.e. with log-χ² noise
# of variance π²/2 ≈ 4.93 against an innovation variance of 0.02, so the drawn paths carry
# far less information about their co-movement than the T periods suggest. The estimate
# converges on the oracle (and on the truth) by T ≈ 8000.
Ω_oracle = mean(TCVAR.random_walk_covariance_posterior(
    h_true, Matrix(Distributions.params(Ω_prior)[2]),
    N_TIME + Distributions.params(Ω_prior)[1]))

println("\nΩ — volatility innovation covariance:")
println("     element        true      oracle        mean      median          5%         95%")
for i in 1:N_SERIES, j in 1:i
    draws = Ω_kept[:, i, j]
    @printf("  Ω[%d,%d]      %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f\n",
            i, j, Ω_TRUE[i, j], Ω_oracle[i, j], mean(draws), median(draws),
            quantile(draws, 0.05), quantile(draws, 0.95))
end
println("  (oracle = the same IW draw conditioned on the true h path — the ceiling this " *
        "T allows)")

mkpath(OUTPUT_DIR)
figure_path = joinpath(OUTPUT_DIR, "sv_block_recovery.png")
savefig(plot_volatility_recovery(h_true, residuals, h_mean, h_lower, h_upper), figure_path)
println("\nfigure written to ", figure_path)
