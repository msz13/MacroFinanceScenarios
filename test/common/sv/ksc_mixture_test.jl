using Test
using Distributions
using LinearAlgebra
using Random
using Statistics

isdefined(Main, :TCVAR) || include(joinpath(@__DIR__, "..", "..", "..", "src", "TCVAR", "TCVAR.jl"))

# TCVAR members are reached as `TCVAR.f` rather than via `using .TCVAR` — see the note in
# tcvar_test_utils.jl.
isdefined(TCVAR, :KSC_MIXTURE) || error(
    "The TCVAR module loaded in this session predates the stochastic-volatility block. " *
    "Restart the Julia session (or REPL / IDE worker) and re-run.")

# The mixture is the one place in the SV block where a wrong constant produces a plausible
# but wrong answer rather than an error: forget the -1.2704 shift and every volatility
# comes out a factor exp(1.2704/2) ≈ 1.89 too large, with nothing else complaining. So the
# constants are checked against the moments and the density they are supposed to
# reproduce, not merely transcribed.

"""Exact density of `x = log u`, `u ~ χ²₁`."""
log_chisq1_pdf(x) = exp(x / 2 - exp(x) / 2) / sqrt(2π)

"""Density of the seven-component KSC approximation at `x`."""
ksc_pdf(x) = sum(TCVAR.KSC_MIXTURE.probabilities[j] *
                 pdf(Normal(TCVAR.KSC_MIXTURE.means[j],
                            TCVAR.KSC_MIXTURE.standard_deviations[j]), x)
                 for j in eachindex(TCVAR.KSC_MIXTURE.means))

@testset "common/sv/ksc_mixture" begin

    mixture = TCVAR.KSC_MIXTURE

    @testset "constants" begin
        @test length(mixture.probabilities) == 7
        @test length(mixture.means) == 7
        @test length(mixture.variances) == 7

        @test sum(mixture.probabilities) ≈ 1.0
        @test all(>(0), mixture.probabilities)
        @test all(>(0), mixture.variances)
        @test mixture.standard_deviations ≈ sqrt.(mixture.variances)
        @test mixture.log_scales ≈ log.(mixture.probabilities ./ mixture.standard_deviations)

        # The stored means are the published table shifted by E[log χ²₁].
        raw = [-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819]
        @test mixture.means ≈ raw .+ TCVAR.LOG_CHISQ1_MEAN

        # The shift is E[log χ²₁] = ψ(1/2) + log 2, recovered here by integrating the
        # exact density rather than restated as the same decimal.
        step = 0.001
        grid = -30:step:8
        @test TCVAR.LOG_CHISQ1_MEAN ≈ sum(x * log_chisq1_pdf(x) for x in grid) * step atol = 1e-3
    end

    @testset "moments match log χ²₁" begin
        q, m, v² = mixture.probabilities, mixture.means, mixture.variances

        mixture_mean = dot(q, m)
        mixture_variance = dot(q, v² .+ m .^ 2) - mixture_mean^2

        # E[log χ²₁] = ψ(1/2) + log 2 = -1.2704 ...
        @test mixture_mean ≈ TCVAR.LOG_CHISQ1_MEAN atol = 1e-4
        # ... and Var[log χ²₁] = ψ'(1/2) = π²/2.
        @test mixture_variance ≈ π^2 / 2 atol = 1e-3
    end

    @testset "density approximates log χ²₁" begin
        step = 0.02
        # Wide enough to hold component 1 (mean -11.4, sd 2.41) out to its own tails, so
        # the normalisation check below is not measuring a truncated grid.
        grid = -30:step:8

        # KSC's seven components are accurate to about 1e-2 in density and 4e-2 in total
        # variation; the tolerances here would catch a transposed or mis-scaled component
        # while leaving the published approximation error alone.
        @test maximum(abs(ksc_pdf(x) - log_chisq1_pdf(x)) for x in grid) < 0.02
        @test sum(abs(ksc_pdf(x) - log_chisq1_pdf(x)) for x in grid) * step < 0.05

        # Both densities integrate to one over the grid, so the comparison is between two
        # normalised objects rather than an accidental scale difference.
        @test sum(ksc_pdf(x) for x in grid) * step ≈ 1.0 atol = 1e-3
        @test sum(log_chisq1_pdf(x) for x in grid) * step ≈ 1.0 atol = 1e-3
    end

    @testset "draw_mixture_indicators is a valid categorical" begin
        Random.seed!(2024)
        y_star = randn(50, 3) .- 1.0
        h = randn(50, 3) .* 0.2

        indicators = TCVAR.draw_mixture_indicators(y_star, h)

        @test size(indicators) == (50, 3)
        @test eltype(indicators) == Int
        @test all(j -> 1 ≤ j ≤ 7, indicators)

        @test_throws DimensionMismatch TCVAR.draw_mixture_indicators(y_star, h[1:49, :])
        @test_throws DimensionMismatch TCVAR.draw_mixture_indicators(y_star, h[:, 1:2])
    end

    @testset "indicator frequencies match the analytic weights" begin
        # At a single fixed deviation the conditional distribution of the label is known
        # in closed form; 20 000 draws pin it down to well inside a Monte Carlo error of
        # about sqrt(0.25/20000) ≈ 0.0035.
        deviation = 0.4
        weights = mixture.probabilities ./ mixture.standard_deviations .*
                  exp.(-(deviation .- mixture.means) .^ 2 ./ (2 .* mixture.variances))
        weights ./= sum(weights)

        Random.seed!(31)
        n_draws = 20_000
        drawn = TCVAR.draw_mixture_indicators(fill(deviation, n_draws, 1), zeros(n_draws, 1))
        empirical = [count(==(j), drawn) / n_draws for j in 1:7]

        @test empirical ≈ weights atol = 0.01
    end

    @testset "concentrates on the component that explains the deviation" begin
        # The far left tail is covered by component 1 alone (component 3 carries weight
        # 2e-5, component 2 sits four log-weights away), so a deviation at m₁ has to be
        # labelled 1 almost always. This is the assertion that fails first if the means
        # are permuted relative to the weights or the variances.
        Random.seed!(32)
        drawn = TCVAR.draw_mixture_indicators(fill(mixture.means[1], 5_000, 1), zeros(5_000, 1))
        @test count(==(1), drawn) / 5_000 > 0.95

        # ... and at every component's own centre the drawn frequencies reproduce the
        # analytic weights. Stated as frequencies rather than as "component j wins":
        # components 4 and 6 overlap closely enough that their analytic weights at m₄ are
        # within 6%% of each other, so which one is modal there is not a stable assertion.
        for j in eachindex(mixture.means)
            deviation = mixture.means[j]
            weights = mixture.probabilities ./ mixture.standard_deviations .*
                      exp.(-(deviation .- mixture.means) .^ 2 ./ (2 .* mixture.variances))
            weights ./= sum(weights)

            Random.seed!(100 + j)
            labels = TCVAR.draw_mixture_indicators(fill(deviation, 5_000, 1), zeros(5_000, 1))
            empirical = [count(==(k), labels) / 5_000 for k in 1:7]

            @test empirical ≈ weights atol = 0.025
        end
    end

    @testset "extreme deviations still yield a proper draw" begin
        # Weights are formed in logs and re-centred before exponentiating; without that,
        # every component underflows to zero here and the normalisation is 0/0.
        Random.seed!(33)
        indicators = TCVAR.draw_mixture_indicators(fill(-400.0, 20, 1), zeros(20, 1))
        @test all(j -> 1 ≤ j ≤ 7, indicators)
        @test all(==(1), indicators)          # the widest, left-most component
    end

end
