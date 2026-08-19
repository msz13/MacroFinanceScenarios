using Test
using Distributions
using LinearAlgebra
using Statistics

isdefined(Main, :TCVAR) || include(joinpath(@__DIR__, "..", "..", "..", "src", "TCVAR", "TCVAR.jl"))

# TCVAR members are reached as `TCVAR.f` rather than via `using .TCVAR` — see the note in
# tcvar_test_utils.jl.
isdefined(TCVAR, :sv_priors) || error(
    "The TCVAR module loaded in this session predates the stochastic-volatility block. " *
    "Restart the Julia session (or REPL / IDE worker) and re-run.")

@testset "common/sv/sv_priors" begin

    @testset "defaults" begin
        n = 3
        priors = TCVAR.sv_priors(n)

        @test keys(priors) == (:volatility_mean, :volatility_ar, :volatility_covariance,
                               :simultaneity)

        # μ centred on log(0.1²) — a prior median innovation sd of 0.1 — and wide.
        @test mean(priors.volatility_mean) ≈ fill(log(0.1^2), n)
        @test Matrix(cov(priors.volatility_mean)) ≈ Matrix(4.0I, n, n)

        @test mean(priors.volatility_ar) ≈ fill(0.8, n)
        @test Matrix(cov(priors.volatility_ar)) ≈ Matrix(0.04I, n, n)

        # The IW is stated by its *scale*, so the prior mean is what has to be checked:
        # IW(ν, Ψ) has mean Ψ/(ν − n − 1), and the intended mean is 0.2²·I.
        @test mean(priors.volatility_covariance) ≈ Matrix(0.04I, n, n)
        @test priors.volatility_covariance.df == n + 11

        @test length(mean(priors.simultaneity)) == n * (n - 1) ÷ 2
        @test mean(priors.simultaneity) ≈ zeros(3)
        @test Matrix(cov(priors.simultaneity)) ≈ Matrix(10.0I, 3, 3)
    end

    @testset "keywords" begin
        priors = TCVAR.sv_priors(2; volatility_level = 0.5, mean_sd = 1.5, ar_mean = 0.6,
                                 ar_sd = 0.1, covariance_sd = 0.3, covariance_df = 30.0,
                                 simultaneity_variance = 4.0)

        @test mean(priors.volatility_mean) ≈ fill(log(0.25), 2)
        @test Matrix(cov(priors.volatility_mean)) ≈ Matrix(2.25I, 2, 2)
        @test mean(priors.volatility_ar) ≈ fill(0.6, 2)
        @test Matrix(cov(priors.volatility_ar)) ≈ Matrix(0.01I, 2, 2)
        @test mean(priors.volatility_covariance) ≈ Matrix(0.09I, 2, 2)
        @test Matrix(cov(priors.simultaneity)) ≈ Matrix(4.0I, 1, 1)
    end

    @testset "edge cases and validation" begin
        # A single series has no free simultaneity element; the block is empty, not absent.
        @test length(mean(TCVAR.sv_priors(1).simultaneity)) == 0

        @test_throws ArgumentError TCVAR.sv_priors(0)
        @test_throws ArgumentError TCVAR.sv_priors(3; volatility_level = 0.0)
        @test_throws ArgumentError TCVAR.sv_priors(3; mean_sd = -1.0)
        @test_throws ArgumentError TCVAR.sv_priors(3; ar_sd = 0.0)
        @test_throws ArgumentError TCVAR.sv_priors(3; covariance_sd = 0.0)
        @test_throws ArgumentError TCVAR.sv_priors(3; simultaneity_variance = 0.0)

        # ν must exceed n + 1 for the IW to have a mean at all.
        @test_throws ArgumentError TCVAR.sv_priors(3; covariance_df = 4.0)
    end

    @testset "the μ prior is unconditional" begin
        # D3: the prior on μ used to be a closure over the current (ρ, σ²) draws, i.e. the
        # ergodic variance σ²/(1−ρ²). That is not a prior — it moves with the sampler and
        # breaks the conjugacy of the μ update — so it is a plain distribution now.
        @test TCVAR.sv_priors(2).volatility_mean isa MvNormal
    end

end
