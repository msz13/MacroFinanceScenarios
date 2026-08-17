using Test
using Distributions
using LinearAlgebra
using Random
using Statistics

isdefined(Main, :TCVAR) || include(joinpath(@__DIR__, "..", "src", "TCVAR", "TCVAR.jl"))
using .TCVAR
using FlexiChains: @varname

isdefined(Main, :tcvar_test_priors) || include(joinpath(@__DIR__, "tcvar_test_utils.jl"))

@testset "TCVAR priors NamedTuple" begin

    @testset "var_priors returns three consistent objects" begin
        n, p, ψ = 3, 2, [2.0, 1.0, 0.5]
        Σ_prior, β_prior, c0_prior = var_priors(0.2, p, ψ; δ = zeros(n))

        @test Σ_prior isa InverseWishart
        @test size(Σ_prior) == (n, n)
        d, Ψ = params(Σ_prior)
        @test d == n + 2
        @test Matrix(Ψ) ≈ diagm(ψ)
        @test mean(Σ_prior) ≈ diagm(ψ)          # d = n+2 ⇒ E[Σ] = Ψ

        @test β_prior isa MinnesotaPrior
        @test β_prior.n == n
        @test β_prior.p == p
        @test β_prior.k == n * p
        @test !TCVAR.has_intercept(β_prior)
        @test iszero(β_prior.Φ₀)                # δ = 0 ⇒ white-noise prior mean

        @test c0_prior isa MvNormal
        @test length(c0_prior) == n * p
        @test mean(c0_prior) == zeros(n * p)
        # Φ₀ = 0 ⇒ the stationary companion covariance is exactly I_p ⊗ E[Σ]
        @test Matrix(cov(c0_prior)) ≈ kron(Matrix(I, p, p), mean(Σ_prior))
    end

    @testset "lag-order accessors (oldest-lag-first, intercept dropped)" begin
        n, p, λ, ψ = 2, 2, 0.5, [4.0, 1.0]
        Σ_prior = InverseWishart(float(n + 2), diagm(ψ))
        pr = MinnesotaPrior(λ, p, Σ_prior; δ = [0.9, 0.8], intercept = true)

        @test pr.k == n * p + 1
        @test TCVAR.has_intercept(pr)

        # own first lag δᵢ sits in the *last* (newest) block of the reversed layout
        @test prior_var_coeff(pr) == [0.0 0.0 0.9 0.0
                                      0.0 0.0 0.0 0.8]

        # λ²/(s² σ̄ⱼ) with σ̄ = [4, 1], lag-2 block first, no intercept entry
        @test diag(prior_row_covariance(pr)) ≈ [0.015625, 0.0625, 0.0625, 0.25]
        @test size(prior_row_covariance(pr)) == (n * p, n * p)
    end

    @testset "constructor infers n and p from cycle_β" begin
        n, nt, p = 2, 2, 3
        priors = tcvar_test_priors(; n = n, nt = nt, p = p)
        model = TCVAR.TCVAR(Matrix{Float64}(I, n, nt), priors)

        @test size(model.ssm.T) == (nt + n * p, nt + n * p)
        @test size(model.ssm.Z) == (n, nt + n * p)
        @test iszero(model.ssm.Q)               # skeleton starts zeroed
        @test model.variable_names == ["y1", "y2"]
        @test model.trend_names == ["τ1", "τ2"]

        # stored verbatim — no `cycle` key merged in
        @test keys(model.priors) == (:initial_trend, :initial_cycle, :trend_covariance,
                                     :cycle_covariance, :cycle_β)
        @test model.priors === priors
    end

    @testset "constructor error paths" begin
        n, nt, p = 2, 2, 2
        priors = tcvar_test_priors(; n = n, nt = nt, p = p)
        mapping = Matrix{Float64}(I, n, nt)

        incomplete = Base.structdiff(priors, NamedTuple{(:cycle_covariance,)})
        @test_throws ArgumentError TCVAR.TCVAR(mapping, incomplete)

        for len in (n * p - 1, n * p + 1)
            bad = merge(priors, (initial_cycle = MvNormal(zeros(len), diagm(ones(len))),))
            @test_throws DimensionMismatch TCVAR.TCVAR(mapping, bad)
        end

        bad_trend_cov = merge(priors,
            (trend_covariance = InverseWishart(float(nt + 3), diagm(ones(nt + 1))),))
        @test_throws DimensionMismatch TCVAR.TCVAR(mapping, bad_trend_cov)

        wide = tcvar_test_priors(; n = n + 1, nt = nt, p = p)
        @test_throws DimensionMismatch TCVAR.TCVAR(mapping, wide)

        @test_throws DimensionMismatch TCVAR.TCVAR(mapping, priors;
                                                   variable_names = ["a", "b", "c"])
        @test_throws DimensionMismatch TCVAR.TCVAR(mapping, priors; trend_names = ["t"])
    end

    @testset "sampler shapes and determinism" begin
        n, nt, p = 2, 2, 1
        T = 60
        priors = tcvar_test_priors(; n = n, nt = nt, p = p)
        model = TCVAR.TCVAR(Matrix{Float64}(I, n, nt), priors)

        Random.seed!(11)
        _, obs = simulate_scenarios(model,
            (Στ = diagm([0.01, 0.02]), β = [0.6 0.1; 0.0 0.5], Σc = [1.0 0.2; 0.2 0.5]),
            [1.0, 2.0, 0.0, 0.0], 1, T)
        data = convert(Matrix{Union{Missing,Float64}}, obs[1, :, :])

        Random.seed!(42)
        result = gibbs_sampler(model, data; burnin = 20, n_samples = 20, thin = 2)

        n_kept = length(21:2:40)                # 10 retained draws
        @test n_kept == 10
        @test size(result.trend_states) == (n_kept, T + 1, nt)
        @test size(result.cycle_states) == (n_kept, T + p, n)

        pm = posterior_mean(result)
        @test size(pm.Στ) == (nt, nt)
        @test size(pm.β) == (n * p, n)
        @test size(pm.Σc) == (n, n)
        @test issymmetric(pm.Στ) && isposdef(pm.Στ)
        @test issymmetric(pm.Σc) && isposdef(pm.Σc)
        @test TCVAR.is_stationary(collect(pm.β'), n, p)

        # the returned model carries the reset skeleton again
        @test iszero(result.model.ssm.Q)
        @test iszero(result.model.ssm.T[nt+n*(p-1)+1:end, nt+1:end])

        Random.seed!(42)
        again = gibbs_sampler(model, data; burnin = 20, n_samples = 20, thin = 2)
        @test posterior_mean(again).β == pm.β
        @test again.trend_states == result.trend_states

        Random.seed!(43)
        other = gibbs_sampler(model, data; burnin = 20, n_samples = 20, thin = 2)
        @test posterior_mean(other).β != pm.β
    end

    @testset "simulate_scenarios" begin
        n, nt, p = 2, 2, 2
        T = 50
        priors = tcvar_test_priors(; n = n, nt = nt, p = p)
        model = TCVAR.TCVAR(Matrix{Float64}(I, n, nt), priors)
        n_states = nt + n * p

        truth = (Στ = diagm([0.01, 0.02]),
                 β  = [0.2 0.0; 0.0 0.1; 0.5 0.1; 0.0 0.4],
                 Σc = [1.0 0.2; 0.2 0.5])
        initial_state = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0]

        Random.seed!(5)
        states, obs = simulate_scenarios(model, truth, initial_state, 5, 10)
        @test size(states) == (5, 10, n_states)
        @test size(obs) == (5, 10, n)
        @test all(states[s, 1, :] == initial_state for s in 1:5)

        Random.seed!(5)
        states2, obs2 = simulate_scenarios(model, truth, initial_state, 5, 10)
        @test states2 == states && obs2 == obs

        @test iszero(model.ssm.Q)               # the skeleton was not mutated
        @test_throws DimensionMismatch simulate_scenarios(model, truth,
                                                          initial_state[1:end-1], 5, 10)

        Random.seed!(7)
        _, sim = simulate_scenarios(model, truth, initial_state, 1, T)
        data = convert(Matrix{Union{Missing,Float64}}, sim[1, :, :])
        result = gibbs_sampler(model, data; burnin = 20, n_samples = 20, thin = 2)

        rstates, robs = simulate_scenarios(result, 7, 12)
        @test size(rstates) == (7, 12, n_states)
        @test size(robs) == (7, 12, n)
        @test iszero(result.model.ssm.Q)
    end
end
