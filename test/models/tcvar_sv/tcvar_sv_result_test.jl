using Test
using Distributions
using LinearAlgebra
using Random
using Statistics
using FlexiChains

isdefined(Main, :TCVAR) || include(joinpath(@__DIR__, "..", "..", "..", "src", "TCVAR", "TCVAR.jl"))
isdefined(Main, :tcvar_sv_test_priors) || include(joinpath(@__DIR__, "..", "..", "tcvar_test_utils.jl"))

# TCVAR members are reached as `TCVAR.f` rather than via `using .TCVAR` — see the note in
# tcvar_test_utils.jl.
isdefined(TCVAR, :TCVarSVResult) || error(
    "The TCVAR module loaded in this session predates the TCVAR-SV result type. " *
    "Restart the Julia session (or REPL / IDE worker) and re-run.")

"""Raw Gibbs-draw arrays of a TCVAR-SV run, filled with values that are *distinct per
draw and per element* — every entry is `draw + element/1000` — so that a transposed
reshape or a mis-ordered `hcat` in `build_result` shows up as a wrong number rather than
as a plausible one."""
function fake_sv_draws(; n_draws, n_time, n, nt, p)
    k = n * p
    label(s, offsets...) = s + sum(offsets) / 1000

    trend_states = [label(s, t, 100u) for s in 1:n_draws, t in 1:(n_time + 1), u in 1:nt]
    cycle_states = [label(s, t, 100i) for s in 1:n_draws, t in 1:(n_time + p), i in 1:n]
    volatilities = [label(s, t, 100i) for s in 1:n_draws, t in 1:(n_time + 1), i in 1:n]

    trend_covariance      = [label(s, 10i, j) for s in 1:n_draws, i in 1:nt, j in 1:nt]
    betas                 = [label(s, e) for s in 1:n_draws, e in 1:(k * n)]
    simultaneity          = [label(s, 10i, j) for s in 1:n_draws, i in 1:n, j in 1:n]
    volatility_mean       = [label(s, i) for s in 1:n_draws, i in 1:n]
    volatility_ar         = [label(s, 10i, j) for s in 1:n_draws, i in 1:n, j in 1:n]
    volatility_covariance = [label(s, 10i, j) for s in 1:n_draws, i in 1:n, j in 1:n]

    return (; trend_states, cycle_states, volatilities, trend_covariance, betas,
            simultaneity, volatility_mean, volatility_ar, volatility_covariance)
end

@testset "models/tcvar_sv/tcvar_sv_result" begin

    n, nt, p, n_time = 3, 3, 2, 12
    n_draws, burnin, thin = 20, 8, 2
    k = n * p
    trend_mapping = Matrix(1.0I, n, nt)

    priors = tcvar_sv_test_priors(; n = n, nt = nt, p = p)
    model = TCVAR.TCVARSV(trend_mapping, priors, n_time)
    draws = fake_sv_draws(; n_draws = n_draws, n_time = n_time, n = n, nt = nt, p = p)

    # The sampler leaves the skeleton filled in; build_result has to hand it back empty.
    TCVAR.update_tc_var_sv!(model.ssm, ones(n, k), ones(nt, nt), ones(n_time, n, n), nt, n, p)

    result = TCVAR.build_result(model, draws.trend_states, draws.cycle_states,
                                draws.volatilities, draws.trend_covariance, draws.betas,
                                draws.simultaneity, draws.volatility_mean,
                                draws.volatility_ar, draws.volatility_covariance,
                                burnin, thin)

    kept = burnin+1:thin:n_draws
    n_kept = length(kept)

    @testset "burn-in and thinning" begin
        @test result isa TCVAR.TCVarSVResult
        @test size(result.trend_states) == (n_kept, n_time + 1, nt)
        @test size(result.cycle_states) == (n_kept, n_time + p, n)
        @test size(result.volatilities) == (n_kept, n_time + 1, n)
        @test result.trend_states == draws.trend_states[kept, :, :]
        @test result.cycle_states == draws.cycle_states[kept, :, :]
        @test result.volatilities == draws.volatilities[kept, :, :]
    end

    @testset "every parameter is reshaped back to its own draw" begin
        # The hcat order in build_result and the FlexiChain key spec have to agree; this
        # is what catches them drifting apart.
        for (draw_index, s) in enumerate(kept)
            @test result.params[@varname(Στ)][draw_index] == draws.trend_covariance[s, :, :]
            @test result.params[@varname(β)][draw_index] == reshape(draws.betas[s, :], k, n)
            @test result.params[@varname(A₀)][draw_index] == draws.simultaneity[s, :, :]
            @test result.params[@varname(μ)][draw_index] == draws.volatility_mean[s, :]
            @test result.params[@varname(Φ)][draw_index] == draws.volatility_ar[s, :, :]
            @test result.params[@varname(Ω)][draw_index] == draws.volatility_covariance[s, :, :]
        end

        @test size(result.params[@varname(β)][1]) == (k, n)
        @test result.params[@varname(μ)][1] isa Vector      # μ is a vector, not an n×1
        @test length(result.params[@varname(μ)][1]) == n

        # There is no Σc under stochastic volatility.
        @test_throws Exception result.params[@varname(Σc)]
    end

    @testset "the returned model carries the empty skeleton" begin
        @test all(iszero, result.model.ssm.Q)
        @test all(iszero, result.model.ssm.T[nt+n*(p-1)+1:end, nt+1:end])
        # The constant structure survives.
        @test result.model.ssm.Z == TCVAR.tc_var(trend_mapping; p = p).Z
    end

    @testset "posterior_mean" begin
        summary = TCVAR.posterior_mean(result)

        @test keys(summary) == (:Στ, :β, :A₀, :μ, :Φ, :Ω)
        @test summary.Στ ≈ dropdims(mean(draws.trend_covariance[kept, :, :], dims = 1), dims = 1)
        @test summary.β  ≈ reshape(vec(mean(draws.betas[kept, :], dims = 1)), k, n)
        @test summary.A₀ ≈ dropdims(mean(draws.simultaneity[kept, :, :], dims = 1), dims = 1)
        @test summary.μ  ≈ vec(mean(draws.volatility_mean[kept, :], dims = 1))
        @test summary.Φ  ≈ dropdims(mean(draws.volatility_ar[kept, :, :], dims = 1), dims = 1)
        @test summary.Ω  ≈ dropdims(mean(draws.volatility_covariance[kept, :, :], dims = 1), dims = 1)

        # The keys are exactly the ones simulate_tcvar_sv consumes.
        @test issubset((:Στ, :β, :A₀, :μ, :Φ, :Ω), keys(summary))
    end

    @testset "posterior_volatilities is on the exp(h/2) scale" begin
        mean_sd, lower, upper = TCVAR.posterior_volatilities(result)

        @test size(mean_sd) == (n_time + 1, n)
        @test mean_sd ≈ dropdims(mean(exp.(result.volatilities ./ 2), dims = 1), dims = 1)
        @test all(lower .<= mean_sd .<= upper)
        @test all(lower .> 0)
    end

    @testset "simulate_scenarios starts from the posterior-mean terminal state" begin
        n_scenarios, n_steps = 4, 6

        Random.seed!(3)
        states, observations, volatilities =
            TCVAR.simulate_scenarios(result, n_scenarios, n_steps)

        @test size(states) == (n_scenarios, n_steps, nt + k)
        @test size(observations) == (n_scenarios, n_steps, n)
        @test size(volatilities) == (n_scenarios, n_steps + 1, n)

        # h starts at the posterior-mean terminal log volatility, common to all scenarios.
        h_start = vec(mean(result.volatilities[:, end, :], dims = 1))
        @test all(volatilities[s, 1, :] == h_start for s in 1:n_scenarios)

        # The model in the result is untouched by the simulation.
        @test all(iszero, result.model.ssm.Q)
    end
end
