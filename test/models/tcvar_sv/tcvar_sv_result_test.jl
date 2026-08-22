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

"""True-parameter tuple used by the simulator tests: a stable diagonal cycle VAR, three
genuinely different volatility levels and a non-trivial `A₀`."""
function simulator_params(; n = 3, nt = 3, p = 1, Ω = 0.02 * [1.0 0.5 0.2
                                                              0.5 1.0 0.3
                                                              0.2 0.3 1.0])
    return (Στ = diagm(fill(0.01, nt)),
            β  = vcat(zeros(n * (p - 1), n), 0.6 * Matrix(1.0I, n, n)),
            A₀ = [1.0 0.0 0.0; 0.4 1.0 0.0; -0.2 0.3 1.0],
            μ  = log.([0.5, 1.0, 2.0] .^ 2),
            Φ  = Matrix(0.95I, n, n),
            Ω  = Matrix(float.(Ω)))
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

        # The keys are exactly the ones simulate_scenarios consumes.
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
        # The draws above are deliberate nonsense — good for catching a transposed
        # reshape, useless as a covariance to simulate from — so this one result is built
        # from well-formed parameters, repeated across draws.
        truth = simulator_params(; n = n, nt = nt, p = p)
        n_sim = 4
        stack(M)  = [M[i, j] for _ in 1:n_sim, i in axes(M, 1), j in axes(M, 2)]
        stack_vec(v) = [v[i] for _ in 1:n_sim, i in eachindex(v)]

        Random.seed!(11)
        sim_result = TCVAR.build_result(
            TCVAR.TCVARSV(trend_mapping, priors, n_time),
            randn(n_sim, n_time + 1, nt),                       # trend path
            randn(n_sim, n_time + p, n),                        # cycle path
            randn(n_sim, n_time + 1, n) ./ 10 .+ reshape(truth.μ, 1, 1, n),  # h path
            stack(truth.Στ), repeat(vec(truth.β)', n_sim), stack(truth.A₀),
            stack_vec(truth.μ), stack(truth.Φ), stack(truth.Ω), 0, 1)

        n_scenarios, n_steps = 4, 6

        Random.seed!(3)
        states, observations, volatilities =
            TCVAR.simulate_scenarios(sim_result, n_scenarios, n_steps)

        # `sample` puts the starting state in the first row, so all three arrays share
        # one period axis.
        @test size(states) == (n_scenarios, n_steps, nt + k)
        @test size(observations) == (n_scenarios, n_steps, n)
        @test size(volatilities) == (n_scenarios, n_steps, n)

        # Every scenario starts from the common posterior-mean terminal state: the last
        # trend state, the last p cycle states oldest-lag-first, and h_T.
        trend_start = vec(mean(sim_result.trend_states[:, end, :], dims = 1))
        cycle_mean  = dropdims(mean(sim_result.cycle_states, dims = 1), dims = 1)
        cycle_start = vec(permutedims(cycle_mean[end-p+1:end, :]))
        h_start     = vec(mean(sim_result.volatilities[:, end, :], dims = 1))

        for s in 1:n_scenarios
            @test states[s, 1, :] == [trend_start; cycle_start]
            @test volatilities[s, 1, :] == h_start
        end

        # The model in the result is untouched by the simulation.
        @test all(iszero, sim_result.model.ssm.Q)
        @test all(iszero, sim_result.model.ssm.T[nt+n*(p-1)+1:end, nt+1:end])
    end

    # ---------------------------------------------------------------------------------
    # The parameter-driven form, which the recovery scripts use to generate data.
    # ---------------------------------------------------------------------------------

    @testset "simulate_scenarios at explicit parameters" begin
        lags, n_steps = 1, 120
        sv_model = TCVAR.TCVARSV(trend_mapping, tcvar_sv_test_priors(; n = n, nt = nt, p = lags),
                                 n_steps)
        params = simulator_params(; p = lags)

        Random.seed!(4321)
        states, observations, volatilities =
            TCVAR.simulate_scenarios(sv_model, params, zeros(nt + n * lags), 1, n_steps)

        @test size(states) == (1, n_steps, nt + n * lags)
        @test size(observations) == (1, n_steps, n)
        @test size(volatilities) == (1, n_steps, n)
        @test all(isfinite, states) && all(isfinite, volatilities)

        # h_1 is the starting value, and defaults to μ.
        @test volatilities[1, 1, :] == params.μ
        @test states[1, 1, :] == zeros(nt + n * lags)

        # The observation is y_t = Λ τ_t + c_t up to the 1e-4 jitter `sample` adds to
        # H = eps()·I (a standard deviation of 1e-2).
        implied = states[1, :, :] * sv_model.ssm.Z'
        @test maximum(abs, observations[1, :, :] - implied) < 0.1

        # The volatility path is the VAR(1) it claims to be: what is left after removing
        # μ + Φ(h_{t-1} − μ) has the covariance Ω.
        h = volatilities[1, :, :]
        innovations = h[2:end, :] .- params.μ' .- (h[1:end-1, :] .- params.μ') * params.Φ'
        @test size(innovations) == (n_steps - 1, n)
        @test cov(innovations) ≈ params.Ω rtol = 0.35

        @test_throws ArgumentError TCVAR.simulate_scenarios(sv_model, params,
                                                            zeros(nt + n * lags), 1, 0)
        @test_throws DimensionMismatch TCVAR.simulate_scenarios(sv_model, params, zeros(nt),
                                                                1, n_steps)
        @test_throws DimensionMismatch TCVAR.simulate_scenarios(
            sv_model, params, zeros(nt + n * lags), 1, n_steps;
            initial_volatility = zeros(n + 1))
        @test_throws DimensionMismatch TCVAR.simulate_scenarios(
            sv_model, merge(params, (β = zeros(n * (lags + 1), n),)),
            zeros(nt + n * lags), 1, n_steps)
    end

    @testset "Ω = 0 reproduces a homoskedastic TCVAR path" begin
        # Switch the volatility innovations off and the simulator has to collapse onto a
        # constant-Σ trend-cycle VAR at Σ = A₀⁻¹ diag(exp(μ)) A₀⁻ᵀ. psd_factor is what
        # makes "off" mean exactly off for the volatility path — the state draw itself
        # still picks up the 1e-4 jitter `sample` adds to Q, which is negligible against
        # these variances but is why the trend below is not compared exactly.
        lags, long = 1, 20_000
        sv_model = TCVAR.TCVARSV(trend_mapping, tcvar_sv_test_priors(; n = n, nt = nt, p = lags),
                                 long)
        params = simulator_params(; p = lags, Ω = zeros(n, n))

        Random.seed!(99)
        states, _, volatilities =
            TCVAR.simulate_scenarios(sv_model, params, zeros(nt + n * lags), 1, long)

        # Every period sits at h_1 = μ exactly, so H_t — and therefore Σ_t — is constant.
        @test all(volatilities[1, t, :] == params.μ for t in 1:long)

        # The cycle innovations are then homoskedastic at Σ = A₀⁻¹ diag(exp(μ)) A₀⁻ᵀ.
        cycle = states[1, :, nt+1:end]
        innovations = cycle[2:end, :] - cycle[1:end-1, :] * params.β
        Σ_true = params.A₀ \ diagm(exp.(params.μ)) / params.A₀'
        @test cov(innovations) ≈ Σ_true rtol = 0.06

        # And the trends are the random walk they always were, at Στ plus the jitter.
        trends = states[1, :, 1:nt]
        @test cov(diff(trends, dims = 1)) ≈ params.Στ + 1e-4I rtol = 0.06
    end

    @testset "the n_scenarios form stacks independent paths" begin
        lags, n_steps, n_scenarios = 1, 120, 5
        sv_model = TCVAR.TCVARSV(trend_mapping, tcvar_sv_test_priors(; n = n, nt = nt, p = lags),
                                 n_steps)
        params = simulator_params(; p = lags)

        Random.seed!(7)
        states, observations, volatilities =
            TCVAR.simulate_scenarios(sv_model, params, zeros(nt + n * lags), n_scenarios, n_steps)

        @test size(states) == (n_scenarios, n_steps, nt + n * lags)
        @test size(observations) == (n_scenarios, n_steps, n)
        @test size(volatilities) == (n_scenarios, n_steps, n)

        # Same start, different draws.
        @test all(volatilities[s, 1, :] == params.μ for s in 1:n_scenarios)
        @test states[1, :, :] != states[2, :, :]
        @test volatilities[1, :, :] != volatilities[2, :, :]

        # Scenario 1 is exactly the one-scenario simulation from the same seed.
        Random.seed!(7)
        single, _, _ = TCVAR.simulate_scenarios(sv_model, params, zeros(nt + n * lags), 1, n_steps)
        @test states[1, :, :] == single[1, :, :]
    end

    @testset "p = 2 uses the whole companion" begin
        lags, long = 2, 20_000
        sv_model = TCVAR.TCVARSV(trend_mapping, tcvar_sv_test_priors(; n = n, nt = nt, p = lags),
                                 long)

        # Coefficients on the *oldest* lag only: c_t depends on c_{t-2}, not on c_{t-1}.
        params = merge(simulator_params(; p = lags),
                       (β = vcat(0.5 * Matrix(1.0I, n, n), zeros(n, n)),
                        Ω = zeros(n, n)))

        Random.seed!(5)
        states, _, _ = TCVAR.simulate_scenarios(sv_model, params, zeros(nt + n * lags), 1, long)

        cycle = states[1, :, nt+1:end]
        # The companion shifts: the first block of ξ_t is the second block of ξ_{t-1} —
        # up to the jitter `sample` adds to the (exactly zero) shift block of Q.
        @test maximum(abs, cycle[2:end, 1:n] - cycle[1:end-1, n+1:end]) < 0.1

        # And the oldest-lag-first layout is the one the coefficients are read in: with
        # the coefficients on the oldest lag only, what is left after subtracting
        # 0.5·c_{t-2} is the innovation — and c_{t-2} is the *first* companion block one
        # row earlier, not the first block of the same row (that one is c_{t-1}).
        c_t = cycle[:, n+1:end]
        innovations = c_t[2:end, :] - 0.5 * cycle[1:end-1, 1:n]
        Σ_true = params.A₀ \ diagm(exp.(params.μ)) / params.A₀'
        @test cov(innovations) ≈ Σ_true rtol = 0.06
    end
end
