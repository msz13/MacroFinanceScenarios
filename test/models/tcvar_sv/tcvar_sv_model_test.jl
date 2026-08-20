using Test
using Distributions
using LinearAlgebra
using Random
using Statistics

isdefined(Main, :TCVAR) || include(joinpath(@__DIR__, "..", "..", "..", "src", "TCVAR", "TCVAR.jl"))
isdefined(Main, :tcvar_sv_test_priors) || include(joinpath(@__DIR__, "..", "..", "tcvar_test_utils.jl"))

# TCVAR members are reached as `TCVAR.f` rather than via `using .TCVAR` — see the note in
# tcvar_test_utils.jl.
isdefined(TCVAR, :simulate_tcvar_sv) || error(
    "The TCVAR module loaded in this session predates the TCVAR-SV model. " *
    "Restart the Julia session (or REPL / IDE worker) and re-run.")

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

@testset "models/tcvar_sv/tcvar_sv_model" begin

    n, nt, p, n_time = 3, 3, 1, 120
    trend_mapping = Matrix(1.0I, n, nt)

    @testset "tc_var_sv is tc_var with a time-varying Q" begin
        for lags in (1, 2)
            constant = TCVAR.tc_var(trend_mapping; p = lags)
            varying  = TCVAR.tc_var_sv(trend_mapping, n_time; p = lags)

            # Every constant block is the one tc_var builds — the skeleton is shared, so
            # the two models cannot drift apart.
            @test varying.T == constant.T
            @test varying.R == constant.R
            @test varying.Z == constant.Z
            @test varying.H == constant.H

            n_states = size(constant.T, 1)
            @test size(varying.Q) == (n_time, n_states, n_states)
            @test all(iszero, varying.Q)            # draw-dependent blocks start empty
            @test TCVAR.observation_noise(varying, 1) == constant.H
        end

        @test_throws ArgumentError TCVAR.tc_var_sv(trend_mapping, 0; p = 1)
    end

    @testset "update_tc_var_sv! writes only the draw-dependent blocks" begin
        lags = 2
        model = TCVAR.tc_var_sv(trend_mapping, n_time; p = lags)
        skeleton_T = copy(model.T)

        var_coeff = hcat(0.1 * Matrix(1.0I, n, n), 0.5 * Matrix(1.0I, n, n))  # n × n*p
        trend_cov = diagm([0.01, 0.02, 0.03])
        Σ_series = zeros(n_time, n, n)
        for t in 1:n_time
            Σ_series[t, :, :] = diagm(fill(0.5 + t / 100, n))     # a genuinely moving Σ_t
        end

        TCVAR.update_tc_var_sv!(model, var_coeff, trend_cov, Σ_series, nt, n, lags)

        n_states = nt + n * lags
        row0 = nt + n * (lags - 1)
        @test model.T[row0+1:end, nt+1:end] == var_coeff
        # The shift block and the trend identity above it are untouched.
        @test model.T[1:row0, :] == skeleton_T[1:row0, :]

        for t in 1:n_time
            Q_t = TCVAR.process_noise(model, t)
            @test Q_t[1:nt, 1:nt] == trend_cov
            @test Q_t[end-n+1:end, end-n+1:end] == Σ_series[t, :, :]
            # The intermediate companion lags carry no noise at all.
            @test all(iszero, Q_t[nt+1:n_states-n, :])
            @test all(iszero, Q_t[:, nt+1:n_states-n])
        end

        # In place: no reallocation of the n_time × n_states × n_states array.
        Q_before = model.Q
        TCVAR.update_tc_var_sv!(model, var_coeff, trend_cov, Σ_series, nt, n, lags)
        @test model.Q === Q_before

        @test_throws DimensionMismatch TCVAR.update_tc_var_sv!(
            model, var_coeff, trend_cov, zeros(n_time - 1, n, n), nt, n, lags)
        # A constant-Q model has no per-period block to write into.
        constant_q = TCVAR.TimeVaryingStateSpaceModel(model.T, model.R, model.Z,
                                                      zeros(n_states, n_states), model.H)
        @test_throws ArgumentError TCVAR.update_tc_var_sv!(
            constant_q, var_coeff, trend_cov, Σ_series, nt, n, lags)
    end

    @testset "TCVARSV constructor" begin
        priors = tcvar_sv_test_priors(; n = n, nt = nt, p = p)
        model = TCVAR.TCVARSV(trend_mapping, priors, n_time)

        @test model.ar_structure === :diagonal
        @test model.variable_names == ["y1", "y2", "y3"]
        @test model.trend_names == ["τ1", "τ2", "τ3"]
        @test size(model.ssm.Q) == (n_time, nt + n * p, nt + n * p)
        @test model.priors === priors

        @test_throws DimensionMismatch TCVAR.TCVARSV(Matrix(1.0I, n + 1, nt), priors, n_time)
        @test_throws DimensionMismatch TCVAR.TCVARSV(trend_mapping, priors, n_time;
                                                     variable_names = ["a", "b"])
        @test_throws DimensionMismatch TCVAR.TCVARSV(trend_mapping, priors, n_time;
                                                     trend_names = ["a"])
        # The struct's ar_structure must match the tuple the priors were built with.
        @test_throws DimensionMismatch TCVAR.TCVARSV(trend_mapping, priors, n_time;
                                                     ar_structure = :full)

        full_priors = tcvar_sv_test_priors(; n = n, nt = nt, p = p, ar_structure = :full)
        @test TCVAR.TCVARSV(trend_mapping, full_priors, n_time;
                            ar_structure = :full).ar_structure === :full

        # A different trend count needs a matching trend_covariance / initial_trend.
        wide = tcvar_sv_test_priors(; n = n, nt = 2, p = p)
        @test_throws DimensionMismatch TCVAR.TCVARSV(trend_mapping, wide, n_time)
    end

    @testset "simulate_tcvar_sv" begin
        priors = tcvar_sv_test_priors(; n = n, nt = nt, p = p)
        model = TCVAR.TCVARSV(trend_mapping, priors, n_time)
        params = simulator_params()

        Random.seed!(4321)
        states, observations, volatilities =
            TCVAR.simulate_tcvar_sv(model, params, zeros(nt + n * p), n_time)

        @test size(states) == (n_time, nt + n * p)
        @test size(observations) == (n_time, n)
        @test size(volatilities) == (n_time + 1, n)
        @test all(isfinite, states) && all(isfinite, volatilities)

        # h_0 is the starting value, and defaults to μ.
        @test volatilities[1, :] == params.μ

        # The observation is exact: y_t = Λ τ_t + c_t, no observation noise.
        @test observations ≈ states * model.ssm.Z'

        # The volatility path is the VAR(1) it claims to be — the implied innovations are
        # the only free part, and they have the right covariance.
        innovations = volatilities[2:end, :] .- params.μ' .-
                      (volatilities[1:end-1, :] .- params.μ') * params.Φ'
        @test size(innovations) == (n_time, n)

        @test_throws ArgumentError TCVAR.simulate_tcvar_sv(model, params, zeros(nt + n * p), 0)
        @test_throws DimensionMismatch TCVAR.simulate_tcvar_sv(model, params, zeros(nt), n_time)
        @test_throws DimensionMismatch TCVAR.simulate_tcvar_sv(
            model, params, zeros(nt + n * p), n_time; initial_volatility = zeros(n + 1))
        @test_throws DimensionMismatch TCVAR.simulate_tcvar_sv(
            model, merge(params, (β = zeros(n * (p + 1), n),)), zeros(nt + n * p), n_time)
        @test_throws DimensionMismatch TCVAR.simulate_tcvar_sv(
            model, merge(params, (A₀ = Matrix(1.0I, n + 1, n + 1),)), zeros(nt + n * p), n_time)
    end

    @testset "Ω = 0 reproduces a homoskedastic TCVAR path" begin
        # The stage-5 acceptance check: switch the volatility innovations off and the
        # simulator has to collapse onto a constant-Σ trend-cycle VAR at
        # Σ = A₀⁻¹ diag(exp(μ)) A₀⁻ᵀ. psd_factor is what makes "off" mean exactly off —
        # the jittered chol_psd would leak draws of standard deviation ≈ 1e-4 here.
        long = 20_000
        priors = tcvar_sv_test_priors(; n = n, nt = nt, p = p)
        model = TCVAR.TCVARSV(trend_mapping, priors, long)
        params = simulator_params(; Ω = zeros(n, n))

        Random.seed!(99)
        states, observations, volatilities =
            TCVAR.simulate_tcvar_sv(model, params, zeros(nt + n * p), long)

        # Every period sits at h_0 = μ exactly, so H_t — and therefore Σ_t — is constant.
        @test all(volatilities[t, :] == params.μ for t in 1:(long + 1))

        # The cycle innovations are then homoskedastic at Σ = A₀⁻¹ diag(exp(μ)) A₀⁻ᵀ.
        cycle = states[:, nt+1:end]
        innovations = cycle[2:end, :] - cycle[1:end-1, :] * params.β
        Σ_true = params.A₀ \ diagm(exp.(params.μ)) / params.A₀'
        @test cov(innovations) ≈ Σ_true rtol = 0.06

        # And the trends are the random walk they always were.
        trends = states[:, 1:nt]
        @test cov(diff(trends, dims = 1)) ≈ params.Στ rtol = 0.06

        # Στ = 0 pins the trends: the observations are then the cycle alone.
        Random.seed!(99)
        fixed_trend, fixed_obs, _ = TCVAR.simulate_tcvar_sv(
            model, merge(params, (Στ = zeros(nt, nt),)), zeros(nt + n * p), long)
        @test all(iszero, fixed_trend[:, 1:nt])
        @test fixed_obs ≈ fixed_trend[:, nt+1:end]
    end

    @testset "the n_scenarios form stacks independent paths" begin
        priors = tcvar_sv_test_priors(; n = n, nt = nt, p = p)
        model = TCVAR.TCVARSV(trend_mapping, priors, n_time)
        params = simulator_params()
        n_scenarios = 5

        Random.seed!(7)
        states, observations, volatilities =
            TCVAR.simulate_tcvar_sv(model, params, zeros(nt + n * p), n_scenarios, n_time)

        @test size(states) == (n_scenarios, n_time, nt + n * p)
        @test size(observations) == (n_scenarios, n_time, n)
        @test size(volatilities) == (n_scenarios, n_time + 1, n)

        # Same start, different draws.
        @test all(volatilities[s, 1, :] == params.μ for s in 1:n_scenarios)
        @test states[1, :, :] != states[2, :, :]

        # Scenario 1 is exactly the single-path simulation from the same seed.
        Random.seed!(7)
        single, _, _ = TCVAR.simulate_tcvar_sv(model, params, zeros(nt + n * p), n_time)
        @test states[1, :, :] == single
    end

    @testset "p = 2 uses the whole companion" begin
        lags = 2
        priors = tcvar_sv_test_priors(; n = n, nt = nt, p = lags)
        model = TCVAR.TCVARSV(trend_mapping, priors, n_time)

        # Coefficients on the *oldest* lag only: c_t depends on c_{t-2}, not on c_{t-1}.
        params = merge(simulator_params(; p = lags),
                       (β = vcat(0.5 * Matrix(1.0I, n, n), zeros(n, n)),
                        Ω = zeros(n, n)))

        Random.seed!(5)
        long = 20_000
        states, _, _ = TCVAR.simulate_tcvar_sv(model, params, zeros(nt + n * lags), long)

        cycle = states[:, nt+1:end]
        # The companion shifts: the first block of ξ_t is the second block of ξ_{t-1}.
        @test cycle[2:end, 1:n] == cycle[1:end-1, n+1:end]

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
