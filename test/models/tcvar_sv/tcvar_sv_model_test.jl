using Test
using Distributions
using LinearAlgebra
using Random
using Statistics

isdefined(Main, :TCVAR) || include(joinpath(@__DIR__, "..", "..", "..", "src", "TCVAR", "TCVAR.jl"))
isdefined(Main, :tcvar_sv_test_priors) || include(joinpath(@__DIR__, "..", "..", "tcvar_test_utils.jl"))

# TCVAR members are reached as `TCVAR.f` rather than via `using .TCVAR` — see the note in
# tcvar_test_utils.jl.
isdefined(TCVAR, :TCVARSV) || error(
    "The TCVAR module loaded in this session predates the TCVAR-SV model. " *
    "Restart the Julia session (or REPL / IDE worker) and re-run.")

# Forward simulation is tested in tcvar_sv_result_test.jl, next to `simulate_scenarios`.

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

end
