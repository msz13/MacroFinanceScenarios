using Test
using Distributions
using LinearAlgebra
using Random

isdefined(Main, :TCVAR) || include(joinpath(@__DIR__, "..", "..", "src", "TCVAR", "TCVAR.jl"))

# TCVAR members are reached as `TCVAR.f` rather than via `using .TCVAR` — see the note in
# tcvar_test_utils.jl.
isdefined(TCVAR, :TimeVaryingStateSpaceModel) || error(
    "The TCVAR module loaded in this session predates the time-varying state-space seam. " *
    "Restart the Julia session (or REPL / IDE worker) and re-run.")

# The `common/state_space.jl` seam: `kalman_filter` / `carter_kohn_sampler` reach the two
# noise covariances only through `process_noise` / `observation_noise`, so a model whose
# covariances move over time is a second implementation of the same interface rather than
# a second filter. The tests below pin (a) the accessor dispatch, (b) that a
# `TimeVaryingStateSpaceModel` fed constant covariances reproduces `StateSpaceModel`
# bit-for-bit, and (c) that a genuinely moving covariance does change the filter in the
# direction the model says it should.

# Scalar local level: x_t = x_{t-1} + u_t, y_t = x_t + v_t. Small enough that the
# filter's response to a moved Q_t or H_t can be read off by hand.
local_level(q, h) = TCVAR.StateSpaceModel(fill(1.0, 1, 1), fill(1.0, 1, 1), fill(1.0, 1, 1),
                                          fill(q, 1, 1), fill(h, 1, 1))

# Two-state model with a non-trivial R, so R*Q*R' is not just Q.
function two_state_model()
    T = [0.7 0.2; -0.1 0.5]
    R = [1.0 0.0; 0.5 1.0]
    Z = [1.0 0.0; 0.0 1.0]
    Q = [0.4 0.1; 0.1 0.3]
    H = [0.2 0.0; 0.0 0.5]
    return TCVAR.StateSpaceModel(T, R, Z, Q, H)
end

"""Repeat the constant covariance `C` into the `n_time × n × n` layout the time-varying
model indexes by period."""
function repeat_over_time(C, n_time)
    n = size(C, 1)
    stacked = zeros(n_time, n, n)
    for t in 1:n_time
        stacked[t, :, :] = C
    end
    return stacked
end

@testset "common/state_space" begin

    @testset "noise accessors" begin
        constant = local_level(0.3, 0.7)

        # A constant model returns the same matrices at every period, and the accessor
        # hands back the field itself rather than a copy.
        @test TCVAR.process_noise(constant, 1) === constant.Q
        @test TCVAR.process_noise(constant, 17) === constant.Q
        @test TCVAR.observation_noise(constant, 1) === constant.H
        @test TCVAR.observation_noise(constant, 17) === constant.H

        Q_series = reshape([1.0, 2.0, 3.0], 3, 1, 1)
        H_series = reshape([10.0, 20.0, 30.0], 3, 1, 1)
        varying = TCVAR.TimeVaryingStateSpaceModel(fill(1.0, 1, 1), fill(1.0, 1, 1),
                                                   fill(1.0, 1, 1), Q_series, H_series)

        @test [TCVAR.process_noise(varying, t)[1] for t in 1:3] == [1.0, 2.0, 3.0]
        @test [TCVAR.observation_noise(varying, t)[1] for t in 1:3] == [10.0, 20.0, 30.0]

        # Each covariance is independently constant or time varying: the SV block needs a
        # constant Q with a moving H, the TCVAR-SV state draw the other way round.
        mixed = TCVAR.TimeVaryingStateSpaceModel(fill(1.0, 1, 1), fill(1.0, 1, 1),
                                                 fill(1.0, 1, 1), fill(0.5, 1, 1), H_series)
        @test TCVAR.process_noise(mixed, 1) === mixed.Q
        @test TCVAR.process_noise(mixed, 3) === mixed.Q
        @test TCVAR.observation_noise(mixed, 3)[1] == 30.0
    end

    @testset "constant_state_noise hoist" begin
        model = two_state_model()

        @test TCVAR.constant_state_noise(model) == model.R * model.Q * model.R'

        # A time-varying model with a constant Q keeps the hoist ...
        constant_Q = TCVAR.TimeVaryingStateSpaceModel(model.T, model.R, model.Z, model.Q,
                                                      repeat_over_time(model.H, 4))
        @test TCVAR.constant_state_noise(constant_Q) == model.R * model.Q * model.R'

        # ... and only a genuinely moving Q forces the per-period product.
        moving_Q = TCVAR.TimeVaryingStateSpaceModel(model.T, model.R, model.Z,
                                                    repeat_over_time(model.Q, 4), model.H)
        @test TCVAR.constant_state_noise(moving_Q) === nothing
    end

    @testset "constant covariances reproduce StateSpaceModel bit-for-bit" begin
        model = two_state_model()
        n_time = 12

        Random.seed!(4242)
        observations = Matrix{Union{Missing,Float64}}(randn(n_time, 2))
        observations[5, 2] = missing        # exercise the missing-observation branch
        observations[9, :] .= missing       # ... and the no-observation branch

        μ₀ = [0.3, -0.2]
        P₀ = [1.5 0.2; 0.2 0.9]

        equivalent = TCVAR.TimeVaryingStateSpaceModel(model.T, model.R, model.Z,
                                                      repeat_over_time(model.Q, n_time),
                                                      repeat_over_time(model.H, n_time))

        constant_out = TCVAR.kalman_filter(model, observations, μ₀, P₀)
        varying_out  = TCVAR.kalman_filter(equivalent, observations, μ₀, P₀)

        # `==` on Float64 arrays is bit equality: the two paths must agree exactly, not
        # just to a tolerance, because TCVAR's draws go through the constant one.
        for (constant_array, varying_array) in zip(constant_out[1:4], varying_out[1:4])
            @test constant_array == varying_array
        end

        # Same for the backward sampling pass, under a shared seed.
        Random.seed!(99)
        constant_draw = TCVAR.carter_kohn_sampler(model, observations, μ₀, P₀)
        Random.seed!(99)
        varying_draw = TCVAR.carter_kohn_sampler(equivalent, observations, μ₀, P₀)

        @test constant_draw[1] == varying_draw[1]
        @test constant_draw[2] == varying_draw[2]
    end

    @testset "time-varying covariances move the filter" begin
        μ₀, P₀ = [0.0], fill(1.0, 1, 1)
        observations = reshape([1.0, 1.0, 1.0], 3, 1)

        # An observation whose noise variance is enormous carries no information, so the
        # filtered state falls back on the prediction; a tiny one pins the state on y_t.
        H_series = reshape([1.0, 1e8, 1e-8], 3, 1, 1)
        varying = TCVAR.TimeVaryingStateSpaceModel(fill(1.0, 1, 1), fill(1.0, 1, 1),
                                                   fill(1.0, 1, 1), fill(0.1, 1, 1), H_series)

        filtered, _, predicted, _, _ = TCVAR.kalman_filter(varying, observations, μ₀, P₀)

        @test filtered[2, 1] ≈ predicted[2, 1] atol = 1e-6
        @test filtered[3, 1] ≈ 1.0 atol = 1e-6

        # A zero state-noise period leaves the predicted covariance at T·P·T' alone.
        Q_series = reshape([0.5, 0.0, 0.5], 3, 1, 1)
        varying_Q = TCVAR.TimeVaryingStateSpaceModel(fill(1.0, 1, 1), fill(1.0, 1, 1),
                                                     fill(1.0, 1, 1), Q_series, fill(1.0, 1, 1))

        state_filtered, covariance_filtered, _, covariance_predicted, _ =
            TCVAR.kalman_filter(varying_Q, observations, μ₀, P₀)

        @test covariance_predicted[1, 1, 1] ≈ P₀[1, 1] + 0.5
        @test covariance_predicted[2, 1, 1] ≈ covariance_filtered[1, 1, 1]
    end

    @testset "relaxed argument types" begin
        # The SV block hands the filter a plain Matrix{Float64} of log-volatility
        # observations and a non-Vector{Float64} initial mean; both must be accepted
        # without a conversion at the call site.
        model = local_level(0.2, 0.4)
        observations = randn(6, 1)

        filtered, _, _, _, _ = TCVAR.kalman_filter(model, observations, [0.0], fill(1.0, 1, 1))
        @test size(filtered) == (6, 1)

        @test TCVAR.kalman_filter(model, observations, 0:0, Diagonal([1.0]))[1] ==
              TCVAR.kalman_filter(model, observations, [0.0], fill(1.0, 1, 1))[1]
    end

end
