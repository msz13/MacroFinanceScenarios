using Test
using Distributions
using LinearAlgebra
using Random
using Statistics

isdefined(Main, :TCVAR) || include(joinpath(@__DIR__, "..", "src", "TCVAR", "TCVAR.jl"))

# TCVAR members are reached as `TCVAR.f` rather than via `using .TCVAR` — see the note in
# tcvar_test_utils.jl.
#
# A session that loaded TCVAR before the posteriors refactor keeps a module without the
# names below, and the guard above deliberately reuses it rather than re-including, so
# report that directly instead of failing later with an UndefVarError per function.
isdefined(TCVAR, :random_walk_covariance_posterior) || error(
    "The TCVAR module loaded in this session predates the common/posteriors.jl refactor. " *
    "Restart the Julia session (or REPL / IDE worker) and re-run — re-including " *
    "TCVAR.jl in place would replace the module and break name resolution elsewhere.")

# The `common/posteriors.jl` layer: the `*_posterior` functions are pure (no RNG), so
# they are checked against closed-form conjugate results; `draw_from_factor` is the only
# one that consumes RNG and is checked for reproducibility and second moments.
# The functions are internal to the module, hence the `TCVAR.` qualification.

@testset "common/posteriors" begin

    @testset "inverse_wishart_posterior" begin
        ε = [1.0 0.5; -0.5 2.0; 0.25 -1.0; 1.5 0.0]
        Ψ = [3.0 0.5; 0.5 2.0]
        df = 12.0

        d = TCVAR.inverse_wishart_posterior(ε, Ψ, df)
        df_post, Ψ_post = params(d)

        @test d isa InverseWishart
        @test df_post == df
        @test Matrix(Ψ_post) ≈ ε'ε + Ψ

        # two-argument form takes an already-assembled scale
        @test Matrix(params(TCVAR.inverse_wishart_posterior(ε'ε + Ψ, df))[2]) ==
              Matrix(Ψ_post)

        # the scale is symmetrised: only the upper triangle survives
        asym = [4.0 1.0; 1.0 + 1e-9 5.0]
        @test Matrix(params(TCVAR.inverse_wishart_posterior(asym, df))[2]) == [4.0 1.0; 1.0 5.0]
    end

    @testset "random_walk_covariance_posterior" begin
        states = [0.0 0.0; 0.5 -0.25; 0.75 0.5; 0.25 1.0]   # includes the pre-sample point
        Ψ = [0.1 0.0; 0.0 0.2]
        df = 9.0

        d = TCVAR.random_walk_covariance_posterior(states, Ψ, df)
        innovations = diff(states, dims = 1)

        @test params(d)[1] == df
        # a T-point path yields T-1 innovations
        @test size(innovations, 1) == size(states, 1) - 1
        @test Matrix(params(d)[2]) ≈ innovations' * innovations + Ψ
        # identical to the residual form applied to the differenced path
        @test Matrix(params(d)[2]) ==
              Matrix(params(TCVAR.inverse_wishart_posterior(innovations, Ψ, df))[2])
    end

    @testset "normal_coefficient_posterior_mean" begin
        Random.seed!(3)
        T, k, n = 60, 3, 2
        X = randn(T, k)
        Y = randn(T, n)
        β₀ = ones(k, n)

        # a vanishing prior precision collapses the posterior mean onto OLS
        @test TCVAR.normal_coefficient_posterior_mean(Y, X, β₀, 1e-10 * I(k)) ≈ X \ Y atol = 1e-6

        # an overwhelming prior precision pins it to the prior mean
        @test TCVAR.normal_coefficient_posterior_mean(Y, X, β₀, 1e12 * I(k)) ≈ β₀ atol = 1e-6

        # closed form in between
        Ω_inv = diagm([2.0, 1.0, 0.5])
        @test TCVAR.normal_coefficient_posterior_mean(Y, X, β₀, Ω_inv) ≈
              inv(X'X + Ω_inv) * (X'Y + Ω_inv * β₀)
    end

    @testset "kron_cholesky_factor" begin
        Σ = [1.0 0.2; 0.2 0.5]
        V = [2.0 0.3 0.0; 0.3 1.0 0.1; 0.0 0.1 0.75]
        L = TCVAR.kron_cholesky_factor(Σ, V)

        m = size(Σ, 1) * size(V, 1)
        @test size(L) == (m, m)
        @test istril(L)

        # chol(A ⊗ B) = chol(A) ⊗ chol(B): the factor reproduces the jittered Kronecker
        # covariance exactly, and the un-jittered one to within the jitter.
        jitter = 1e-5
        @test L * L' ≈ kron(Σ + jitter * I, V + jitter * I)
        @test L * L' ≈ kron(Σ, V) atol = 1e-4
    end

    @testset "draw_from_factor" begin
        Σ = [1.0 0.2; 0.2 0.5]
        V = [2.0 0.3; 0.3 1.0]
        L = TCVAR.kron_cholesky_factor(Σ, V)
        μ = reshape(collect(1.0:4.0), 2, 2)

        # a matrix mean is vectorised column-wise; a zero factor returns it unchanged
        @test TCVAR.draw_from_factor(μ, zeros(4, 4)) == vec(μ)

        Random.seed!(99); a = TCVAR.draw_from_factor(μ, L)
        Random.seed!(99); b = TCVAR.draw_from_factor(μ, L)
        @test a == b
        @test length(a) == 4

        # first two moments of the draw match N(vec(μ), L L')
        Random.seed!(5)
        n_draws = 100_000
        draws = Matrix{Float64}(undef, 4, n_draws)
        for j in 1:n_draws
            draws[:, j] = TCVAR.draw_from_factor(μ, L)
        end
        @test vec(mean(draws, dims = 2)) ≈ vec(μ) atol = 0.03
        @test cov(draws, dims = 2) ≈ L * L' atol = 0.06
    end

end
