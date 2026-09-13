using Test
using Distributions
using LinearAlgebra
using Random

isdefined(Main, :TCVAR) || include(joinpath(@__DIR__, "..", "..", "src", "TCVAR", "TCVAR.jl"))

# TCVAR members are reached as `TCVAR.f` rather than via `using .TCVAR` — see the note in
# tcvar_test_utils.jl.

@testset "var/var_sampling" begin

    @testset "normal_inverse_wishart_posterior" begin
        Random.seed!(11)
        T, n, p = 80, 2, 2
        k = n * p
        Y, X = TCVAR.prepare_var_data(randn(T, n), p)
        β₀ = zeros(k, n)
        Ω_inv = diagm(fill(2.0, k))
        S = [1.0 0.1; 0.1 0.5]
        df = size(Y, 1) + n + 2.0
        Σ = [0.8 0.2; 0.2 0.6]

        post = TCVAR.normal_inverse_wishart_posterior(Y, X, Σ, β₀, Ω_inv, S, df)
        β̂ = inv(X'X + Ω_inv) * (X'Y + Ω_inv * β₀)
        ε = Y - X * β̂

        # one block per parameter, shaped as the sampler stores them
        x = rand(post)
        @test keys(x) == (:β, :Σ)
        @test size(x.β) == (k * n,)
        @test size(x.Σ) == (n, n)

        # β | Σ, Y ~ N(vec(β̂), Σ ⊗ (X'X + Ω⁻¹)⁻¹), to within the jitter of its factor
        @test mean(post.dists.β) ≈ vec(β̂)
        @test cov(post.dists.β) ≈ kron(Σ, inv(X'X + Ω_inv)) atol = 1e-4

        # Σ | Y ~ IW(df, ε'ε + (β̂ − β₀)' Ω⁻¹ (β̂ − β₀) + S)
        df_post, scale_post = params(post.dists.Σ)
        @test df_post == df
        @test Matrix(scale_post) ≈ ε'ε + (β̂ - β₀)' * Ω_inv * (β̂ - β₀) + S

        # only the coefficient block is conditioned on Σ
        other = TCVAR.normal_inverse_wishart_posterior(Y, X, 2Σ, β₀, Ω_inv, S, df)
        @test cov(other.dists.β) ≈ 2 * cov(post.dists.β) atol = 1e-4
        @test Matrix(params(other.dists.Σ)[2]) == Matrix(scale_post)

        # at the Σ it was built with, the log density is the joint one: the sum of the blocks'
        b = 0.3 * randn(k * n)
        @test logpdf(post, (β = b, Σ = Σ)) ≈ logpdf(post.dists.β, b) + logpdf(post.dists.Σ, Σ)
    end

end
