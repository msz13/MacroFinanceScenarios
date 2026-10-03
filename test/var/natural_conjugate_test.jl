using Test
using Distributions
using LinearAlgebra
using Random

isdefined(Main, :TCVAR) || include(joinpath(@__DIR__, "..", "..", "src", "TCVAR", "TCVAR.jl"))

# TCVAR members are reached as `TCVAR.f` rather than via `using .TCVAR` — see the note in
# tcvar_test_utils.jl.
isdefined(TCVAR, :NaturalConjugate) || error(
    "The TCVAR module loaded in this session predates NaturalConjugate. " *
    "Restart the Julia session (or REPL / IDE worker) and re-run.")

# VAR likelihood of the stacked observations: vec(Y) ~ N(vec(X B), Σ ⊗ I_T).
function var_likelihood(β, Σ, X)
    T, k = size(X)
    B = reshape(β, k, :)
    return MvNormal(vec(X * B), collect(Hermitian(kron(Σ, Matrix(1.0I, T, T)))))
end

# Joint log density of the natural-conjugate VAR: β | Σ ~ N(vec(B₀), Σ ⊗ Ω), Σ ~ sigma_prior
# and the likelihood above. Differs from the posterior only by the log marginal likelihood.
function var_joint_logpdf(β, Σ, Y, X, β_prior_μ, Ω, sigma_prior)
    beta_prior = MvNormal(vec(β_prior_μ), collect(Hermitian(kron(Σ, Ω))))
    return logpdf(beta_prior, β) + logpdf(sigma_prior, Σ) + logpdf(var_likelihood(β, Σ, X), vec(Y))
end

@testset "var/natural_conjugate" begin
    T, n, p = 10, 3, 2
    k = n * p

    Random.seed!(20261003)
    Y = rand(T, n)
    X = rand(T, k)
    β_prior_μ = rand(k, n)
    Ω = diagm(rand(k) .+ 0.1)
    sigma_prior = InverseWishart(n + 2.0, Matrix(1.0I, n, n))

    posterior = TCVAR.NaturalConjugate(Y, X, β_prior_μ, inv(Ω), sigma_prior)

    @testset "shapes" begin
        β, Σ = rand(posterior)
        @test β isa Vector{Float64}
        @test length(β) == k * n == length(posterior)
        @test size(Σ) == (n, n)
        @test isposdef(Σ)

        @test length(TCVAR.rand_coefficients(posterior, Σ)) == k * n
        @test size(rand(TCVAR.covariance_posterior(posterior))) == (n, n)
        @test length(TCVAR.coefficient_posterior(posterior, Σ)) == k * n
        @test logpdf(posterior, β, Σ) isa Float64
        # β accepted as the k × n matrix too
        @test logpdf(posterior, reshape(β, k, n), Σ) == logpdf(posterior, β, Σ)
    end

    @testset "posterior hyperparameters" begin
        ν_post, S_post = params(TCVAR.covariance_posterior(posterior))
        @test ν_post == params(sigma_prior)[1] + T
        @test posterior.Ω ≈ inv(X'X + inv(Ω))
        @test posterior.β_μ ≈ (X'X + inv(Ω)) \ (X'Y + Ω \ β_prior_μ)
    end

    # The posterior equals prior × likelihood up to the normalising constant, so the
    # difference of log densities between two parameter values must agree exactly.
    @testset "matches the joint density up to a constant" begin
        joint(β, Σ) = var_joint_logpdf(β, Σ, Y, X, β_prior_μ, Ω, sigma_prior)

        β1, Σ1 = rand(posterior)
        β2, Σ2 = rand(posterior)

        # vary β, Σ fixed
        @test isapprox(logpdf(posterior, β1, Σ1) - logpdf(posterior, β2, Σ1),
                       joint(β1, Σ1) - joint(β2, Σ1), atol = 1e-5)

        # vary Σ, β fixed
        @test isapprox(logpdf(posterior, β1, Σ1) - logpdf(posterior, β1, Σ2),
                       joint(β1, Σ1) - joint(β1, Σ2), atol = 1e-5)

        # vary both
        @test isapprox(logpdf(posterior, β1, Σ1) - logpdf(posterior, β2, Σ2),
                       joint(β1, Σ1) - joint(β2, Σ2), atol = 1e-5)
    end

    @testset "rand is reproducible" begin
        a = rand(MersenneTwister(1), posterior)
        b = rand(MersenneTwister(1), posterior)
        @test a == b
    end
end
