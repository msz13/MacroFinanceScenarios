using Distributions, LinearAlgebra, Test


isdefined(Main, :TCVAR) || include(joinpath(@__DIR__, "..", "src", "TCVAR", "TCVAR.jl"))

# TCVAR members are reached as `TCVAR.f` rather than via `using .TCVAR` — see the note in
# tcvar_test_utils.jl.
isdefined(TCVAR, :inverse_wishart_posterior) || error(
    "The TCVAR module loaded in this session predates inverse_wishart_posterior. " *
    "Restart the Julia session (or REPL / IDE worker) and re-run.")

using .TCVAR


# Conjugate posterior for β in  Y = Xβ + ε,  ε ~ N(0, Σ),  β ~ N(prior.μ, prior.Σ).
# MvNormalCanon takes the *canonical* parameters: precision J and potential η = J * mean,
# so both Σ's have to enter as precisions.
function NormalBetaPosterior(Y, X, prior::MvNormal, Σ)
    Λ₀ = inv(prior.Σ)                    # prior precision
    Σ_inv = inv(Σ)                       # observation precision
    J = Symmetric(Λ₀ + X' * Σ_inv * X)   # posterior precision
    η = Λ₀ * prior.μ + X' * Σ_inv * Y     # posterior potential
    return MvNormalCanon(η, J)
end

function normal_inverse_wishart_joint_prob(beta_prior, sigma_prior, y, x, Σ, beta)
    return logpdf(beta_prior, beta) + logpdf(sigma_prior, Σ) + logpdf(MvNormal(vec(x * beta), Σ), y)
end

# Joint of Σ ~ IW(prior) and T residual rows ε_t ~ N(0, Σ), as a function of Σ.
# `residuals` is T × n, matching the layout `inverse_wishart_posterior` expects.
function inverse_wishart_joint_prob(sigma_prior, residuals, Σ)
    n = size(residuals, 2)
    return logpdf(sigma_prior, Σ) + sum(logpdf(MvNormal(zeros(n), Σ), permutedims(residuals)))
end

@testset "conjugate posteriors" begin

    # Both blocks check the same thing: a conjugate posterior equals the joint
    # (prior × likelihood) only up to a normalising constant, so the *difference* of the
    # log densities at two parameter values has to agree exactly.

    @testset "normal beta posterior" begin
        n = 2
        beta_prior = MvNormal(rand(n), diagm(rand(n)))
        sigma_prior = InverseWishart(rand()*100+1, diagm(rand(n)))

        data = rand(n)
        X = rand(n,n)
        Σ = diagm(rand(n))

        beta1 = rand(n)
        beta2 = rand(n)

        beta1_prob = logpdf(NormalBetaPosterior(data, X, beta_prior, Σ), beta1)
        beta2_prob = logpdf(NormalBetaPosterior(data, X, beta_prior, Σ), beta2)

        joint_prob1 = normal_inverse_wishart_joint_prob(beta_prior, sigma_prior,data,X, Σ, beta1)

        joint_prob2 = normal_inverse_wishart_joint_prob(beta_prior, sigma_prior,data,X, Σ, beta2)

        @test isapprox(beta1_prob .- beta2_prob, joint_prob1 .- joint_prob2, atol=1e-5)
    end

    @testset "inverse_wishart_posterior" begin
        n = 2
        p = 1
        T = 20

        # df has to stay well clear of the n - 1 existence bound, otherwise `rand` draws
        # near-singular Σ's whose log densities are too large for the tolerance below.
        df_prior = rand()*100 + n + 2
        scale_prior = diagm(rand(n))
        sigma_prior = InverseWishart(df_prior, scale_prior)

        Y, X = prepare_var_data(rand(T, n), p)

        beta = rand(n * p, n)
        residuals = Y - X * beta

        # the caller owns the degrees-of-freedom update: one row per *usable* observation
        # is added to the prior, and the p lags consumed by `prepare_var_data` are gone.
        df_posterior = df_prior + size(residuals, 1)

        posterior = TCVAR.inverse_wishart_posterior(residuals, scale_prior, df_posterior)

        Σ1 = rand(sigma_prior)
        Σ2 = rand(sigma_prior)

        sigma1_prob = logpdf(posterior, Σ1)
        sigma2_prob = logpdf(posterior, Σ2)

        # Conditional on β the joint is prior × the likelihood of *every* residual row —
        # a single row would only identify a one-observation posterior.
        joint_prob1 = inverse_wishart_joint_prob(sigma_prior, residuals, Σ1)
        joint_prob2 = inverse_wishart_joint_prob(sigma_prior, residuals, Σ2)

        @test isapprox(sigma1_prob - sigma2_prob, joint_prob1 - joint_prob2, atol=1e-5)
    end
end
