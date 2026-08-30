using Distributions, LinearAlgebra, Test

n = 2
beta_prior = MvNormal(rand(n), diagm(rand(n)))
sigma_prior = InverseWishart(rand()*100+1, diagm(rand(n)))

h = rand(n)
data = rand(n)
X = rand(n,n)
Σ = diagm(rand(n))

beta1 = rand(n)
beta2 = rand(n)

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

function normal_inverse_wishart_joint_prob(beta_prior, sigma_prior,random_y,random_x, random_Sigma, beta)
    return logpdf(beta_prior, beta) + logpdf(sigma_prior, random_Sigma) + logpdf(MvNormal(random_x*beta, random_Sigma), random_y)

end

beta1_prob = logpdf(NormalBetaPosterior(data, X, beta_prior, Σ), beta1)
beta2_prob = logpdf(NormalBetaPosterior(data, X, beta_prior, Σ), beta2)

joint_prob1 = normal_inverse_wishart_joint_prob(beta_prior, sigma_prior,data,X, Σ, beta1)

joint_prob2 = normal_inverse_wishart_joint_prob(beta_prior, sigma_prior,data,X, Σ, beta2)


@test isapprox(beta1_prob .- beta2_prob, joint_prob1 .- joint_prob2, atol=1e-5)
