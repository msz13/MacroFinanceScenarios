###TCVAR bvar posteriors refactor

Refactor var_sampling(../src/var/var_sampling.jl) to draw from natural conjugate posterior of coefficients and covariance implemented as custom distributions.jl distribution.
- implement in separate file in var folder NaturalConjugate struct, and rand and logpdf functions which draws and estiamte pdf for both beta and sigma, 
- implement test in separate file to check shapes of draws
- implement test like this BUT create function for var model joint probability:
```julia
T=10

n = 3
p = 2

betaPrior = Normal(rand(n*(n*p)), rand(diagm(rand(n*(n*p))))
sigmaPrior = sigmaPrior
Y = rand(T,n)
X = rand(T,n*p)
posterior = ConjugateNormal(Y,X, betaPrior, sigmaPrior)

beta1, sigma1 = rand(posterior)
beta2, sigma2 = rand(posterior)

params_prob1 = logpdf(posterior, beta1, sigma1)
params_prob2 = logpdf(posterior, beta2, sigma1)

join_llik1 = logpdf(betaPrior, beta1) + logpdf(sigmaPrior, sigma1) + logpdf(var_llik(beta1, sigma1, X), Y)
join_llik2 = logpdf(betaPrior, beta2) + logpdf(sigmaPrior, sigma1) + logpdf(var_llik(beta2, sigma1, X), Y)

params_prob1 = logpdf(posterior, beta1, sigma1)
params_prob2 = logpdf(posterior, beta1, sigma2)

join_llik1 = logpdf(betaPrior, beta1) + logpdf(sigmaPrior, sigma1) + logpdf(var_llik(beta1, sigma1, X), Y)
join_llik2 = logpdf(betaPrior, beta1) + logpdf(sigmaPrior, sigma2) + logpdf(var_llik(beta1, sigma2, X), Y)

@test isapprox(params_prob1 - params_prob2, join_llik1 -join_llik2, atol=1e-5)) 
```

####Code review
1. Change var_sampling, should draw both beta and sigma first,  L = coefficient_factor(posterior, Σ) should be inside rand natutalconjugate.
2. move var_coeff(β) as separate function, and then refactor tcvar gibs sampler and update_tc_var function which should get in params betas as vector and use var_coeff function inside
3. natural conjugate should take β_prior_μ, Ω_inv priors as distribution, not separate 
4. size(Y, 1) extract and assign variable T = size(Y, 1)
5. przejrzec minnesota prior if coeff precision should be estimated in constructor.
tests:
3. "posterior hyperparameters" what is for
4. change assertations from length to size 
