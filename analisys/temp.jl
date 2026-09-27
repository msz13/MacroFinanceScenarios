using StatsBase
using LinearAlgebra

# Named `sds` / `correlation` / `covariance`, not `std` / `cor` / `cov`: a scratch file is
# usually `include`d into a live REPL, and a Main binding of one of those names shadows the
# Statistics function of the same name for everything else in that session.
sds = [2, .5, 1.5]
sds2 = [2, 1.5, 1.5] 

correlation = [1 .3 .6
               .3 1 .4
               .6 .4 1]

covariance = cor2cov(correlation, sds)


L = cholesky(covariance).L

d = diag(L)

A = L / diagm(d)

D = diagm(d .^ 2)

A * D * A'

D2 = diagm(sds2 .^2)

covariance2 = A * D2 * A'

corr2 = cov2cor(covariance2)

using Distributions, LinearAlgebra

dist = [MvNormal(zeros(4), I(4)), MvNormal(zeros(4), I(4))]

p_d = product_distribution(dist)

B = rand(p_d)

logpdf(p_d, zeros(4,2))

X = randn(50,4)

X * B

dist = [MvNormal(zeros(4),I(4)), InverseWishart(5, diagm(ones(4)))]

normal_inverse_wishart = product_distribution(dist)

