

analise result class how i sholuld keep estimated parameters. There are to options: 
- named tuple for coefficientnt cycle and trend covariances with raw arrays as values
- FlexiChains
-  named tuple for coefficientnt cycle and trend covariances with FlexiChains as values
I need parameters, to:

1. create reports with content
- plots of parameters for convergence check
- display summary stats (mean, median, confidence bands) of parameters as tables, split by kind of parameters(coeffcients, covariances) and variables (like variable 1 coefficents, variable 2, etc)
- plot each estimated states
2. simulate feature result with:
- or all of them


TCVAR-SV
- refactor file structure
- add metropolis hastings step:
  - create separate function
  - use mapping function
- create new gibs sampler - extension of TCVAR with stochastic volatility
  - 
  create function for draw multivariate stochastic volatility block with chib approximation
  - stochastic valatility is modeled as autoregresive model with mean and corellated innovations
  - function should take as input resuduals and parameters - mean, matrix of ar coefficients, covariance matrix of volatility 
  - function should call two sub functions: sample mixture indicator s and sample volatilities h
  - volatilities should be sampled from carter kohn algorythm
  - add script to test 

  extend tcvar gibs sampler to add stochastic volatility with constand corellations:
  - create new sampler but use already created function for same steps in two models
  - gibs sampler should have followint steps 
    - draw tcvar states 
    - draw trend covariance matrix (estimate posterios with gls)
    - draw var coeficients
    - draw var corellations parameters L
    - draw var volatilieties
    - draw volatility mean
    - draw volatility coefficients
    - draw volatility covariance matrix
Problems:
- correlation posterior and priors




- Structure
- TCVAR model
- TCVAR-SV
- BVAR_steps
- Steady state
- common

- for initial cycle covariance refactor to uncoditional covariance 


refactor file structure 
- i want to create new model to extend @src/tcvar and create tcvar with stochastiv volatility. plan refactor file structure to separate common function  and specific functions for tc var to prepare to adding new functionalities. var functions should be separate. write plan to md file.

Co ja robie 
- specyfikacja TCVAR-SV

TODO:
- funckja cycle_priors
- restrukturyzacja plików
- zrobienie sampling SV
- zrobienie TCVAR-SV


Σ_prior, β_prior, c0_prior = var_priors(.2, 2, [2., 1., .1, 25., 1]; δ = [0,0,0,0,0] )

priors = (
    initial_trend = MvNormal([2., .5, 1., 2., 5.], diagm(ones(5))),
    initial_cycle = c0_prior,
    trend_covariance = InverseWishart(100, diagm([2., 1., 1., 2., 1.].^2 ./ 400)),
    cycle_covariance = Σ_prior,
    cycle_β = β_prior
)

refactor file structure:
- from gibs_sampler covariance_posterior to separate file
- from gibs sampler simulate scenarios and mean posterior to tcvar_result
- gibs_var_steps rename to BVAR.jl
- gibis var steps prepare_var_data seprate file
- gibis var steps covariance_posterior_separate file
- separate folder for var 
- zastanowić się nad konwencją nazywania funckji: albo gibs sampler step - draw covariance, draw beta coefficients, albo posterior distrubutions: inverse wishart_posteror, minnesota_inverse wishar


TODO:
initial_cycle - separate function uncondutional variance in var