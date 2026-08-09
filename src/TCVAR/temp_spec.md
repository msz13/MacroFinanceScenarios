

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
  - use chib approximate method for stochastic volatlility 
  - create seperate file SV
  - add separate functions for two stochastic volatility steps


- Structure
- TCVAR model
- TCVAR-SV
- BVAR_steps
- Steady state
- common

