Refactor gibs sampler:

- extract method which transformes sampled params and states to return result
- use FlexiChains.jl for storing parameters not MCCChains
- concat trend_covariance, betas, sigmas and convert array into one FlexiChain
- create result struct:
    TCVarResult:
    model #model with empty params
    params # FlexiChain
    trend_states
    cycle_states


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
- mean of parameters 
- or all of them