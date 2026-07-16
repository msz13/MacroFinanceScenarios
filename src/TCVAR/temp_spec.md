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
