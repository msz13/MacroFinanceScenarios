module TCVAR

using Plots
using Distributions
using FlexiChains
using FlexiChains: FlexiChain, Parameter
#using StatsPlots
using LinearAlgebra

using StatsBase
using PrettyTables
using TimeSeries


include("cycle_prior.jl") # defines MinnesotaPrior / var_priors, needed by gibbs_var_steps.jl and the TCVAR constructor
include("gibbs_var_steps.jl")
include("state_space.jl") # defines StateSpaceModel, used by TCVAR_model.jl
include("TCVAR_model.jl")
include("carter_kohn_algorythm.jl")
include("tcvar_result.jl")
include("gibbs_sampler.jl")
include("model_visualisation.jl")
include("utils.jl")

export tc_var, sample
export prepare_var_data #TODO remove and use in sample function
export plot_variable_states, plot_states
export gibbs_sampler, MinnesotaPrior, TCVarResult
export var_priors, initial_cycle_prior
export posterior_mean, simulate_scenarios
export carter_kohn_sampler
export compute_posterior_statistics
# utils.jl
export max_drawdown_and_length, returns_summarystats, cor_returns, annualise
export print_percentiles, sum_returns_between_periods, cum_returns_in_periods
export print_scenarios_summary, print_scenarios_percentiles, girf
export calculate_equity_returns, calculate_bond_returns

#re-export from FlexiChains package
export summarystats

end


