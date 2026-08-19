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


# Includes are order-sensitive: a struct must be defined before the files whose
# signatures mention it. Layers go common → var → models → reporting.

# common/ — model-agnostic state space, filtering/smoothing and posteriors
include("common/linalg.jl")          # chol_psd, sample_mvn, lyapunov_covariance
include("common/state_space.jl")     # defines the AbstractStateSpaceModel hierarchy, used by kalman_filter.jl and tcvar_model.jl
include("common/kalman_filter.jl")
include("common/carter_kohn.jl")
include("common/posteriors.jl")

# common/sv/ — the stochastic-volatility block, model-agnostic: it takes orthogonalised
# residuals and the volatility parameters, and knows nothing about what produced them.
include("common/sv/ksc_mixture.jl")   # defines KSC_MIXTURE, needed by sv_block.jl
include("common/sv/sv_block.jl")
include("common/sv/sv_priors.jl")

# var/ — VAR algebra and the Minnesota prior
include("var/var_data.jl")
include("var/companion.jl")
include("var/minnesota_prior.jl")    # defines MinnesotaPrior, needed by var_sampling.jl and tcvar_priors.jl
include("var/var_sampling.jl")

# models/tcvar/ — the trend-cycle VAR itself
include("models/tcvar/tcvar_model.jl")   # defines TCVAR, needed by tcvar_result.jl and tcvar_gibbs.jl
include("models/tcvar/tcvar_priors.jl")
include("models/tcvar/tcvar_states.jl")
include("models/tcvar/tcvar_result.jl")  # defines TCVarResult, needed by plots.jl
include("models/tcvar/tcvar_gibbs.jl")

# reporting/ — summaries, plots and scenario statistics
include("reporting/posterior_summaries.jl")
include("reporting/plots.jl")
include("reporting/scenario_stats.jl")

export tc_var, sample
export prepare_var_data #TODO remove and use in sample function
export plot_variable_states, plot_states
export gibbs_sampler, MinnesotaPrior, TCVarResult
export var_priors, initial_cycle_prior, prior_var_coeff, prior_row_covariance
export posterior_mean, simulate_scenarios
export carter_kohn_sampler
# common/sv/
export sv_priors, draw_stochastic_volatility, draw_mixture_indicators, draw_log_volatilities
export compute_posterior_statistics
# reporting/scenario_stats.jl
export max_drawdown_and_length, returns_summarystats, cor_returns, annualise
export print_percentiles, sum_returns_between_periods, cum_returns_in_periods
export print_scenarios_summary, print_scenarios_percentiles, girf
export calculate_equity_returns, calculate_bond_returns

#re-export from FlexiChains package
export summarystats

end


