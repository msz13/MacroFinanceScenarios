module IbbotsonSinquefield

export load_shiller_annual, subperiod
export AR1, fit_ar1, unconditional_mean, unconditional_std
export Scenarios, scenario_years, is_coefficients, issm_bootstrap_matrix
export simulate_var_bootstrapped_res, returns_from_components

using Dates
using LinearAlgebra: I, diag, eigvals, mul!
using Random
using Statistics
using TimeSeries
using XLSX

include(joinpath(@__DIR__, "..", "common", "bond_returns.jl"))
include(joinpath(@__DIR__, "data.jl"))
include(joinpath(@__DIR__, "ar1.jl"))
include(joinpath(@__DIR__, "simulate.jl"))

end
