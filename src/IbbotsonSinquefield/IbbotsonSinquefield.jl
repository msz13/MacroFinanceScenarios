module IbbotsonSinquefield

export load_shiller_annual, subperiod
export ColumnTransformer, fit_columns, transform_columns, inverse_transform_columns
export Except, Affine, Invertible, Demean, Standardize
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
include(joinpath(@__DIR__, "transform.jl"))

end
