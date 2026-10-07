module IbbotsonSinquefield

export load_shiller_annual, subperiod

using Dates
using Statistics
using TimeSeries
using XLSX

include(joinpath(@__DIR__, "..", "common", "bond_returns.jl"))
include(joinpath(@__DIR__, "data.jl"))

end
