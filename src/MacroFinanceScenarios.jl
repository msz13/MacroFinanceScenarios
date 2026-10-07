module MacroFinanceScenarios

module TCFSimulation

    export run_monte_carlo, print_summary, ModelParams

    using Random, Distributions, LinearAlgebra, Statistics, PrettyTables
    include(joinpath(@__DIR__, "tcf_model", "HillenbrandMcCarthyModel.jl"))
    include(joinpath(@__DIR__, "tcf_model", "simulate.jl"))
end

module ScoreSimulation

end

module EDA

    export describe_series, correlation_table, print_table

    using DataFrames, PrettyTables, Statistics, StatsBase, TimeSeries
    include(joinpath(@__DIR__, "EDA", "eda.jl"))
end

include(joinpath(@__DIR__, "IbbotsonSinquefield", "IbbotsonSinquefield.jl"))

export TCFSimulation, ScoreSimulation, EDA, IbbotsonSinquefield

end
