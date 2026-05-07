module MacroFinanceScenarios

module TCFSimulation

    export run_monte_carlo, print_summary, ModelParams

    using Random, Distributions, LinearAlgebra, Statistics, PrettyTables
    include(joinpath(@__DIR__, "tcf_model", "HillenbrandMcCarthyModel.jl"))
    include(joinpath(@__DIR__, "tcf_model", "simulate.jl"))
end

module ScoreSimulation
   
end

export TCFSimulation, ScoreSimulation

end
