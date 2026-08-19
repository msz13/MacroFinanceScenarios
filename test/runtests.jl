using MacroFinanceScenarios
using Test

# MacroFinanceScenarios does not include the TCVAR module, so load it here once — the
# included test files reuse it (they carry their own guard so each can also be run
# standalone, e.g. `julia --project test/tcvar_priors_test.jl`).
include(joinpath(@__DIR__, "..", "src", "TCVAR", "TCVAR.jl"))
using .TCVAR

@testset "MacroFinanceScenarios.jl" begin
    # Write your tests here.
end

include(joinpath(@__DIR__, "tcvar_test_utils.jl"))
include(joinpath(@__DIR__, "common", "linalg_test.jl"))
include(joinpath(@__DIR__, "common", "state_space_test.jl"))
include(joinpath(@__DIR__, "common", "sv", "ksc_mixture_test.jl"))
include(joinpath(@__DIR__, "common", "sv", "sv_block_test.jl"))
include(joinpath(@__DIR__, "common", "sv", "sv_priors_test.jl"))
include(joinpath(@__DIR__, "tcvar_posteriors_test.jl"))
include(joinpath(@__DIR__, "tcvar_priors_test.jl"))
include(joinpath(@__DIR__, "models", "tcvar", "tcvar_gibbs_regression_test.jl"))
include(joinpath(@__DIR__, "tcvar_recovery_test.jl"))
