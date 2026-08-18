using Test
using Distributions
using LinearAlgebra
using Random
using Statistics

isdefined(Main, :TCVAR) || include(joinpath(@__DIR__, "..", "..", "..", "src", "TCVAR", "TCVAR.jl"))
using FlexiChains: @varname

# TCVAR members are reached as `TCVAR.f` rather than via `using .TCVAR` — see the note in
# tcvar_test_utils.jl.
include(joinpath(@__DIR__, "..", "..", "tcvar_test_utils.jl"))

# Non-regression guard for the shared filtering layer.
#
# TCVAR-SV generalises `kalman_filter` / `carter_kohn_sampler` / `sample_states` to
# time-varying noise covariances, and those are the very functions TCVAR itself samples
# through. The generalisation is only safe if it leaves TCVAR's arithmetic — not merely
# its distribution — untouched, so this pins a seeded sweep to the exact draws produced
# before the `AbstractStateSpaceModel` seam existed. Any reordering or re-association of
# the floating-point operations on the constant-covariance path shows up here.
#
# The checksums fold the raw IEEE bit patterns of every draw (FNV-1a over
# `reinterpret(UInt64, ·)`), so they are exact and independent of how the numbers print.
# They do depend on the RNG stream: a Julia or Distributions upgrade that changes how
# `randn` / `rand(::InverseWishart)` consume randomness will move them without anything
# being wrong. Re-derive them in that case by checking out the last commit whose sampler
# is known good, re-running this file, and pasting the printed values — do not update
# them to whatever the current code happens to produce.

"""FNV-1a over the IEEE bit patterns of `values`; order sensitive and exact."""
function bit_checksum(values)
    h = UInt64(0xcbf29ce484222325)
    for x in values
        h ⊻= reinterpret(UInt64, Float64(x))
        h *= UInt64(0x100000001b3)
    end
    return h
end

flat_draws(chain_entry) = Iterators.flatten(vec(collect(chain_entry)))

@testset "TCVAR gibbs_sampler non-regression" begin
    n, nt, p, T = 2, 2, 1, 40

    Random.seed!(20240617)
    Στ_true = diagm([0.01, 0.02])
    Σc_true = [0.5 0.1; 0.1 0.3]
    B_true  = [0.6 0.1; -0.2 0.5]

    τ = zeros(T + 1, nt)
    c = zeros(T + 1, n)
    for t in 2:T+1
        τ[t, :] = τ[t-1, :] + rand(MvNormal(zeros(nt), Στ_true))
        c[t, :] = B_true * c[t-1, :] + rand(MvNormal(zeros(n), Σc_true))
    end
    data = Matrix{Union{Missing,Float64}}(τ[2:end, :] + c[2:end, :])

    model = TCVAR.TCVAR(Matrix(1.0I, n, n), tcvar_test_priors(; n = n, nt = nt, p = p))

    Random.seed!(987654321)
    result = TCVAR.gibbs_sampler(model, data; burnin = 5, n_samples = 10)

    checksums = (trend_states = bit_checksum(result.trend_states),
                 cycle_states = bit_checksum(result.cycle_states),
                 Στ = bit_checksum(flat_draws(result.params[@varname(Στ)])),
                 β  = bit_checksum(flat_draws(result.params[@varname(β)])),
                 Σc = bit_checksum(flat_draws(result.params[@varname(Σc)])))

    # Reference values, recorded on commit 195de40 (before the time-varying seam).
    reference = (trend_states = 0x5e3a5074c294a505,
                 cycle_states = 0x75ada2b9914e635b,
                 Στ = 0xb2d8086d3da5b9fa,
                 β  = 0x0cebd48ea4589feb,
                 Σc = 0x87eaafa69058dbde)

    for name in keys(reference)
        if checksums[name] != reference[name]
            println("  tcvar_gibbs_regression: $name checksum is ",
                    repr(checksums[name]), ", reference ", repr(reference[name]))
        end
        @test checksums[name] == reference[name]
    end
end
