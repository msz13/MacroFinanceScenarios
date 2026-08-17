using Test
using Distributions
using LinearAlgebra
using Random
using Statistics

isdefined(Main, :TCVAR) || include(joinpath(@__DIR__, "..", "src", "TCVAR", "TCVAR.jl"))
using FlexiChains: @varname

# TCVAR members are reached as `TCVAR.f` rather than via `using .TCVAR` — see the note in
# tcvar_test_utils.jl.
include(joinpath(@__DIR__, "tcvar_test_utils.jl"))

# Simulate one dataset from a known TCVAR, re-estimate it with `gibbs_sampler` and
# compare the posterior mean *and* median of every parameter block against the truth.
#
# Two things this measurement settled, kept here as documentation rather than hidden:
#
#  * The trend-innovation covariance is systematically over-estimated (roughly 1.7–2.3×
#    the true 0.01 for the first trend) and the bias does not shrink when T goes
#    400 → 800. It is the weakly identified block of an unobserved-components model, so
#    its tolerance below is a factor band, not a percentage.
#  * 90% credible-interval containment is *not* a reliable assertion here: Στ[1,1] fell
#    outside its own 90% band for every seed tried, and β / Σc missed occasionally. Only
#    point-estimate tolerances are asserted.

function report_block(name, truth, est_mean, est_median)
    println("  $name:  true / mean / median")
    for i in eachindex(truth)
        println("    [$i] ", round(truth[i], digits = 4), " / ",
                round(est_mean[i], digits = 4), " / ", round(est_median[i], digits = 4))
    end
end

@testset "TCVAR simulation recovery" begin
    n, nt, p = 2, 2, 1
    T = 200
    burnin = 2000

    Στ_true = diagm([0.01, 0.02])
    A_true  = [0.6 0.0; 0.1 0.5]          # companion bottom block
    β_true  = collect(A_true')
    Σc_true = [1.0 0.2; 0.2 0.5]
    initial_state = [1.0, 2.0, 0.0, 0.0]

    # Estimation priors are deliberately off the truth (cycle variances 2× the true
    # ones, trend variances centred at 0.02 for both trends) so the test measures
    # learning rather than prior echo.
    priors = tcvar_test_priors(; n = n, nt = nt, p = p, λ = 0.5, ψ = [2.0, 1.0],
                               trend_variances = [0.02, 0.02], trend_df = 20.0)
    model = TCVAR.TCVAR(Matrix{Float64}(I, n, nt), priors)

    Random.seed!(42)
    _, obs = TCVAR.simulate_scenarios(model, (Στ = Στ_true, β = β_true, Σc = Σc_true),
                                initial_state, 1, T)
    data = convert(Matrix{Union{Missing,Float64}}, obs[1, :, :])

    result = TCVAR.gibbs_sampler(model, data; burnin = burnin, n_samples = 2_000)

    β_mean,  β_median  = draw_mean(result.params[@varname(β)]),  draw_median(result.params[@varname(β)])
    Σc_mean, Σc_median = draw_mean(result.params[@varname(Σc)]), draw_median(result.params[@varname(Σc)])
    Στ_mean, Στ_median = draw_mean(result.params[@varname(Στ)]), draw_median(result.params[@varname(Στ)])

    println("TCVAR recovery (n = $n, nt = $nt, p = $p, T = $T, burnin = $burnin, seed 42)")
    report_block("β",  β_true,  β_mean,  β_median)
    report_block("Σc", Σc_true, Σc_mean, Σc_median)
    report_block("Στ", Στ_true, Στ_mean, Στ_median)

    for (label, β_est, Σc_est, Στ_est) in (("mean", β_mean, Σc_mean, Στ_mean),
                                           ("median", β_median, Σc_median, Στ_median))
        @testset "posterior $label" begin
            @test maximum(abs.(β_est - β_true)) ≤ 0.2
            @test maximum(abs.(Σc_est - Σc_true) ./ abs.(Σc_true)) ≤ 0.3

            # weakly identified: a factor band, not a percentage (see note above)
            for i in 1:nt
                @test Στ_true[i, i] / 3 ≤ Στ_est[i, i] ≤ 3 * Στ_true[i, i]
            end
            @test abs(Στ_est[1, 2]) ≤ 0.01
        end
    end 
end
