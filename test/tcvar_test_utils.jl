using Distributions
using LinearAlgebra
using Statistics

isdefined(Main, :TCVAR) || include(joinpath(@__DIR__, "..", "src", "TCVAR", "TCVAR.jl"))
using .TCVAR

"""
    tcvar_test_priors(; n, nt, p, λ, ψ, trend_variances, trend_df) -> NamedTuple

Build the five-key priors NamedTuple used across the TCVAR tests.

`trend_covariance` is written as `InverseWishart(df, diagm(v) * (df - nt - 1))` so that
`mean(trend_covariance) == diagm(trend_variances)`: the sampler now takes the IW scale
at face value, so a *prior mean* has to be converted to a scale explicitly here rather
than inside the sampler.
"""
function tcvar_test_priors(; n::Int = 2, nt::Int = 2, p::Int = 1, λ::Real = 0.5,
                           ψ::AbstractVector = fill(1.0, n),
                           trend_variances::AbstractVector = fill(0.02, nt),
                           trend_df::Real = 20.0,
                           initial_trend_mean::AbstractVector = zeros(nt),
                           initial_trend_covariance::AbstractMatrix = diagm(ones(nt)))

    Σ_prior, β_prior, c0_prior = var_priors(λ, p, ψ; δ = zeros(n))

    return (initial_trend    = MvNormal(collect(float.(initial_trend_mean)),
                                        Matrix(float.(initial_trend_covariance))),
            initial_cycle    = c0_prior,
            trend_covariance = InverseWishart(float(trend_df),
                                              diagm(collect(float.(trend_variances))) *
                                              (float(trend_df) - nt - 1)),
            cycle_covariance = Σ_prior,
            cycle_β          = β_prior)
end

"""
    draw_matrix(draws) -> Array{Float64,3}

Stack a `DimMatrix` of matrix-valued draws (as returned by
`result.params[@varname(β)]`) into an `n_rows × n_cols × n_draws` array.
"""
function draw_matrix(draws)
    flat = reduce(hcat, vec.(vec(collect(draws))))   # n_elements × n_draws
    r, c = size(first(vec(collect(draws))))
    return reshape(flat, r, c, size(flat, 2))
end

"""
    draw_mean(draws), draw_median(draws) -> Matrix

Element-wise posterior mean / median of a matrix-valued parameter's draws. `mean` works
directly on the `DimMatrix` of matrices; `median` does not, so the draws are stacked first.
"""
draw_mean(draws) = mean(draws)

function draw_median(draws)
    stacked = draw_matrix(draws)
    return dropdims(median(stacked, dims = 3), dims = 3)
end
