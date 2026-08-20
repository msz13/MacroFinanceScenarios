using Distributions
using LinearAlgebra
using Statistics

isdefined(Main, :TCVAR) || include(joinpath(@__DIR__, "..", "src", "TCVAR", "TCVAR.jl"))

# Module members are reached as `TCVAR.f`, deliberately, instead of `using .TCVAR`.
# Including TCVAR.jl a second time in a live session (a REPL / IDE worker that already
# loaded it) defines a second `TCVAR` module without retiring the first, and every name
# imported by `using` is then exported by two modules at once — Julia refuses to resolve
# the ambiguity and reports `var_priors` and friends as undefined in `Main`. Qualified
# access goes through the `Main.TCVAR` binding, which always points at the newest module,
# so the tests keep working in such a session.

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

    Σ_prior, β_prior, c0_prior = TCVAR.var_priors(λ, p, ψ; δ = zeros(n))

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

"""
    tcvar_sv_test_priors(; n, nt, p, λ, ψ, trend_variances, trend_df, ar_structure, …)
        -> TCVARSVPriors

The nine-key TCVAR-SV prior tuple used across the TCVAR-SV tests: the five TCVAR keys of
[`tcvar_test_priors`](@ref) merged with `TCVAR.sv_priors(n; …)`, assembled and validated by
`TCVAR.tcvar_sv_priors`.
"""
function tcvar_sv_test_priors(; n::Int = 3, nt::Int = 3, p::Int = 1, λ::Real = 0.5,
                              ψ::AbstractVector = fill(1.0, n),
                              trend_variances::AbstractVector = fill(0.02, nt),
                              trend_df::Real = 20.0,
                              initial_trend_mean::AbstractVector = zeros(nt),
                              initial_trend_covariance::AbstractMatrix = diagm(ones(nt)),
                              ar_structure::Symbol = :diagonal,
                              volatility_level::Real = 0.1)

    tc_keys = tcvar_test_priors(; n = n, nt = nt, p = p, λ = λ, ψ = ψ,
                                trend_variances = trend_variances, trend_df = trend_df,
                                initial_trend_mean = initial_trend_mean,
                                initial_trend_covariance = initial_trend_covariance)

    sv_keys = TCVAR.sv_priors(n; ar_structure = ar_structure,
                              volatility_level = volatility_level)

    return TCVAR.tcvar_sv_priors(tc_keys, sv_keys; ar_structure = ar_structure)
end
