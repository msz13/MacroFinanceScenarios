# Model-agnostic posterior distributions and the draws taken from them.
#
# Two kinds of function live here, and the split is what makes the layer testable:
#
#   *_posterior(...)  pure, no RNG — returns a Distributions.jl object or posterior
#                     moments, so it can be checked against closed-form conjugate
#                     results without seeding anything.
#   draw_*(...)       consumes RNG, returns a draw. Thin: build the posterior, `rand` it.
#
# Nothing here knows about trends, cycles or the Minnesota prior; the model- and
# VAR-specific scales are assembled by the callers in `var/` and `models/`. The
# inverse-Wishart constructor, conjugate coefficient mean and Kronecker factor live
# with the natural-conjugate VAR posterior in `var/natural_conjugate.jl`.

"""
    random_walk_covariance_posterior(states, scale_prior, df_posterior) -> InverseWishart

Covariance posterior of a random-walk state, `xₜ = xₜ₋₁ + εₜ` with `εₜ ~ N(0, Σ)`.

The innovations are the first differences of the sampled state path, so this is
[`inverse_wishart_posterior`](@ref) applied to `diff(states, dims = 1)`. `states`
is `T × n` and must include the pre-sample point (a path of `T` points yields
`T − 1` innovations, which is what `df_posterior` should account for).
"""
random_walk_covariance_posterior(states, scale_prior, df_posterior) =
    inverse_wishart_posterior(diff(states, dims = 1), scale_prior, df_posterior)

"""
    draw_from_factor(mean, L) -> Vector

Draw from `N(vec(mean), L L')` given a precomputed lower-triangular factor `L`
(e.g. from [`kron_cholesky_factor`](@ref)). `mean` may be a matrix; it is
vectorised column-wise to match the Kronecker layout.
"""
draw_from_factor(mean, L) = vec(mean) + L * randn(size(L, 1))
