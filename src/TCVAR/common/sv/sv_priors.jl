# Priors of the stochastic-volatility layer, in the same distribution-keyed NamedTuple
# style as `models/tcvar/tcvar_priors.jl`: every key is a `Distributions.jl` object whose
# *scale* is what the sampler consumes, so a prior mean stated in economic terms is
# converted here rather than inside a sweep.

"""
    sv_priors(n; volatility_level = 0.1, mean_sd = 2.0, ar_mean = 0.8, ar_sd = 0.2,
                 covariance_sd = 0.2, covariance_df = n + 11, simultaneity_variance = 10.0,
                 ar_structure = :diagonal)
        -> NamedTuple

Priors for an `n`-series stochastic-volatility block:

* `volatility_mean` — `μ ~ N(log(volatility_level²)·1, mean_sd²·I)`, the unconditional
  level of each log variance. Centred so that the prior median innovation standard
  deviation is `volatility_level`.
* `volatility_ar` — `diag(Φ) ~ N(ar_mean·1, ar_sd²·I)`, the AR(1) persistence of each
  log variance (Clark & Ravazzolo 2015). The default `Φ` is diagonal; under
  `ar_structure = :full` the same prior is lifted to the `n²` elements of `vec(Φᵀ)` —
  mean `vec(ar_mean·I)`, covariance `ar_sd²·I` — so a full `Φ` is centred on the same
  diagonal AR(1) and every off-diagonal spillover is shrunk towards zero with the same
  tightness.
* `volatility_covariance` — `Ω ~ IW(covariance_df, Ψ)` with `Ψ` set so that
  `mean(Ω) = covariance_sd²·I`.
* `simultaneity` — `N(0, simultaneity_variance·I)` on the `n(n−1)/2` free elements of the
  unit-lower-triangular `A₀`, stacked row by row (`A₀[2,1]`, `A₀[3,1]`, `A₀[3,2]`, …).
  Deliberately loose: `A₀` is well identified by the residuals and shrinking it towards
  zero is shrinking towards a diagonal `Σ_t`, which is not a defensible prior belief.

`mean_sd` is wide by design. The natural prior on `μ` is the ergodic one,
`N(log(volatility_level²), σ²/(1−ρ²))`, but that conditions on the current `(Φ, Ω)` draws
and so is not a prior at all — it would make the conjugate `μ` update circular. A fixed,
generously wide normal at the same centre keeps the update conjugate; at the default
`mean_sd = 2` the prior spans a factor of `e ≈ 2.7` in volatility per standard deviation.

# Example
```julia
priors = sv_priors(3)
mean(priors.volatility_covariance)          # 0.04·I
```
"""
function sv_priors(n::Integer;
                   volatility_level::Real = 0.1,
                   mean_sd::Real = 2.0,
                   ar_mean::Real = 0.8,
                   ar_sd::Real = 0.2,
                   covariance_sd::Real = 0.2,
                   covariance_df::Real = n + 11,
                   simultaneity_variance::Real = 10.0,
                   ar_structure::Symbol = :diagonal)

    n ≥ 1 || throw(ArgumentError("n must be ≥ 1, got $n"))
    volatility_level > 0 || throw(ArgumentError("volatility_level must be > 0, got $volatility_level"))
    mean_sd > 0 || throw(ArgumentError("mean_sd must be > 0, got $mean_sd"))
    ar_sd > 0 || throw(ArgumentError("ar_sd must be > 0, got $ar_sd"))
    covariance_sd > 0 || throw(ArgumentError("covariance_sd must be > 0, got $covariance_sd"))
    covariance_df > n + 1 || throw(ArgumentError(
        "covariance_df must exceed n + 1 = $(n + 1) for the IW prior to have a mean, " *
        "got $covariance_df"))
    simultaneity_variance > 0 || throw(ArgumentError(
        "simultaneity_variance must be > 0, got $simultaneity_variance"))
    ar_structure in (:diagonal, :full) || throw(ArgumentError(
        "ar_structure must be :diagonal or :full, got :$ar_structure"))

    df = float(covariance_df)
    # IW(df, Ψ) has mean Ψ/(df − n − 1), so the scale carries the (df − n − 1) factor.
    scale = Matrix(covariance_sd^2 * (df - n - 1) * I, n, n)

    n_simultaneity = n * (n - 1) ÷ 2

    # :diagonal puts the prior on diag(Φ); :full puts it on vec(Φᵀ), centred on the same
    # diagonal matrix so the two structures agree wherever they overlap.
    ar_mean_vector = ar_structure === :diagonal ? fill(float(ar_mean), n) :
                                                  vec(Matrix(float(ar_mean) * I, n, n))
    n_ar = length(ar_mean_vector)

    return (volatility_mean       = MvNormal(fill(log(float(volatility_level)^2), n),
                                             Matrix(float(mean_sd)^2 * I, n, n)),
            volatility_ar         = MvNormal(ar_mean_vector,
                                             Matrix(float(ar_sd)^2 * I, n_ar, n_ar)),
            volatility_covariance = InverseWishart(df, scale),
            simultaneity          = MvNormal(zeros(n_simultaneity),
                                             Matrix(float(simultaneity_variance) * I,
                                                    n_simultaneity, n_simultaneity)))
end
