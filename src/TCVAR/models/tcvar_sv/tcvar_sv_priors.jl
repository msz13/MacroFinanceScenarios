# The prior tuple of TCVAR-SV, fixed as a named type rather than as a convention.
#
# TCVAR gets away with `priors::NamedTuple` and a `haskey` loop because it has five keys.
# TCVAR-SV has nine, four of them new, and the sweep reads them in eight different places —
# so the shape is written down once, in the type, and the constructor below is the only
# door into it.

"""
    TCVARSVPriors

The nine priors of a TCVAR-SV model, as a fixed-shape `NamedTuple` type.

| key | type | shape | consumed by |
|---|---|---|---|
| `initial_trend` | `MvNormal` | `n_trends` | state draw — `τ₀` |
| `initial_cycle` | `MvNormal` | `n*p` | state draw — `ξ₀ = [c_{-p+1}; …; c_0]`, oldest-lag-first |
| `trend_covariance` | `InverseWishart` | `n_trends × n_trends` | `Στ`; scale used as written, no rescaling |
| `cycle_covariance` | `InverseWishart` | `n × n` | `Σ̄ = mean(·)` only — **not a sampled block**, see below |
| `cycle_β` | `MinnesotaPrior` | `Φ₀ : k × n`, `Ω : k × k` | `β`; the sole source of `n` and `p` |
| `volatility_mean` | `MvNormal` | `n` | `μ` |
| `volatility_ar` | `MvNormal` | `n` (`:diagonal`) or `n²` (`:full`) | `Φ` |
| `volatility_covariance` | `InverseWishart` | `n × n` | `Ω` |
| `simultaneity` | `MvNormal` | `n(n−1)÷2` | `A₀`, free elements row by row: `A₀[2,1], A₀[3,1], A₀[3,2], …` |

The first five keys are exactly the TCVAR tuple (so the output of [`var_priors`](@ref)
drops straight in) and the last four are exactly what [`sv_priors`](@ref) returns;
[`tcvar_sv_priors`](@ref) is an assembly-plus-validation function, not a new source of
prior distributions.

**`cycle_covariance` under stochastic volatility.** `Σc` is no longer a free parameter —
`Σ_t = A₀⁻¹ H_t A₀⁻ᵀ` is. The key stays because two things still need
`Σ̄ = mean(priors.cycle_covariance)`: the Minnesota prior precision `P₀ = Σ̄⁻¹ ⊗ Ω_M⁻¹` of
the coefficient block, and the pilot initialisation of the volatility path. There is no
`Σc` in the posterior chain.

**There is no `initial_volatility` key.** Unlike `τ₀` and `ξ₀`, `h₀` has no prior of its
own: `h_0 ~ N(μ, P₀)` with `P₀` the stationary covariance implied by the current `(Φ, Ω)`.
The diffuse fallback used when `Φ` has a unit root is a keyword of the sweep, not a
prior — it is a numerical fallback rather than a belief.

!!! note "Construct, never `isa`"
    `NamedTuple` is invariant in its value-type parameter, so a tuple carrying a
    `FullNormal` is **not** `isa TCVARSVPriors` even with the keys in the declared order.
    Validate by constructing: `TCVARSVPriors(nt)` selects fields by name (key order at the
    call site is free), converts, and throws on a missing key or a wrong distribution
    type. It also *drops* keys the type does not name — which is what makes the stored
    tuple canonical, and why the keyword form of [`tcvar_sv_priors`](@ref) is the
    recommended entry point: there a stray key is a loud `MethodError`.
"""
const TCVARSVPriors = @NamedTuple{
    initial_trend         :: MvNormal,        # τ₀            length n_trends
    initial_cycle         :: MvNormal,        # ξ₀            length n*p, oldest-lag-first
    trend_covariance      :: InverseWishart,  # Στ            n_trends × n_trends
    cycle_covariance      :: InverseWishart,  # Σ̄  (mean only) n × n
    cycle_β               :: MinnesotaPrior,  # β             carries n, p, k
    volatility_mean       :: MvNormal,        # μ             length n
    volatility_ar         :: MvNormal,        # Φ             length n (:diagonal) / n² (:full)
    volatility_covariance :: InverseWishart,  # Ω             n × n
    simultaneity          :: MvNormal,        # A₀            length n(n−1)÷2
}

"""
    volatility_ar_length(n, ar_structure) -> Int

Length of the `volatility_ar` prior for `n` series: `n` for a diagonal `Φ` (one
persistence per series) and `n^2` for a full `Φ` (the prior is on `vec(Φᵀ)`).
"""
function volatility_ar_length(n::Integer, ar_structure::Symbol)
    ar_structure === :diagonal && return n
    ar_structure === :full     && return n^2
    throw(ArgumentError("ar_structure must be :diagonal or :full, got :$ar_structure"))
end

"""
    tcvar_sv_priors(; initial_trend, initial_cycle, trend_covariance, cycle_covariance,
                      cycle_β, volatility_mean, volatility_ar, volatility_covariance,
                      simultaneity, ar_structure = :diagonal) -> TCVARSVPriors

    tcvar_sv_priors(tc_keys::NamedTuple, sv_keys::NamedTuple; ar_structure = :diagonal)

Assemble and validate the nine priors of a TCVAR-SV model (see [`TCVARSVPriors`](@ref)).

The keyword form is the canonical entry point — an unknown keyword is a `MethodError`
there, which is the error a typo deserves. The two-tuple form is the convenience one that
merges the output of [`var_priors`](@ref) and [`sv_priors`](@ref); it selects the nine
named keys out of the merge and silently ignores anything else.

Everything checked here is *internal to the tuple*, read off `cycle_β.n` and `cycle_β.p`.
The checks that need the trend mapping — that `n` matches the number of observed series,
and that `n_trends` matches its columns — belong to the [`TCVARSV`](@ref) constructor.
`ar_structure` is not stored in the tuple; it only selects the expected length of
`volatility_ar`, and [`TCVARSV`](@ref) re-checks that length against its own field so the
two cannot drift apart.

```julia
Σc_prior, β_prior, c₀_prior = var_priors(0.2, 1, [0.5, 1.0, 2.0] .^ 2; δ = zeros(3))

priors = tcvar_sv_priors(
    (initial_trend    = MvNormal(τ₀_mean, τ₀_cov),
     initial_cycle    = c₀_prior,
     trend_covariance = InverseWishart(dτ, Ψτ),
     cycle_covariance = Σc_prior,
     cycle_β          = β_prior),
    sv_priors(3))                      # the four SV keys, unchanged
```
"""
function tcvar_sv_priors(; initial_trend, initial_cycle, trend_covariance, cycle_covariance,
                           cycle_β, volatility_mean, volatility_ar, volatility_covariance,
                           simultaneity, ar_structure::Symbol = :diagonal)

    n, p = cycle_β.n, cycle_β.p
    n_ar = volatility_ar_length(n, ar_structure)   # also validates ar_structure

    length(initial_cycle) == n * p || throw(DimensionMismatch(
        "initial_cycle must have length n*p = $(n * p), got $(length(initial_cycle))"))
    size(cycle_covariance) == (n, n) || throw(DimensionMismatch(
        "cycle_covariance must be $n × $n, got $(size(cycle_covariance))"))
    size(volatility_covariance) == (n, n) || throw(DimensionMismatch(
        "volatility_covariance must be $n × $n, got $(size(volatility_covariance))"))
    length(volatility_mean) == n || throw(DimensionMismatch(
        "volatility_mean must have length $n, got $(length(volatility_mean))"))
    length(volatility_ar) == n_ar || throw(DimensionMismatch(
        "volatility_ar must have length $n_ar for ar_structure = :$ar_structure, " *
        "got $(length(volatility_ar))"))
    length(simultaneity) == n * (n - 1) ÷ 2 || throw(DimensionMismatch(
        "simultaneity must have length n*(n-1)÷2 = $(n * (n - 1) ÷ 2), " *
        "got $(length(simultaneity))"))
    size(trend_covariance, 1) == length(initial_trend) || throw(DimensionMismatch(
        "trend_covariance is $(size(trend_covariance, 1)) × $(size(trend_covariance, 1)) " *
        "but initial_trend has length $(length(initial_trend)); both are n_trends"))

    return TCVARSVPriors((initial_trend         = initial_trend,
                          initial_cycle         = initial_cycle,
                          trend_covariance      = trend_covariance,
                          cycle_covariance      = cycle_covariance,
                          cycle_β               = cycle_β,
                          volatility_mean       = volatility_mean,
                          volatility_ar         = volatility_ar,
                          volatility_covariance = volatility_covariance,
                          simultaneity          = simultaneity))
end

function tcvar_sv_priors(tc_keys::NamedTuple, sv_keys::NamedTuple;
                         ar_structure::Symbol = :diagonal)
    # Selecting the nine names is what drops the extras; a *missing* key still throws
    # (a FieldError naming it) rather than reaching the keyword method as `nothing`.
    selected = NamedTuple{fieldnames(TCVARSVPriors)}(merge(tc_keys, sv_keys))
    return tcvar_sv_priors(; ar_structure = ar_structure, selected...)
end
