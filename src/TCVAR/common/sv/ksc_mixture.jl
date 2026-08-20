# The Kim, Shephard & Chib (1998) mixture approximation of `log χ²₁`, and the indicator
# draw that goes with it.
#
# A stochastic volatility measurement equation is multiplicative,
#
#     e_t = exp(h_t / 2) · η_t,      η_t ~ N(0, 1)
#
# which the log square turns additive,
#
#     y*_t = log(e_t²) = h_t + log(η_t²),     log(η_t²) ~ log χ²₁
#
# — linear in `h_t`, but with a non-Gaussian error, so a Kalman smoother cannot be applied
# directly. KSC replace `log χ²₁` by a seven-component Gaussian mixture; conditional on
# which component each observation came from, the system *is* linear Gaussian and the
# whole `h` path is drawn in one Carter–Kohn sweep. The component labels `s_t` become an
# extra Gibbs block, drawn here.

"""
    LOG_CHISQ1_MEAN

`E[log χ²₁] = ψ(1/2) + log 2 ≈ -1.2704`, the mean the mixture has to reproduce. The
component means of [`KSC_MIXTURE`](@ref) are the published table values shifted by this
constant: KSC tabulate the mixture for the *centred* variate `log χ²₁ − E[log χ²₁]`, so
using the raw table numbers would leave the volatility level biased upwards by 1.27 in
logs (a factor of `exp(1.2704/2) ≈ 1.89` in standard deviations).
"""
const LOG_CHISQ1_MEAN = -1.2704

"""
    KSC_MIXTURE

Seven-component Gaussian mixture approximating `log χ²₁` (Kim, Shephard & Chib 1998,
Table 4):

    log χ²₁  ≈  Σⱼ qⱼ · N(mⱼ, vⱼ²)

| j | qⱼ | mⱼ (raw) | vⱼ² |
|---|---|---|---|
| 1 | 0.00730 | -10.12999 | 5.79596 |
| 2 | 0.10556 |  -3.97281 | 2.61369 |
| 3 | 0.00002 |  -8.56686 | 5.17950 |
| 4 | 0.04395 |   2.77786 | 0.16735 |
| 5 | 0.34001 |   0.61942 | 0.64009 |
| 6 | 0.24566 |   1.79518 | 0.34023 |
| 7 | 0.25750 |  -1.08819 | 1.26261 |

The `means` field holds the raw table values already shifted by
[`LOG_CHISQ1_MEAN`](@ref), so `dot(probabilities, means) ≈ -1.2704` and the mixture
variance is `≈ π²/2`. `components` holds the seven `Normal(mⱼ, vⱼ)` themselves, so the
indicator draw in [`draw_mixture_indicators`](@ref) evaluates a density rather than
re-deriving one, and `log_probabilities` caches `log qⱼ` for the log-domain branch there.

Omori, Chib, Shephard & Nakajima (2007) give a ten-component table that approximates the
same density more accurately; it is a drop-in replacement for the vectors below.
"""
const KSC_MIXTURE = let
    probabilities = [0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750]
    raw_means     = [-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819]
    variances     = [5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261]

    means               = raw_means .+ LOG_CHISQ1_MEAN
    standard_deviations = sqrt.(variances)

    (probabilities       = probabilities,
     means               = means,
     variances           = variances,
     standard_deviations = standard_deviations,
     components          = Normal.(means, standard_deviations))
     
end

"""
    draw_mixture_indicators(y_star, h) -> Matrix{Int}

Draw the KSC mixture component label of every observation, independently across series
and periods:

    P(s_it = j | ·)  ∝  qⱼ · N(y*_it − h_it; mⱼ, vⱼ²)

`y_star` and `h` are both `T × n` — the log squared residuals `log(e_it² + c̄)` and the
*current* log-volatility path over the same periods, `h_{1:T}` (not the `h_0` row).
Returns the `T × n` matrix of labels, each in `1:7`.

Each weight is the component probability times the component density at the deviation,
normalised over the seven components and handed to `rand(Categorical(·))`. Densities
underflow to zero once the deviation is some 90 standard deviations into the tail of even
the widest component (`|y*_it − h_it| ≳ 82`, reachable only from a residual near `1e-23`
or a badly misplaced `h`); there the same weights are rebuilt as `log qⱼ + logpdf(·)` and
shifted by their maximum, which rescales them into range without changing the
distribution, the shift cancelling in the normalisation.
"""
function draw_mixture_indicators(y_star::AbstractMatrix{<:Real}, h::AbstractMatrix{<:Real})
    size(y_star) == size(h) || throw(DimensionMismatch(
        "y_star is $(size(y_star)) but h is $(size(h)); both must be T × n over the " *
        "same periods (h_{1:T}, without the h_0 row)"))

    probabilities, components =
        KSC_MIXTURE.probabilities, KSC_MIXTURE.components
    n_components = length(components)

    indicators = Matrix{Int}(undef, size(y_star))
    weights = Vector{Float64}(undef, n_components)

    @inbounds for i in axes(y_star, 2), t in axes(y_star, 1)
        deviation = y_star[t, i] - h[t, i]

        for j in 1:n_components
            weights[j] = probabilities[j] * pdf(components[j], deviation)            
        end
 
        weights = weights ./sum(weights)  

        # `Categorical` wraps `weights` without copying, so the buffer is reused across
        # observations; it is normalised in place above because the constructor requires
        # a probability vector.
        indicators[t, i] = rand(Categorical(weights))
    end

    return indicators
end
