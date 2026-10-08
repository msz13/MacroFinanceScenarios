# AR(1) fitted by OLS: x_t = c + φ x_{t-1} + e_t. Dependency-free (plain least squares).

"""
    AR1

OLS fit of `x_t = c + φ x_{t-1} + e_t`.

- `c`, `φ`    coefficients; `se = (se_c, se_φ)` their OLS standard errors
- `σ`         residual std, with `n − 2` degrees of freedom
- `r2`        R² of the regression
- `resid`, `fitted`, `years`: one entry per observation `t = 2..n` (the first is lost to the lag)
- `period`    `(first_year, last_year)` of the sample passed to `fit_ar1`, including the lag year
"""
struct AR1
    c::Float64
    φ::Float64
    σ::Float64
    se::NTuple{2,Float64}
    r2::Float64
    resid::Vector{Float64}
    fitted::Vector{Float64}
    years::Vector{Int}
    period::Tuple{Int,Int}
end

"""
    fit_ar1(x::AbstractVector; years=1:length(x)) -> AR1
    fit_ar1(ta::TimeArray) -> AR1
    fit_ar1(ta::TimeArray, var::Symbol) -> AR1

Fit an AR(1) by OLS. With a `TimeArray` (one column, or column `var`), the years come from
its timestamps, so calling it on a `subperiod` slice records that period.
"""
function fit_ar1(x::AbstractVector{<:Real}; years::AbstractVector{<:Integer} = 1:length(x))
    n = length(x)
    n >= 3 || throw(ArgumentError("fit_ar1 needs at least 3 observations, got $n"))
    length(years) == n || throw(DimensionMismatch("years has length $(length(years)), x has length $n"))

    y = x[2:end]
    X = [ones(n - 1) x[1:end-1]]
    β = X \ y
    fitted = X * β
    resid = y .- fitted

    σ² = sum(abs2, resid) / (n - 1 - 2)
    se = sqrt.(σ² .* diag(inv(X' * X)))
    r2 = 1 - sum(abs2, resid) / sum(abs2, y .- mean(y))

    return AR1(β[1], β[2], sqrt(σ²), (se[1], se[2]), r2, resid, fitted,
               collect(Int, years[2:end]), (Int(first(years)), Int(last(years))))
end

function fit_ar1(ta::TimeArray)
    length(colnames(ta)) == 1 || throw(ArgumentError("fit_ar1 needs a single-column TimeArray; pass the column name"))
    return fit_ar1(vec(values(ta)); years = year.(timestamp(ta)))
end

fit_ar1(ta::TimeArray, var::Symbol) = fit_ar1(ta[var])

"""Unconditional mean `c / (1 − φ)` of a stationary AR(1)."""
unconditional_mean(ar::AR1) = ar.c / (1 - ar.φ)

"""Unconditional std `σ / √(1 − φ²)` of a stationary AR(1)."""
unconditional_std(ar::AR1) = ar.σ / sqrt(1 - ar.φ^2)
