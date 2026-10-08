# VAR(1) with bootstrapped residuals: y_t = c + A y_{t-1} + u_t, parametrised by the
# intercepts c (length n) and the lag matrix A (n × n). The Ibbotson–Sinquefield model is
# the special case with a diagonal A (AR(1) for π, r and tp, i.i.d. erp around its intercept).

"""
    Scenarios(data, names, start_year)

Simulated paths: `data` is `n_vars × T × n_scen`, `names` labels the rows and `start_year`
is the calendar year of the first simulated period.
"""
struct Scenarios
    data::Array{Float64,3}
    names::Vector{Symbol}
    start_year::Int

    function Scenarios(data::AbstractArray{<:Real,3}, names::AbstractVector{Symbol}, start_year::Integer)
        size(data, 1) == length(names) ||
            throw(DimensionMismatch("data has $(size(data, 1)) variables, names has $(length(names))"))
        return new(Array{Float64,3}(data), collect(names), Int(start_year))
    end
end

"""Simulated years covered by `sc`."""
scenario_years(sc::Scenarios) = sc.start_year .+ (0:size(sc.data, 2)-1)

"""`T × n_scen` paths of variable `name`."""
Base.getindex(sc::Scenarios, name::Symbol) = sc.data[varindex(sc.names, name), :, :]

function varindex(names::AbstractVector{Symbol}, name::Symbol)
    i = findfirst(==(name), names)
    i === nothing && throw(ArgumentError("variable $name not in $names"))
    return i
end

# Check that c, A and names describe the same n-variable VAR(1).
function check_var_dims(c::AbstractVector, A::AbstractMatrix, names::AbstractVector{Symbol})
    n = length(c)
    size(A) == (n, n) || throw(DimensionMismatch("A must be $n × $n to match c, got $(size(A))"))
    length(names) == n || throw(DimensionMismatch("c has length $n, names has $(length(names))"))
    return n
end

"""Unconditional mean `(I − A)⁻¹ c` of the VAR(1) with intercepts `c` and lag matrix `A`."""
unconditional_mean(c::AbstractVector, A::AbstractMatrix) = (I - A) \ c

"""
    is_coefficients(ar_π::AR1, ar_r::AR1, ar_tp::AR1; erp_target, tp_mean=nothing) -> (c, A, names)

Intercepts `c` and lag matrix `A` of the Ibbotson–Sinquefield model, state `[π, r, tp, erp]`:
π, r and the term spread tp follow their AR(1) fits; erp has zero lag coefficients and
intercept `erp_target` (its simulated mean, required). `tp_mean` optionally overrides the
unconditional mean of tp: its intercept becomes `tp_mean·(1 − φ_tp)`, keeping `φ_tp`.
"""
function is_coefficients(ar_π::AR1, ar_r::AR1, ar_tp::AR1; erp_target::Real,
                         tp_mean::Union{Real,Nothing} = nothing)
    c_tp = tp_mean === nothing ? ar_tp.c : tp_mean * (1 - ar_tp.φ)
    c = [ar_π.c, ar_r.c, c_tp, Float64(erp_target)]
    A = [ar_π.φ 0.0    0.0     0.0
         0.0    ar_r.φ 0.0     0.0
         0.0    0.0    ar_tp.φ 0.0
         0.0    0.0    0.0     0.0]
    return c, A, [:π, :r, :tp, :erp]
end

"""
    issm_bootstrap_matrix(data::TimeArray, c, A, names, period) -> (years, U)

Residuals `u_t = y_t − c − A y_{t-1}` of the VAR(1) with intercepts `c` and lag matrix `A`
(rows `names`, which must be columns of `data`) over the years in `period = (y0, y1)`. The
first year of the period is lost to the lag, so `U` is `(n_years − 1) × n` and every column
comes from the same years. Columns are demeaned, so the simulated means come from `c` and
`A` alone.
"""
function issm_bootstrap_matrix(data::TimeArray, c::AbstractVector{<:Real}, A::AbstractMatrix{<:Real},
                               names::AbstractVector{Symbol}, period::Tuple{Integer,Integer})
    check_var_dims(c, A, names)
    sub = subperiod(data, period)
    Y = values(sub[names...])
    size(Y, 1) >= 2 || throw(ArgumentError("period $period has fewer than 2 years of data"))

    U = Y[2:end, :] .- c' .- Y[1:end-1, :] * A'
    U .-= mean(U; dims = 1)
    return year.(timestamp(sub))[2:end], U
end

"""
    simulate_var_bootstrapped_res(c, A, names, U, T, n_scen;
                                  y0=:last, data=nothing, start_year=nothing,
                                  rng=Random.default_rng()) -> Scenarios

Simulate `n_scen` paths of length `T` from `y_t = c + A y_{t-1} + u_t`, with intercepts `c`
(length n) and lag matrix `A` (n × n), rows labelled by `names`. Each `u_t` is
a whole row of `U` drawn i.i.d. with replacement (keeping the cross-sectional correlation
of the shocks).

- `y0 = :last` starts from the last observation of `names` in the `TimeArray` `data`;
  `y0 = :mean` from the unconditional mean `(I − A)⁻¹ c`; a vector sets it explicitly.
- `start_year`: year of the first simulated period; defaults to the year after the last
  observation in `data`, or 1 without `data`.

Throws if `A` has an eigenvalue on or outside the unit circle.
"""
function simulate_var_bootstrapped_res(c::AbstractVector{<:Real}, A::AbstractMatrix{<:Real},
                                       names::AbstractVector{Symbol},
                                       U::AbstractMatrix, T::Integer, n_scen::Integer;
                                       y0::Union{Symbol,AbstractVector{<:Real}} = :last,
                                       data::Union{TimeArray,Nothing} = nothing,
                                       start_year::Union{Integer,Nothing} = nothing,
                                       rng::AbstractRNG = Random.default_rng())
    n = check_var_dims(c, A, names)
    size(U, 2) == n || throw(DimensionMismatch("U has $(size(U, 2)) columns, expected $n"))
    size(U, 1) >= 1 || throw(ArgumentError("U has no rows"))
    ρ = maximum(abs, eigvals(A))
    ρ < 1 || throw(ArgumentError("A is not stable: spectral radius $ρ ≥ 1"))

    y_init = if y0 === :last
        data === nothing && throw(ArgumentError("y0 = :last needs the observed `data`"))
        vec(values(data[names...])[end, :])
    elseif y0 === :mean
        unconditional_mean(c, A)
    elseif y0 isa Symbol
        throw(ArgumentError("y0 must be :last, :mean or a vector, got :$y0"))
    else
        length(y0) == n || throw(DimensionMismatch("y0 has length $(length(y0)), expected $n"))
        Vector{Float64}(y0)
    end

    if start_year === nothing
        start_year = data === nothing ? 1 : year(timestamp(data)[end]) + 1
    end

    sims = Array{Float64,3}(undef, n, T, n_scen)
    Ut = permutedims(U)                       # n × n_obs: draw columns
    n_obs = size(Ut, 2)
    y, y_prev = similar(y_init), similar(y_init)
    for s in 1:n_scen
        y_prev .= y_init
        for t in 1:T
            k = rand(rng, 1:n_obs)
            mul!(y, A, y_prev)
            @views y .+= c .+ Ut[:, k]
            sims[:, t, s] .= y
            y, y_prev = y_prev, y
        end
    end
    return Scenarios(sims, names, start_year)
end

"""
    returns_from_components(sc::Scenarios; real=false, bond_maturity=10) -> Scenarios

Asset log returns from the components `π, r, tp, erp` of `sc`:

- T-bill `rf = π + r`
- 10y bond `rb`: `rf + tp` is the log 10y yield at the start of each year, and `rb_t` is the
  par-bond return (`calculate_bond_returns`, annual step) from the yield at the start of
  year t to the yield at the start of year t+1
- equity `re = rf + erp`

The last simulated year has no closing yield, so the result covers years `1:T-1` of `sc`
for every variable. With `real = true`, π is subtracted from each.
"""
function returns_from_components(sc::Scenarios; real::Bool = false, bond_maturity::Real = 10)
    T = size(sc.data, 2)
    T >= 2 || throw(ArgumentError("bond returns need at least 2 simulated years, got $T"))
    comp(name) = @view sc.data[varindex(sc.names, name), :, :]
    π, r, tp, erp = comp(:π), comp(:r), comp(:tp), comp(:erp)

    rf = π .+ r                                             # T × n_scen
    y10 = exp.(rf .+ tp) .- 1
    rb = log.(1 .+ calculate_bond_returns(y10, bond_maturity, 1))   # (T−1) × n_scen
    rf, re = rf[1:end-1, :], (rf .+ erp)[1:end-1, :]
    if real
        π1 = π[1:end-1, :]
        rf, rb, re = rf .- π1, rb .- π1, re .- π1
    end
    rets = cat(rf, rb, re; dims = 3)                        # (T−1) × n_scen × 3
    return Scenarios(permutedims(rets, (3, 1, 2)), [:rf, :rb, :re], sc.start_year)
end
