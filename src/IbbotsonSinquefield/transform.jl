# Column-wise, invertible transformations of TimeArrays, in the spirit of scikit-learn's
# ColumnTransformer: `fit_columns` learns the steps (e.g. the mean to subtract) on one
# sample, `transform_columns` applies them, `inverse_transform_columns` maps transformed
# data or simulated `Scenarios` back to the original scale.

"""
    Except(cols...)

Column selector for `fit_columns`/`transform_columns`: every column of the `TimeArray`
except `cols`. `Except()` selects all columns.
"""
struct Except
    cols::Vector{Symbol}
end
Except(cols::Symbol...) = Except(collect(cols))

select_columns(names, c::Symbol) = [c]
select_columns(names, cs::AbstractVector{Symbol}) = collect(cs)
select_columns(names, e::Except) = setdiff(names, e.cols)

abstract type ColumnStep end

"""
    Affine(a, b)

Step `x ↦ a x + b`, inverse `y ↦ (y − b) / a`.
"""
struct Affine <: ColumnStep
    a::Float64
    b::Float64
end
apply_step(s::Affine, x) = s.a .* x .+ s.b
invert_step(s::Affine, y) = (y .- s.b) ./ s.a

"""
    Invertible(f, finv)

Element-wise step `x ↦ f(x)` with inverse `finv`, both scalar functions (broadcast over the
column). In a spec, `f => finv` is shorthand, e.g. `:cape => log => exp`.
"""
struct Invertible{F,G} <: ColumnStep
    f::F
    finv::G
end
apply_step(s::Invertible, x) = s.f.(x)
invert_step(s::Invertible, y) = s.finv.(y)

"""
    Demean()

Subtract the column mean. Fitted to `Affine(1, −mean)`, so the mean is frozen at fit time.
"""
struct Demean <: ColumnStep end

"""
    Standardize()

Subtract the column mean and divide by its standard deviation. Fitted to
`Affine(1/σ, −μ/σ)`, so `μ` and `σ` are frozen at fit time.
"""
struct Standardize <: ColumnStep end

# A plain function on the whole column vector: applied as is, cannot be inverted, and is
# re-evaluated on the new data by `transform_columns(ct, ta)`.
struct VectorFunction{F} <: ColumnStep
    f::F
end
apply_step(s::VectorFunction, x) = s.f(x)
invert_step(::VectorFunction, y) =
    throw(ArgumentError("a plain function step cannot be inverted; use `f => finv`, " *
                        "Invertible(f, finv), Affine, Demean or Standardize"))

fit_step(::Demean, x) = Affine(1.0, -mean(x))
fit_step(::Standardize, x) = (σ = std(x); Affine(1 / σ, -mean(x) / σ))
fit_step(s::ColumnStep, x) = s
fit_step(p::Pair, x) = Invertible(p.first, p.second)
fit_step(f, x) = VectorFunction(f)

function apply_checked(step::ColumnStep, x::AbstractVector, col::Symbol)
    y = apply_step(step, x)
    length(y) == length(x) ||
        throw(DimensionMismatch("transformation of :$col returned $(length(y)) values, expected $(length(x))"))
    return y
end

columns_dict(ta::TimeArray) = Dict(n => values(ta)[:, j] for (j, n) in enumerate(colnames(ta)))

"""
    ColumnTransformer

Fitted column transformations, from `fit_columns`. `input` are the columns of the fitting
data, `output` the columns `transform_columns` returns, `steps` the fitted
`column => step` operations in the order they are applied.
"""
struct ColumnTransformer
    input::Vector{Symbol}
    output::Vector{Symbol}
    steps::Vector{Pair{Symbol,ColumnStep}}
end

"""
    fit_columns(ta, specs...; remainder=:passthrough) -> ColumnTransformer

Fit column-wise transformations on `ta`. Each spec is `selector => steps`:

- `selector` is a column name (`:cape`), a vector of names (`[:re, :rb]`) or
  `Except(:cape, ...)` for all columns but those listed;
- `steps` is one step or a vector/tuple of steps, applied left to right as a pipeline. A
  step is `Demean()`, `Standardize()`, `Affine(a, b)`, `Invertible(f, finv)` / `f => finv`
  (scalar functions, broadcast), or a plain function of the whole column vector returning
  a vector of the same length (not invertible).

Specs are applied in order, each to the output of the previous ones, so a column selected
by several specs gets all their steps. Columns not selected by any spec are kept unchanged
(`remainder = :passthrough`) or dropped (`remainder = :drop`). The original column order is
kept.

```julia
ct = fit_columns(ta,
    :cape         => log => exp,                      # invertible, element-wise
    Except(:cape) => [Affine(100, 0), Standardize()]) # percent, then z-score
z  = transform_columns(ct, ta)
ta ≈ inverse_transform_columns(ct, z)
```
"""
function fit_columns(ta::TimeArray, specs::Pair...; remainder::Symbol = :passthrough)
    remainder in (:passthrough, :drop) ||
        throw(ArgumentError("remainder must be :passthrough or :drop, got :$remainder"))
    names = colnames(ta)
    data = columns_dict(ta)
    steps = Pair{Symbol,ColumnStep}[]
    selected = Set{Symbol}()

    for (selector, fs) in specs
        cols = select_columns(names, selector)
        unknown = setdiff(cols, names)
        isempty(unknown) || throw(ArgumentError("unknown columns: $(join(unknown, ", "))"))
        pipeline = fs isa Union{AbstractVector,Tuple} ? fs : (fs,)
        for c in cols, f in pipeline
            step = fit_step(f, data[c])
            data[c] = apply_checked(step, data[c], c)
            push!(steps, c => step)
        end
        union!(selected, cols)
    end

    output = remainder == :drop ? filter(in(selected), names) : names
    isempty(output) && throw(ArgumentError("no columns left after remainder = :drop"))
    return ColumnTransformer(names, output, steps)
end

"""
    transform_columns(ct::ColumnTransformer, ta) -> TimeArray
    transform_columns(ta, specs...; remainder=:passthrough) -> TimeArray

Apply fitted transformations to `ta` (which may be a different sample from the one `ct` was
fitted on: fitted means and scales are reused). The second form fits on `ta` and applies in
one go; see `fit_columns` for the specs.
"""
function transform_columns(ct::ColumnTransformer, ta::TimeArray)
    data = columns_dict(ta)
    unknown = setdiff(union(first.(ct.steps), ct.output), keys(data))
    isempty(unknown) || throw(ArgumentError("missing columns: $(join(unknown, ", "))"))
    for (c, step) in ct.steps
        data[c] = apply_checked(step, data[c], c)
    end
    return TimeArray(timestamp(ta), hcat((data[c] for c in ct.output)...), ct.output)
end

transform_columns(ta::TimeArray, specs::Pair...; remainder::Symbol = :passthrough) =
    transform_columns(fit_columns(ta, specs...; remainder), ta)

"""
    inverse_transform_columns(ct::ColumnTransformer, ta::TimeArray) -> TimeArray
    inverse_transform_columns(ct::ColumnTransformer, sc::Scenarios) -> Scenarios

Map transformed data back to the original scale by undoing the steps of `ct` in reverse
order. The columns (variables of `sc`) may be any subset of `ct.output`, in any order; each
is inverted on its own and the names and order are kept. Fails if a step on a present
column is a plain (non-invertible) function.
"""
function inverse_transform_columns(ct::ColumnTransformer, ta::TimeArray)
    names = colnames(ta)
    check_output_columns(ct, names)
    data = columns_dict(ta)
    for (c, step) in Iterators.reverse(ct.steps)
        haskey(data, c) && (data[c] = invert_step(step, data[c]))
    end
    return TimeArray(timestamp(ta), hcat((data[c] for c in names)...), names)
end

function inverse_transform_columns(ct::ColumnTransformer, sc::Scenarios)
    check_output_columns(ct, sc.names)
    data = copy(sc.data)
    for (c, step) in Iterators.reverse(ct.steps)
        i = findfirst(==(c), sc.names)
        i === nothing || (data[i, :, :] = invert_step(step, data[i, :, :]))
    end
    return Scenarios(data, sc.names, sc.start_year)
end

function check_output_columns(ct::ColumnTransformer, names)
    unknown = setdiff(names, ct.output)
    isempty(unknown) || throw(ArgumentError("columns not produced by the transformer: $(join(unknown, ", "))"))
end
