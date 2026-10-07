# EDA helpers: functions return labelled DataFrames and never print; `print_table` does
# the printing, so the same tables work in the REPL (:text), notebooks/Quarto (:html)
# and tests.

"""
    describe_series(ta::TimeArray) -> DataFrame

One row per column of `ta` with mean, std, skewness, excess kurtosis, lag-1
autocorrelation, min, p25, p50, p75 and max.
"""
function describe_series(ta::TimeArray)
    rows = map(colnames(ta)) do name
        x = values(ta[name])
        q = quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0])
        (variable = String(name), mean = mean(x), std = std(x),
         skewness = skewness(x), ex_kurtosis = kurtosis(x),   # StatsBase.kurtosis is excess kurtosis
         ar1 = autocor(x, [1])[1],
         min = q[1], p25 = q[2], p50 = q[3], p75 = q[4], max = q[5])
    end
    return DataFrame(rows)
end

"""
    correlation_table(ta::TimeArray) -> DataFrame

Correlation matrix of the columns of `ta`; the first column holds the row labels.
"""
function correlation_table(ta::TimeArray)
    names = String.(colnames(ta))
    C = cor(values(ta))
    df = DataFrame(C, names)
    insertcols!(df, 1, :variable => names)
    return df
end

"""
    print_table([io::IO,] t::DataFrame; backend=:text, digits=4, kwargs...)

Print a table returned by the EDA/evaluation functions, rounding floats to `digits`.
`backend` is passed to PrettyTables 3 (`:text`, `:html`, `:markdown`, ...).
"""
function print_table(io::IO, t::DataFrame; backend::Symbol = :text, digits::Int = 4, kwargs...)
    pretty_table(io, t; backend, formatters = [fmt__round(digits)], kwargs...)
end

print_table(t::DataFrame; kwargs...) = print_table(stdout, t; kwargs...)
