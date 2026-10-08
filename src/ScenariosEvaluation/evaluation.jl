# Scenario evaluation: per-path moments, mean correlations, horizon percentiles and
# drawdowns of simulated `Scenarios` (n_vars × T × n_scen, log units). Like the EDA
# helpers, every function returns a labelled DataFrame and never prints; use
# `EDA.print_table` for the output.

const DEFAULT_PROBS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Column name for probability p: 0.05 → :p5, 0.025 → :p2.5.
function prob_name(p::Real)
    x = round(100p; digits = 6)
    return Symbol("p", isinteger(x) ? Int(x) : x)
end

# NamedTuple-ready pairs: mean of `x` followed by its quantiles at `probs`.
summary_pairs(x::AbstractVector, probs) =
    [:mean => mean(x); [prob_name(p) => q for (p, q) in zip(probs, quantile(x, probs))]]

"""
    path_moments(sc::Scenarios, name; probs=[0.05, 0.25, 0.5, 0.75, 0.95]) -> DataFrame
    path_moments(sc::Scenarios; probs) -> Vector{Pair{Symbol,DataFrame}}

Moments of variable `name` computed within each simulated path (mean, std, skewness,
excess kurtosis, lag-1 autocorrelation), summarised across scenarios: one row per
statistic, columns `mean` and the percentiles `probs` of that statistic. Without `name`,
one table per variable of `sc`, in order.
"""
function path_moments(sc::Scenarios, name::Symbol; probs = DEFAULT_PROBS)
    X = sc[name]                                        # T × n_scen
    size(X, 1) >= 2 || throw(ArgumentError("path moments need at least 2 simulated years"))
    paths = eachcol(X)
    stats = [:mean => mean.(paths),
             :std => std.(paths),
             :skewness => skewness.(paths),
             :ex_kurtosis => kurtosis.(paths),          # StatsBase.kurtosis is excess kurtosis
             :ar1 => [autocor(x, [1])[1] for x in paths]]
    rows = [(; :statistic => String(s), summary_pairs(v, probs)...) for (s, v) in stats]
    return DataFrame(rows)
end

path_moments(sc::Scenarios; probs = DEFAULT_PROBS) =
    [name => path_moments(sc, name; probs) for name in sc.names]

"""
    mean_correlations(sc::Scenarios) -> DataFrame

Correlation matrix of the variables of `sc` computed within each path (over time), then
averaged across scenarios. Same layout as `EDA.correlation_table`: the first column holds
the row labels.
"""
function mean_correlations(sc::Scenarios)
    n, T, n_scen = size(sc.data)
    T >= 2 || throw(ArgumentError("correlations need at least 2 simulated years"))
    C = zeros(n, n)
    for s in 1:n_scen
        C .+= cor(permutedims(@view sc.data[:, :, s]))
    end
    names = String.(sc.names)
    df = DataFrame(C ./ n_scen, names)
    insertcols!(df, 1, :variable => names)
    return df
end

"""
    horizon_values(X::AbstractMatrix, h; kind=:annualised) -> Vector

Value at horizon `h` of each path (column) of the `T × n_scen` log series `X`:

- `:annualised`: annualised cumulative `sum(x_1..h) / h` (returns, average inflation)
- `:level`: the level in year h, `x_h` (state variables: π, r, rf, tp)
- `:wealth`: cumulative wealth `exp(sum(x_1..h))` (or the price level for π)
"""
function horizon_values(X::AbstractMatrix, h::Integer; kind::Symbol = :annualised)
    1 <= h <= size(X, 1) || throw(ArgumentError("horizon $h outside 1:$(size(X, 1))"))
    kind === :level && return X[h, :]
    S = vec(sum(@view(X[1:h, :]); dims = 1))
    kind === :annualised && return S ./ h
    kind === :wealth && return exp.(S)
    throw(ArgumentError("kind must be :annualised, :level or :wealth, got :$kind"))
end

"""
    horizon_percentiles(sc::Scenarios; vars=sc.names, horizons=[1, 5, 10, 25],
                        probs=[0.05, 0.25, 0.5, 0.75, 0.95], kind=:annualised) -> DataFrame

Distribution across scenarios of each variable in `vars` at each horizon (see
`horizon_values` for `kind`): one row per variable × horizon with columns `variable`,
`horizon`, `year`, `mean` and the percentiles `probs`. Horizons beyond the simulated length
throw.

Which `kind` fits which variable: `:annualised` for asset returns (rf, rb, re), π, erp;
`:level` for the persistent state variables π, r, rf and tp (with `horizons = 1:T` this is
the fan chart data).
"""
function horizon_percentiles(sc::Scenarios; vars = sc.names, horizons = [1, 5, 10, 25],
                             probs = DEFAULT_PROBS, kind::Symbol = :annualised)
    rows = map(Iterators.product(horizons, vars)) do (h, name)
        v = horizon_values(sc[name], h; kind)
        (; :variable => String(name), :horizon => h, :year => sc.start_year + h - 1,
           summary_pairs(v, probs)...)
    end
    return DataFrame(vec(rows))
end

"""
    max_drawdown_and_length(returns::AbstractMatrix) -> (max_drawdowns, max_dd_lengths)

Per path (column) of the `T × n_scen` log returns: the maximum drawdown of wealth
`exp(cumsum(returns))` from its running peak (a fraction, 0 = never below the peak) and the
longest drawdown, the number of consecutive years spent below the previous peak. Wealth
starts at 1 before the first year, so a loss in year 1 already counts as a drawdown.
"""
function max_drawdown_and_length(returns::AbstractMatrix)
    T, n_scen = size(returns)
    max_dd = zeros(n_scen)
    max_len = zeros(Int, n_scen)
    for s in 1:n_scen
        logw, peak, len = 0.0, 0.0, 0          # log wealth and its running peak
        for t in 1:T
            logw += returns[t, s]
            if logw >= peak
                peak, len = logw, 0
            else
                len += 1
                max_dd[s] = max(max_dd[s], 1 - exp(logw - peak))
                max_len[s] = max(max_len[s], len)
            end
        end
    end
    return max_dd, max_len
end

"""
    drawdown_table(sc::Scenarios; vars=sc.names, probs=[0.05, 0.25, 0.5, 0.75, 0.95]) -> DataFrame

Maximum drawdown and longest drawdown length (`max_drawdown_and_length`) per path for each
return variable in `vars`, summarised across scenarios: rows `variable` × `statistic`
(`max_drawdown`, `max_dd_length`), columns `mean` and the percentiles `probs`. Pass nominal
and real returns (`returns_from_components`) separately.
"""
function drawdown_table(sc::Scenarios; vars = sc.names, probs = DEFAULT_PROBS)
    rows = NamedTuple[]
    for name in vars
        dd, len = max_drawdown_and_length(sc[name])
        for (stat, v) in (:max_drawdown => dd, :max_dd_length => float.(len))
            push!(rows, (; :variable => String(name), :statistic => String(stat),
                         summary_pairs(v, probs)...))
        end
    end
    return DataFrame(rows)
end

"""
    annualise(scenarios::AbstractMatrix, shift=2) -> Matrix

Sum non-overlapping blocks of `shift` columns of `scenarios` (rows = scenarios, columns =
periods), e.g. quarterly log returns into annual ones with `shift = 4`. A trailing
incomplete block is dropped.
"""
function annualise(scenarios::AbstractMatrix, shift::Integer = 2)
    n_blocks = size(scenarios, 2) ÷ shift
    result = zeros(size(scenarios, 1), n_blocks)
    for p in 1:n_blocks
        result[:, p] .= vec(sum(@view(scenarios[:, (p-1)*shift+1:p*shift]); dims = 2))
    end
    return result
end

"""
    period_percentiles(X::AbstractMatrix, probs; freq=1) -> DataFrame

Percentiles `probs` across scenarios of each period of `X` (rows = scenarios, columns =
periods), after summing blocks of `freq` periods (`annualise`). One row per aggregated
period. Replaces TCVAR's `print_percentiles`: it returns the table instead of printing it.
"""
function period_percentiles(X::AbstractMatrix, probs; freq::Integer = 1)
    Y = annualise(X, freq)
    rows = [(; :period => p, [prob_name(q) => v for (q, v) in zip(probs, quantile(y, probs))]...)
            for (p, y) in enumerate(eachcol(Y))]
    return DataFrame(rows)
end
