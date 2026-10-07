# Annual (calendar-year, Dec→Dec) building blocks from Shiller's ie_data.xlsx, sheet Data2.
# All series are log returns in decimals. Year t uses December t-1 as the start point, so the
# first full year is one year after the first December in the data (Data2 starts 1934-01,
# hence 1935).

const SHILLER_COLUMNS = ["Date", "P", "D", "CPI", "Rate GS10", "CAPE", "TBILL"]

"""
    load_shiller_annual(path; sheet="Data2", bond_maturity=10) -> TimeArray

Annual series, timestamped at 31 December of each year:

- `π`      inflation, `log(CPI_Dec,t / CPI_Dec,t-1)`
- `rf`     T-bill return, `log(1 + TBILL_Dec,t-1)`: the 3m yield at the start of the year
           rolled forward for the whole year (approximation: ignores the reinvestment at
           the 3m rates during the year)
- `r`      ex-post real short rate, `rf − π`
- `rb`     10y par-bond return from GS10 (`calculate_bond_returns`, annual step), in logs
- `tp`     term premium, `rb − rf`
- `re`     equity total return, `log((P_Dec,t + D_t) / P_Dec,t-1)` with `D_t` the sum of
           monthly D/12 over the year
- `xr`     equity excess return, `re − rf`
- `cape`   CAPE level in December
- `dlcape` `Δlog CAPE`
- `erp`    valuation-corrected equity premium, `xr − dlcape`

Only full calendar years are kept (a trailing partial year is dropped).
"""
function load_shiller_annual(path::AbstractString; sheet::AbstractString = "Data2",
                             bond_maturity::Real = 10)
    raw = XLSX.readxlsx(path)[sheet][:]
    header = string.(raw[1, :])
    col(name) = raw[2:end, findfirst(==(name), header)]

    dates = Date.(string.(col("Date")))
    P, D, CPI, GS10, CAPE, TBILL = (Float64.(col(c)) for c in SHILLER_COLUMNS[2:end])

    # Full calendar years only: those with a December observation.
    dec = findall(d -> month(d) == 12, dates)
    years = year.(dates[dec])
    dividends = [sum(D[year.(dates) .== y]) / 12 for y in years]

    lag(x) = x[1:end-1]
    cur(x) = x[2:end]

    π  = log.(cur(CPI[dec]) ./ lag(CPI[dec]))
    rf = log.(1 .+ lag(TBILL[dec]) ./ 100)
    r  = rf .- π
    rb = log.(1 .+ vec(calculate_bond_returns(GS10[dec] ./ 100, bond_maturity, 1)))
    tp = rb .- rf
    re = log.((cur(P[dec]) .+ cur(dividends)) ./ lag(P[dec]))
    xr = re .- rf
    cape = cur(CAPE[dec])
    dlcape = log.(cape ./ lag(CAPE[dec]))
    erp = xr .- mean(dlcape)

    stamps = Date.(cur(years), 12, 31)
    return TimeArray(stamps, hcat(π, rf, r, rb, tp, re, xr, cape, dlcape, erp),
                     [:π, :rf, :r, :rb, :tp, :re, :xr, :cape, :dlcape, :erp])
end

"""
    subperiod(ta, (y0, y1)) -> TimeArray

Rows of `ta` with timestamps in calendar years `y0` through `y1` (inclusive).
"""
subperiod(ta::TimeArray, (y0, y1)::Tuple{Integer,Integer}) =
    to(from(ta, Date(y0, 1, 1)), Date(y1, 12, 31))
