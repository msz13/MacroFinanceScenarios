"""
    analyse.jl  —  Delle Monache, Petrella & Venditti (2021)

Reads Monte Carlo simulation output and produces:
  1. Quantile distribution tables for d, pd, annualised returns at 1/5/10/25yr horizons
  2. Plots of 3 random scenarios for each state variable
  3. Distribution plots (histograms) of annualised returns at each horizon

Adapted from src/tcf_model/analyse.jl for the score-driven state-space model.
Requires: UnicodePlots (lightweight, no system dependencies)

Usage:
    julia analyse.jl
"""

using Statistics, Printf, DelimitedFiles

# ─────────────────────────────────────────────
# 1. Load simulation results from CSV
# ─────────────────────────────────────────────

const RESULTS_DIR  = joinpath(@__DIR__, "results")
const HORIZONS_YRS = [1, 5, 10, 25]

struct HorizonData
    horizon_years::Int
    d::Vector{Float64}          # log dividends
    g::Vector{Float64}          # total expected growth g_t (quarterly)
    c::Vector{Float64}          # transitory growth g̃_t
    μ::Vector{Float64}          # total expected return μ_t (quarterly)
    pd::Vector{Float64}         # log price-dividend ratio
    cum_log_r::Vector{Float64}  # cumulative log return
    ann_log_r::Vector{Float64}  # annualised log return
end

struct SamplePath
    quarter::Vector{Float64}
    d::Vector{Float64}
    τ::Vector{Float64}          # permanent growth ḡ_t (stored as τ)
    g::Vector{Float64}          # total expected growth g_t
    c::Vector{Float64}          # transitory growth g̃_t
    μ::Vector{Float64}          # total expected return μ_t
    pd::Vector{Float64}
    cum_log_r::Vector{Float64}
end

function load_horizon(hy::Int)
    fname = joinpath(RESULTS_DIR, "horizon_$(hy)yr.csv")
    if !isfile(fname)
        @warn "File not found: $fname — skipping horizon $hy yr"
        return nothing
    end
    data = readdlm(fname, ',', Float64; skipstart=1)
    return HorizonData(hy,
        data[:,1], data[:,2], data[:,3], data[:,4],
        data[:,5], data[:,6], data[:,7])
end

function load_sample_path(k::Int)
    fname = joinpath(RESULTS_DIR, "sample_path_$(k).csv")
    data  = readdlm(fname, ',', Float64; skipstart=1)
    return SamplePath(
        data[:,1], data[:,2], data[:,3], data[:,4],
        data[:,5], data[:,6], data[:,7], data[:,8])
end

# ─────────────────────────────────────────────
# 2. Quantile Analysis
# ─────────────────────────────────────────────

const QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

function quantile_table(label::String, values::Vector{Float64};
                        scale::Float64=1.0, unit::String="")
    q_vals = quantile(values, QUANTILES)
    println("  ┌─────────────────────────────────────────────────────┐")
    @printf("  │  %-48s │\n", label)
    println("  ├──────────┬──────────────────────────────────────────┤")
    @printf("  │  Mean    │  %10.4f %-5s                         │\n", mean(values)*scale, unit)
    @printf("  │  Std     │  %10.4f %-5s                         │\n", std(values)*scale, unit)
    @printf("  │  Skew    │  %10.4f                               │\n", skewness_simple(values))
    println("  ├──────────┼──────────────────────────────────────────┤")
    for (q, v) in zip(QUANTILES, q_vals)
        @printf("  │  Q%-6s │  %10.4f %-5s                         │\n",
                "$(round(Int, q*100))%", v*scale, unit)
    end
    println("  └──────────┴──────────────────────────────────────────┘")
end

function skewness_simple(x)
    m = mean(x)
    s = std(x)
    n = length(x)
    return sum(((xi - m)/s)^3 for xi in x) / n
end

function analyse_horizon(hd::HorizonData)
    hy = hd.horizon_years
    println("\n" * "="^60)
    println("  HORIZON: $hy YEARS ($(hy*4) quarters)")
    println("="^60)

    quantile_table("Log Dividends (d)", hd.d)
    quantile_table("Expected Growth (g_t, quarterly)", hd.g; scale=100.0, unit="%")
    quantile_table("Transitory Growth (g̃_t, quarterly)", hd.c; scale=100.0, unit="%")
    quantile_table("Log Price-Dividend Ratio (pd)", hd.pd)
    quantile_table("Cumulative Log Return", hd.cum_log_r; scale=100.0, unit="%")
    quantile_table("Annualised Log Return", hd.ann_log_r; scale=100.0, unit="%")

    gross_ann = exp.(hd.ann_log_r) .- 1.0
    quantile_table("Annualised Gross Return", gross_ann; scale=100.0, unit="%")
end

# ─────────────────────────────────────────────
# 3. Text-Based Plotting (UnicodePlots or fallback)
# ─────────────────────────────────────────────

let
    global HAS_UNICODE_PLOTS
    try
        @eval using UnicodePlots
        HAS_UNICODE_PLOTS = true
    catch
        HAS_UNICODE_PLOTS = false
    end
end

function ascii_histogram(values::Vector{Float64}, title::String;
                         nbins::Int=40, width::Int=50)
    lo, hi = minimum(values), maximum(values)
    if lo == hi
        println("  $title: all values = $lo")
        return
    end
    bin_edges = range(lo, hi, length=nbins+1)
    counts    = zeros(Int, nbins)
    for v in values
        idx = clamp(searchsortedlast(collect(bin_edges), v), 1, nbins)
        counts[idx] += 1
    end
    max_count = maximum(counts)

    println("\n  ┌─ $title ─" * "─"^max(0, width-length(title)-5) * "┐")
    for i in 1:nbins
        bar_len = round(Int, counts[i] / max_count * width)
        label   = @sprintf("%8.3f", (bin_edges[i] + bin_edges[i+1]) / 2)
        if i % 4 == 1 || i == nbins
            println("  │$label │" * "█"^bar_len)
        end
    end
    println("  └─" * "─"^(width+10) * "┘")
end

function plot_histogram(values::Vector{Float64}, title::String)
    if HAS_UNICODE_PLOTS
        plt = UnicodePlots.histogram(values, title=title, nbins=50, width=60)
        display(plt)
        println()
    else
        ascii_histogram(values, title)
    end
end

function plot_line(x::Vector{Float64}, ys::Vector{Vector{Float64}},
                   labels::Vector{String}, title::String)
    if HAS_UNICODE_PLOTS
        plt = UnicodePlots.lineplot(x, ys[1], title=title, name=labels[1],
                                     width=70, height=15)
        for i in 2:length(ys)
            UnicodePlots.lineplot!(plt, x, ys[i], name=labels[i])
        end
        display(plt)
        println()
    else
        println("\n  ── $title ──")
        for (i, lab) in enumerate(labels)
            y = ys[i]
            @printf("    %s: start=%.4f  end=%.4f  min=%.4f  max=%.4f\n",
                    lab, y[1], y[end], minimum(y), maximum(y))
        end
    end
end

# ─────────────────────────────────────────────
# 4. Scenario Plots (3 paths per variable)
# ─────────────────────────────────────────────

function plot_sample_paths(paths::Vector{SamplePath})
    n      = length(paths)
    labels = ["Path $i" for i in 1:n]
    x      = paths[1].quarter ./ 4.0   # time axis in years

    println("\n" * "="^60)
    println("  SAMPLE PATH PLOTS (3 scenarios)")
    println("="^60)

    plot_line(x, [p.d for p in paths], labels,
              "Log Dividends (d_t)")

    plot_line(x, [p.g .* 400.0 for p in paths], labels,
              "Total Expected Growth (g_t, annualised %)")

    plot_line(x, [p.c .* 400.0 for p in paths], labels,
              "Transitory Growth (g̃_t, annualised %)")

    plot_line(x, [p.μ .* 400.0 for p in paths], labels,
              "Total Expected Return (μ_t, annualised %)")

    plot_line(x, [p.pd for p in paths], labels,
              "Log Price-Dividend Ratio (pd_t)")

    plot_line(x, [p.cum_log_r .* 100.0 for p in paths], labels,
              "Cumulative Log Return (%)")
end

# ─────────────────────────────────────────────
# 5. Distribution Plots
# ─────────────────────────────────────────────

function plot_distributions(horizons::Dict{Int, HorizonData})
    println("\n" * "="^60)
    println("  DISTRIBUTION PLOTS")
    println("="^60)

    for hy in sort(collect(keys(horizons)))
        hd = horizons[hy]

        plot_histogram(hd.ann_log_r .* 100.0,
                       "Annualised Log Return (%) — $(hy)-year horizon")

        plot_histogram(hd.pd,
                       "Log Price-Dividend Ratio — $(hy)-year horizon")

        plot_histogram(hd.d,
                       "Log Dividends — $(hy)-year horizon")
    end
end

# ─────────────────────────────────────────────
# 6. Cross-Horizon Comparison Table
# ─────────────────────────────────────────────

function print_comparison_table(horizons::Dict{Int, HorizonData})
    println("\n" * "="^90)
    println("  CROSS-HORIZON COMPARISON")
    println("="^90)

    function print_row_header()
        @printf("  %-10s", "Horizon")
        for q in QUANTILES
            @printf("  Q%-5s", "$(round(Int, q*100))%")
        end
        @printf("  %8s  %8s\n", "Mean", "Std")
        println("  " * "─"^88)
    end

    # Annualised returns
    println("\n  Annualised Log Returns (%)")
    print_row_header()
    for hy in sort(collect(keys(horizons)))
        hd     = horizons[hy]
        q_vals = quantile(hd.ann_log_r, QUANTILES) .* 100
        @printf("  %-10s", "$(hy) yr")
        for v in q_vals; @printf("  %6.2f", v); end
        @printf("  %8.2f  %8.2f\n", mean(hd.ann_log_r)*100, std(hd.ann_log_r)*100)
    end

    # Price-Dividend ratio
    println("\n  Log Price-Dividend Ratio")
    print_row_header()
    for hy in sort(collect(keys(horizons)))
        hd     = horizons[hy]
        q_vals = quantile(hd.pd, QUANTILES)
        @printf("  %-10s", "$(hy) yr")
        for v in q_vals; @printf("  %6.2f", v); end
        @printf("  %8.2f  %8.2f\n", mean(hd.pd), std(hd.pd))
    end

    # Log Dividends
    println("\n  Log Dividends")
    print_row_header()
    for hy in sort(collect(keys(horizons)))
        hd     = horizons[hy]
        q_vals = quantile(hd.d, QUANTILES)
        @printf("  %-10s", "$(hy) yr")
        for v in q_vals; @printf("  %6.2f", v); end
        @printf("  %8.2f  %8.2f\n", mean(hd.d), std(hd.d))
    end

    # Total expected growth (annualised)
    println("\n  Total Expected Growth (annualised %)")
    print_row_header()
    for hy in sort(collect(keys(horizons)))
        hd     = horizons[hy]
        ann_g  = hd.g .* 400.0
        q_vals = quantile(ann_g, QUANTILES)
        @printf("  %-10s", "$(hy) yr")
        for v in q_vals; @printf("  %6.2f", v); end
        @printf("  %8.2f  %8.2f\n", mean(ann_g), std(ann_g))
    end
end

# ─────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────

function main()
    println("═══════════════════════════════════════════════════")
    println("  Delle Monache, Petrella & Venditti (2021)")
    println("  Score-Driven State Space Model — MC Analysis")
    println("═══════════════════════════════════════════════════")

    horizons = Dict{Int, HorizonData}()
    for hy in HORIZONS_YRS
        hd = load_horizon(hy)
        if !isnothing(hd)
            horizons[hy] = hd
        end
    end

    if isempty(horizons)
        error("No horizon data found in $RESULTS_DIR. Run simulate.jl first.")
    end

    sample_paths = SamplePath[]
    for k in 1:3
        fname = joinpath(RESULTS_DIR, "sample_path_$(k).csv")
        if isfile(fname)
            push!(sample_paths, load_sample_path(k))
        end
    end

    for hy in sort(collect(keys(horizons)))
        analyse_horizon(horizons[hy])
    end

    print_comparison_table(horizons)

    if !isempty(sample_paths)
        plot_sample_paths(sample_paths)
    end

    plot_distributions(horizons)

    println("\n  Analysis complete.")
end

main()
