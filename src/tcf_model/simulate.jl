"""
    simulate.jl

Monte Carlo simulation engine for the Hillenbrand & McCarthy (2026) model.

Usage:
    julia simulate.jl              # defaults: 10_000 paths, 100 quarters (25 years)
    julia simulate.jl 50000 200    # 50,000 paths, 200 quarters (50 years)

Outputs:
    results/simulation_data.jld2   (or falls back to CSV if JLD2 unavailable)
"""

# ── Load model ──
include(joinpath(@__DIR__, "src", "HillenbrandMcCarthyModel.jl"))
using .HillenbrandMcCarthyModel
using Random, Statistics

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

const N_PATHS       = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10_000
const T_QUARTERS    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100    # 25 years
const BASE_SEED     = 20260316   # paper date as seed
const HORIZONS_YRS  = [1, 5, 10, 25]
const HORIZONS_Q    = HORIZONS_YRS .* 4

# ─────────────────────────────────────────────
# Data structures for collecting results
# ─────────────────────────────────────────────

"""
    HorizonSnapshot

Collected values across all Monte Carlo paths at a single horizon.
"""
struct HorizonSnapshot
    horizon_years::Int
    d::Vector{Float64}          # log dividends
    g::Vector{Float64}          # trend growth (quarterly)
    c::Vector{Float64}          # cycle
    μ::Vector{Float64}          # expected return (quarterly)
    pd::Vector{Float64}         # log price-dividend ratio
    cum_log_r::Vector{Float64}  # cumulative log return from t=0
    ann_log_r::Vector{Float64}  # annualised log return
end

"""
    MCResults

Full Monte Carlo output.
"""
struct MCResults
    params::ModelParams
    n_paths::Int
    t_quarters::Int
    snapshots::Dict{Int, HorizonSnapshot}  # keyed by horizon in years
    # Store 3 full paths for plotting
    sample_paths::Vector{SimulationResult}
end

# ─────────────────────────────────────────────
# Run simulation
# ─────────────────────────────────────────────

function run_monte_carlo(;n_paths=N_PATHS, t_quarters=T_QUARTERS,
                          horizons_q=HORIZONS_Q, horizons_yrs=HORIZONS_YRS)

    p = default_params()

    println("═══════════════════════════════════════════════════")
    println("  Hillenbrand & McCarthy (2026) Monte Carlo")
    println("═══════════════════════════════════════════════════")
    println("  Paths:      $n_paths")
    println("  Quarters:   $t_quarters  ($(t_quarters÷4) years)")
    println("  Horizons:   $horizons_yrs years")
    println("  Parameters:")
    println("    ρ_c   = $(p.ρ_c)")
    println("    θ_c   = $(p.θ_c)")
    println("    σ_g   = $(p.σ_g)  ($(round(p.σ_g*100*4*sqrt(4), digits=2)) bps ann.)")
    println("    σ_c   = $(p.σ_c)")
    println("    ρ_μ   = $(p.ρ_μ)  (annual: $(round(p.ρ_μ^4, digits=4)))")
    println("    σ_μ   = $(p.σ_μ)")
    println("    μ̄     = $(p.μ_bar)  ($(round(p.μ_bar*4*100, digits=2))% ann.)")
    println("═══════════════════════════════════════════════════")

    # Pre-allocate horizon collectors
    # Only collect horizons that fit within simulation length
    valid_mask = horizons_q .<= t_quarters
    valid_h_q  = horizons_q[valid_mask]
    valid_h_y  = horizons_yrs[valid_mask]

    collectors = Dict{Int, HorizonSnapshot}()
    for (hq, hy) in zip(valid_h_q, valid_h_y)
        collectors[hy] = HorizonSnapshot(
            hy,
            Vector{Float64}(undef, n_paths),
            Vector{Float64}(undef, n_paths),
            Vector{Float64}(undef, n_paths),
            Vector{Float64}(undef, n_paths),
            Vector{Float64}(undef, n_paths),
            Vector{Float64}(undef, n_paths),
            Vector{Float64}(undef, n_paths)
        )
    end

    # Select 3 sample paths to store fully (for plotting)
    sample_indices = [1, div(n_paths, 2), n_paths]
    sample_paths = Vector{SimulationResult}(undef, 3)

    # ── Main simulation loop ──
    print("  Simulating...")
    t_start = time()

    for i in 1:n_paths
        sim = simulate_path(p, t_quarters; seed=BASE_SEED + i)

        # Collect horizon snapshots
        for (hq, hy) in zip(valid_h_q, valid_h_y)
            vals = extract_at_horizon(sim, hq)
            snap = collectors[hy]
            snap.d[i]          = vals.d
            snap.g[i]          = vals.g
            snap.c[i]          = vals.c
            snap.μ[i]          = vals.μ
            snap.pd[i]         = vals.pd
            snap.cum_log_r[i]  = vals.cum_log_r
            snap.ann_log_r[i]  = annualised_cumulative_return(vals.cum_log_r, Float64(hy))
        end

        # Store sample paths
        idx_in_samples = findfirst(==(i), sample_indices)
        if !isnothing(idx_in_samples)
            sample_paths[idx_in_samples] = sim
        end
    end

    elapsed = round(time() - t_start, digits=2)
    println(" done in $(elapsed)s")

    return MCResults(p, n_paths, t_quarters, collectors, sample_paths)
end

# ─────────────────────────────────────────────
# Save results to CSV files
# ─────────────────────────────────────────────

function save_results(results::MCResults, output_dir::String)
    mkpath(output_dir)

    # Save horizon snapshots
    for (hy, snap) in results.snapshots
        fname = joinpath(output_dir, "horizon_$(hy)yr.csv")
        open(fname, "w") do io
            println(io, "d,g,c,mu,pd,cum_log_r,ann_log_r")
            for i in 1:results.n_paths
                println(io, "$(snap.d[i]),$(snap.g[i]),$(snap.c[i]),$(snap.μ[i]),$(snap.pd[i]),$(snap.cum_log_r[i]),$(snap.ann_log_r[i])")
            end
        end
        println("  Saved: $fname")
    end

    # Save sample paths
    for (k, sim) in enumerate(results.sample_paths)
        fname = joinpath(output_dir, "sample_path_$(k).csv")
        n = sim.T + 1
        open(fname, "w") do io
            println(io, "quarter,d,tau,g,c,mu,pd,cum_log_r")
            for t in 1:n
                q = t - 1
                clr = sim.cum_log_r[t]
                println(io, "$q,$(sim.d[t]),$(sim.τ[t]),$(sim.g[t]),$(sim.c[t]),$(sim.μ[t]),$(sim.pd[t]),$clr")
            end
        end
        println("  Saved: $fname")
    end

    println("  All results saved to: $output_dir")
end

# ─────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────

function main()
    results = run_monte_carlo()

    output_dir = joinpath(@__DIR__, "results")
    println("\nSaving results...")
    save_results(results, output_dir)

    # Print summary statistics
    println("\n═══════════════════════════════════════════════════")
    println("  Summary Statistics")
    println("═══════════════════════════════════════════════════")

    quantiles_list = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    for hy in sort(collect(keys(results.snapshots)))
        snap = results.snapshots[hy]
        println("\n── Horizon: $(hy) years ──")

        println("  Log Dividends (d):")
        println("    Mean:   $(round(mean(snap.d), digits=4))")
        println("    Std:    $(round(std(snap.d), digits=4))")

        println("  Trend Growth (g, quarterly %):")
        println("    Mean:   $(round(mean(snap.g)*100, digits=4))")
        println("    Std:    $(round(std(snap.g)*100, digits=4))")

        println("  Log Price-Dividend Ratio (pd):")
        println("    Mean:   $(round(mean(snap.pd), digits=4))")
        println("    Std:    $(round(std(snap.pd), digits=4))")

        println("  Annualised Log Return (%):")
        q_vals = quantile(snap.ann_log_r, quantiles_list)
        println("    Mean:   $(round(mean(snap.ann_log_r)*100, digits=2))%")
        println("    Std:    $(round(std(snap.ann_log_r)*100, digits=2))%")
        for (q, v) in zip(quantiles_list, q_vals)
            println("    Q$(round(Int, q*100)):    $(round(v*100, digits=2))%")
        end
    end

    return results
end

# Run
results = main()
