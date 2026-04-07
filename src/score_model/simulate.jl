"""
    simulate.jl  —  Delle Monache, Petrella & Venditti (2020/2021)

Monte Carlo simulation of dividends and cash flows using the
score-driven state-space model for the price-dividend ratio.

Usage:
    julia simulate.jl              # defaults: 10_000 paths, 100 quarters (25 years)
    julia simulate.jl 50000 200    # 50,000 paths, 200 quarters (50 years)

Outputs (in results/ subdirectory):
    horizon_Xyr.csv     — cross-sectional snapshot at each horizon
    sample_path_k.csv   — 3 full paths for scenario plotting
"""

include(joinpath(@__DIR__, "DelleMonacheModel.jl"))
using .DelleMonacheModel
using Random, Statistics

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

const N_PATHS      = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10_000
const T_QUARTERS   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100    # 25 years
const BASE_SEED    = 20212021      # Delle Monache JBES 2021
const HORIZONS_YRS = [1, 5, 10, 25]
const HORIZONS_Q   = HORIZONS_YRS .* 4

# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

"""
    HorizonSnapshot

Cross-sectional values across all Monte Carlo paths at a single horizon.
Field `c` stores the transitory growth component g̃_t (analogous to the
cash-flow cycle in Hillenbrand & McCarthy).
"""
struct HorizonSnapshot
    horizon_years::Int
    d::Vector{Float64}          # log dividends
    g::Vector{Float64}          # total expected growth g_t = ḡ_t + g̃_t (quarterly)
    c::Vector{Float64}          # transitory growth g̃_t (cycle-like component)
    μ::Vector{Float64}          # total expected return μ_t = μ̄_t + μ̃_t (quarterly)
    pd::Vector{Float64}         # log price-dividend ratio
    cum_log_r::Vector{Float64}  # cumulative log return from t=0
    ann_log_r::Vector{Float64}  # annualised log return
end

struct MCResults
    params::ModelParams
    n_paths::Int
    t_quarters::Int
    snapshots::Dict{Int, HorizonSnapshot}
    sample_paths::Vector{SimulationResult}   # 3 full paths for plotting
end

# ─────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────

function run_monte_carlo(; n_paths    = N_PATHS,
                           t_quarters = T_QUARTERS,
                           horizons_q = HORIZONS_Q,
                           horizons_yrs = HORIZONS_YRS)

    p  = default_params()
    κ  = calibrate_kappa(p)   # shown in parameter summary below

    println("═══════════════════════════════════════════════════════")
    println("  Delle Monache, Petrella & Venditti (2021) Monte Carlo")
    println("═══════════════════════════════════════════════════════")
    println("  Paths:      $n_paths")
    println("  Quarters:   $t_quarters  ($(t_quarters÷4) years)")
    println("  Horizons:   $horizons_yrs years")
    println()
    println("  Parameters:")
    println("    ρ       = $(p.ρ)       (Campbell-Shiller constant)")
    println("    ρ_μ     = $(p.ρ_μ)      (transitory ER persistence, quarterly)")
    println("    ḡ₀      = $(p.ḡ₀)    (permanent growth, $(round(p.ḡ₀*400,digits=2))% pa)")
    println("    μ̄₀      = $(p.μ̄₀)     (permanent ER, $(round(p.μ̄₀*400,digits=2))% pa)")
    println("    σ_d     = $(p.σ_d)      (dividend shock std, quarterly)")
    println("    σ_g     = $(p.σ_g)    (growth shock std, quarterly)")
    println("    σ_μ     = $(p.σ_μ)      (ER shock std, quarterly)")
    println("    ρ_dμ    = $(p.ρ_dμ)      (dividend–return shock correlation)")
    println("    ρ_gμ    = $(p.ρ_gμ)       (growth–return shock correlation)")
    println("    κ       = $(round(κ, digits=4))     (Campbell-Shiller constant, calibrated)")
    println("    pd_ss   = $(p.pd_ss)        (steady-state log P/D)")
    println("═══════════════════════════════════════════════════════")

    # Only include horizons that fit within simulation length
    valid_mask  = horizons_q .<= t_quarters
    valid_h_q   = horizons_q[valid_mask]
    valid_h_y   = horizons_yrs[valid_mask]

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

    # Three full paths: first, median-index, last
    sample_indices = [1, div(n_paths, 2), n_paths]
    sample_paths   = Vector{SimulationResult}(undef, 3)

    print("  Simulating...")
    t_start = time()

    for i in 1:n_paths
        sim = simulate_path(p, t_quarters; seed = BASE_SEED + i)

        for (hq, hy) in zip(valid_h_q, valid_h_y)
            vals = extract_at_horizon(sim, hq)
            snap = collectors[hy]
            snap.d[i]         = vals.d
            snap.g[i]         = vals.g
            snap.c[i]         = vals.g̃
            snap.μ[i]         = vals.μ
            snap.pd[i]        = vals.pd
            snap.cum_log_r[i] = vals.cum_log_r
            snap.ann_log_r[i] = annualised_cumulative_return(vals.cum_log_r, Float64(hy))
        end

        idx_in_samples = findfirst(==(i), sample_indices)
        if !isnothing(idx_in_samples)
            sample_paths[idx_in_samples] = sim
        end
    end

    elapsed = round(time() - t_start, digits = 2)
    println(" done in $(elapsed)s")

    return MCResults(p, n_paths, t_quarters, collectors, sample_paths)
end

# ─────────────────────────────────────────────
# Save Results
# ─────────────────────────────────────────────

"""
    save_results(results, output_dir)

Write horizon snapshots and sample paths to CSV files readable by analyse.jl.

Horizon CSV columns:  d, g, c, mu, pd, cum_log_r, ann_log_r
Sample path columns:  quarter, d, tau, g, c, mu, pd, cum_log_r

Variable mapping for the Delle Monache model:
  g   → total expected growth g_t = ḡ_t + g̃_t  (quarterly)
  c   → transitory growth g̃_t  (cycle-like component)
  mu  → total expected return μ_t = μ̄_t + μ̃_t  (quarterly)
  tau → permanent growth ḡ_t  (trend-like component)
"""
function save_results(results::MCResults, output_dir::String)
    mkpath(output_dir)

    # ── Horizon snapshots ──
    for (hy, snap) in results.snapshots
        fname = joinpath(output_dir, "horizon_$(hy)yr.csv")
        open(fname, "w") do io
            println(io, "d,g,c,mu,pd,cum_log_r,ann_log_r")
            for i in 1:results.n_paths
                println(io,
                    "$(snap.d[i]),$(snap.g[i]),$(snap.c[i]),$(snap.μ[i])," *
                    "$(snap.pd[i]),$(snap.cum_log_r[i]),$(snap.ann_log_r[i])")
            end
        end
        println("  Saved: $fname")
    end

    # ── Sample paths ──
    for (k, sim) in enumerate(results.sample_paths)
        fname = joinpath(output_dir, "sample_path_$(k).csv")
        n     = sim.T + 1
        open(fname, "w") do io
            println(io, "quarter,d,tau,g,c,mu,pd,cum_log_r")
            for t in 1:n
                q = t - 1
                println(io,
                    "$q,$(sim.d[t]),$(sim.ḡ[t]),$(sim.g[t]),$(sim.g̃[t])," *
                    "$(sim.μ[t]),$(sim.pd[t]),$(sim.cum_log_r[t])")
            end
        end
        println("  Saved: $fname")
    end

    println("  All results saved to: $output_dir")
end

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

function main()
    results = run_monte_carlo()

    output_dir = joinpath(@__DIR__, "results")
    println("\nSaving results...")
    save_results(results, output_dir)

    # ── Quick summary ──
    println("\n═══════════════════════════════════════════════════════")
    println("  Summary Statistics")
    println("═══════════════════════════════════════════════════════")

    quantiles_list = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    for hy in sort(collect(keys(results.snapshots)))
        snap = results.snapshots[hy]
        println("\n── Horizon: $(hy) years ──")

        println("  Log Dividends (d):")
        println("    Mean: $(round(mean(snap.d), digits=4))   " *
                "Std: $(round(std(snap.d), digits=4))")

        println("  Total Expected Growth (g, annualised %):")
        println("    Mean: $(round(mean(snap.g)*400, digits=2))%   " *
                "Std: $(round(std(snap.g)*400, digits=2))%")

        println("  Log P/D Ratio (pd):")
        println("    Mean: $(round(mean(snap.pd), digits=4))   " *
                "Std: $(round(std(snap.pd), digits=4))")

        println("  Annualised Log Return:")
        q_vals = quantile(snap.ann_log_r, quantiles_list)
        println("    Mean: $(round(mean(snap.ann_log_r)*100, digits=2))%   " *
                "Std: $(round(std(snap.ann_log_r)*100, digits=2))%")
        for (q, v) in zip(quantiles_list, q_vals)
            @printf("    Q%-3s  %+.2f%%\n", "$(round(Int, q*100))%", v*100)
        end
    end

    return results
end

using Printf
results = main()
