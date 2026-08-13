#!/usr/bin/env julia
# =============================================================================
#  Scenario simulation:  AR(1) model  vs.  Trend–Cycle model
#
#    Model A (AR):    y_t = c + φ y_{t-1} + u_t,        u_t ~ N(0, σ²)
#
#    Model B (UC):    y_t = μ_t + c_t
#                     μ_t = μ_{t-1} + η_t,              η_t ~ N(0, ση²)   trend
#                     c_t = ψ c_{t-1} + ε_t,            ε_t ~ N(0, σε²)   cycle
#
#  All parameters and starting values are inputs (PARAMETERS block below).
#  Both models simulate forward scenarios; predictive quantiles are compared
#  at horizon h = 1 and h = 25.
#
#  Pure Base/stdlib Julia — no external packages.
#  Run with:  julia ar1_vs_trend_cycle.jl
# =============================================================================

using Random, Statistics, Printf

# =============================================================================
#  PARAMETERS  (edit these — this is the entire input of the script)
# =============================================================================

const SEED = 20260811
const H    = 100          # forecast horizon
const NSIM = 100_000      # number of simulated scenarios
const QLEV = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# --- Model A: AR(1) ----------------------------------------------------------
ar = (c  = 2.5,      # intercept
      φ  = 0.7,      # AR coefficient          (|φ| < 1 for stationarity)
      σ  = 0.81,      # sd of the innovation
      y0 = 2.5)     # starting value y_T

# --- Model B: random-walk trend + AR(1) cycle --------------------------------
uc = (ψ  = 0.7,      # cycle AR coefficient    (|ψ| < 1 for stationarity)
      ση = .1,      # sd of the trend shock
      σε = .8,      # sd of the cycle shock
      μ0 = 2.5,     # starting trend level μ_T
      c0 = 0.0)      # starting cycle value c_T
                      # => starting level y_T = μ0 + c0

                     
# =============================================================================
#  SIMULATORS   (each returns an NSIM × H matrix of scenario paths)
# =============================================================================

function simulate_ar(p, H::Int, nsim::Int)
    paths = zeros(nsim, H)
    for s in 1:nsim
        y = p.y0
        for h in 1:H
            y = p.c + p.φ * (y - p.c) + p.σ * randn()
            paths[s, h] = y
        end
    end
    return paths
end

function simulate_uc(p, H::Int, nsim::Int)
    paths = zeros(nsim, H); trend = zeros(nsim, H); cycle = zeros(nsim, H)
    for s in 1:nsim
        μ = p.μ0; c = p.c0
        for h in 1:H
            μ = μ + p.ση * randn()          # random-walk trend
            c = p.ψ * c + p.σε * randn()    # AR(1) cycle
            trend[s, h] = μ; cycle[s, h] = c; paths[s, h] = μ + c
        end
    end
    return paths, trend, cycle
end

# =============================================================================
#  REPORTING
# =============================================================================

qrow(v) = [quantile(v, q) for q in QLEV]

function print_quantile_block(h, ar_col, uc_col)
    a, u = qrow(ar_col), qrow(uc_col)
    @printf("\n  Horizon h = %d\n", h)
    println("  ", "-"^74)
    @printf("  %-10s", "quantile")
    for q in QLEV; @printf("%9.0f%%", 100q); end
    println()
    println("  ", "-"^74)
    @printf("  %-10s", "AR(1)")
    for x in a; @printf("%10.2f", x); end; println()
    @printf("  %-10s", "Trend-cyc")
    for x in u; @printf("%10.2f", x); end; println()
    @printf("  %-10s", "diff")
    for i in eachindex(a); @printf("%10.2f", u[i] - a[i]); end; println()
    println("  ", "-"^74)
    wa, wu = a[6] - a[2], u[6] - u[2]        # 10th–90th percentile
    @printf("  80%% interval width:  AR = %.2f   trend-cycle = %.2f   ratio = %.2f\n",
            wa, wu, wu / wa)
    @printf("  median:              AR = %.2f   trend-cycle = %.2f   diff  = %+.2f\n",
            a[4], u[4], u[4] - a[4])
end

# =============================================================================
#  MAIN
# =============================================================================

Random.seed!(SEED)

println("="^78)
println(" SCENARIO SIMULATION — $NSIM paths, horizon $H, seed $SEED")
println("="^78)

@printf("\nModel A — AR(1):  y_t = %.2f + %.2f y_{t-1} + u_t,  sd(u) = %.2f\n",
        ar.c, ar.φ, ar.σ)
@printf("  start y_T = %.2f\n", ar.y0)
@printf("  unconditional mean = %.2f,  long-run sd = %.2f\n",
        ar.c / (1 - ar.φ), ar.σ / sqrt(1 - ar.φ^2))

@printf("\nModel B — trend + cycle:  y_t = mu_t + c_t\n")
@printf("  mu_t = mu_{t-1} + eta_t,   sd(eta) = %.2f\n", uc.ση)
@printf("  c_t  = %.2f c_{t-1} + eps_t, sd(eps) = %.2f\n", uc.ψ, uc.σε)
@printf("  start mu_T = %.2f, c_T = %.2f  =>  y_T = %.2f\n",
        uc.μ0, uc.c0, uc.μ0 + uc.c0)
@printf("  cycle long-run sd = %.2f (bounded); trend variance grows as h * %.2f\n",
        uc.σε / sqrt(1 - uc.ψ^2), uc.ση^2)

paths_ar = simulate_ar(ar, H, NSIM)
paths_uc, trend, cycle = simulate_uc(uc, H, NSIM)

println("\n", "="^78)
println(" PREDICTIVE QUANTILES")
println("="^78)
print_quantile_block(1,  paths_ar[:, 1],  paths_uc[:, 1])
print_quantile_block(100, paths_ar[:, 100], paths_uc[:, 100])

println("\n  Fan width (10th–90th pct) across horizons")
println("  ", "-"^46)
@printf("  %6s %12s %12s %10s\n", "h", "AR(1)", "trend-cyc", "ratio")
#for h in (1, 2, 5, 10, 15, 20, 25)
for h in (1, 4, 20, 40, 100)
    wa = quantile(paths_ar[:, h], 0.9) - quantile(paths_ar[:, h], 0.1)
    wu = quantile(paths_uc[:, h], 0.9) - quantile(paths_uc[:, h], 0.1)
    @printf("  %6d %12.2f %12.2f %10.2f\n", h, wa, wu, wu / wa)
end



@printf("\n  AR(1) fan converges to a finite band (long-run 80%% width ≈ %.2f);\n",
        2 * 1.2816 * ar.σ / sqrt(1 - ar.φ^2))
println("  the trend–cycle fan keeps widening because of the random-walk trend.")

open("quantile_fans.csv", "w") do io
    println(io, "model,h," * join(["q" * string(Int(100q)) for q in QLEV], ","))
    for h in 1:H
        println(io, "AR,",       h, ",", join(round.(qrow(paths_ar[:, h]), digits = 4), ","))
        println(io, "TrendCyc,", h, ",", join(round.(qrow(paths_uc[:, h]), digits = 4), ","))
    end
end
println("\nFull quantile fans for h = 1..$H written to quantile_fans.csv")
