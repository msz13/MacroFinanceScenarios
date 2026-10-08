using Test
using Statistics
using StatsBase: autocor, kurtosis, skewness
using MacroFinanceScenarios
using MacroFinanceScenarios.IbbotsonSinquefield: Scenarios
using MacroFinanceScenarios.ScenariosEvaluation
using MacroFinanceScenarios.EDA: print_table

@testset "ScenariosEvaluation" begin
    using Random: Xoshiro
    rng = Xoshiro(11)
    T, n_scen = 30, 200
    data = randn(rng, 3, T, n_scen) .* [0.02, 0.05, 0.17] .+ [0.03, 0.01, 0.07]
    data[2, :, :] .+= 0.5 .* data[1, :, :]              # correlate b with a
    sc = Scenarios(data, [:a, :b, :c], 2026)
    probs = [0.05, 0.25, 0.5, 0.75, 0.95]

    @testset "path_moments" begin
        pm = path_moments(sc, :c)
        @test pm.statistic == ["mean", "std", "skewness", "ex_kurtosis", "ar1"]
        @test propertynames(pm) == [:statistic, :mean, :p5, :p25, :p50, :p75, :p95]
        X = sc[:c]
        means = vec(mean(X; dims = 1))
        @test pm[1, :mean] ≈ mean(means)
        @test collect(pm[1, 3:end]) ≈ quantile(means, probs)
        @test pm[2, :mean] ≈ mean(std.(eachcol(X)))
        @test pm[4, :p50] ≈ median(kurtosis.(eachcol(X)))
        @test pm[5, :mean] ≈ mean(autocor(x, [1])[1] for x in eachcol(X))
        all_pm = path_moments(sc)
        @test first.(all_pm) == [:a, :b, :c]
        @test last(all_pm[3]) == pm
    end

    @testset "mean_correlations" begin
        mc = mean_correlations(sc)
        @test mc.variable == ["a", "b", "c"]
        C = Matrix(mc[:, 2:end])
        @test C ≈ mean(cor(permutedims(data[:, :, s])) for s in 1:n_scen)
        @test all(C[i, i] ≈ 1 for i in 1:3)
        @test C[1, 2] ≈ 0.01 / sqrt(0.05^2 + 0.01^2) atol = 0.03   # b = noise + 0.5a
    end

    @testset "horizon values and percentiles" begin
        X = sc[:a]
        @test horizon_values(X, 5) ≈ vec(sum(X[1:5, :]; dims = 1)) ./ 5
        @test horizon_values(X, 5; kind = :level) == X[5, :]
        @test horizon_values(X, 5; kind = :wealth) ≈ exp.(vec(sum(X[1:5, :]; dims = 1)))
        @test horizon_values(X, 1) ≈ X[1, :]
        @test_throws ArgumentError horizon_values(X, T + 1)
        @test_throws ArgumentError horizon_values(X, 5; kind = :foo)

        hp = horizon_percentiles(sc; vars = [:a, :c], horizons = [1, 10, 25])
        @test hp.variable == ["a", "a", "a", "c", "c", "c"]
        @test hp.horizon == [1, 10, 25, 1, 10, 25]
        @test hp.year == [2026, 2035, 2050, 2026, 2035, 2050]
        @test collect(hp[5, [:p5, :p25, :p50, :p75, :p95]]) ≈ quantile(horizon_values(sc[:c], 10), probs)
        @test hp[5, :mean] ≈ mean(horizon_values(sc[:c], 10))
        # Annualised dispersion shrinks with the horizon for i.i.d. returns.
        @test hp[6, :p95] - hp[6, :p5] < hp[4, :p95] - hp[4, :p5]

        fan = horizon_percentiles(sc; vars = [:a], horizons = 1:T, kind = :level, probs = [0.025, 0.5])
        @test propertynames(fan) == [:variable, :horizon, :year, :mean, Symbol("p2.5"), :p50]
        @test fan.p50 ≈ [median(sc[:a][t, :]) for t in 1:T]
    end

    @testset "drawdowns" begin
        r = log.([1.1 0.9 1.0; 0.8 1.0 1.0; 1.2 1.0 1.0; 1.0 1.2 1.0])  # wealth paths per column
        dd, len = max_drawdown_and_length(r)
        # col 1: 1.1 → 0.88 → 1.056 → 1.056, peak 1.1, under water 3 years
        @test dd[1] ≈ 0.2
        @test len[1] == 3
        # col 2: loss in year 1 counts (wealth starts at 1); 0.9 ×1 ×1 ×1.2 = 1.08 recovers
        @test dd[2] ≈ 0.1
        @test len[2] == 3
        @test dd[3] == 0 && len[3] == 0

        dt = drawdown_table(sc; vars = [:c])
        @test dt.statistic == ["max_drawdown", "max_dd_length"]
        d, l = max_drawdown_and_length(sc[:c])
        @test dt[1, :mean] ≈ mean(d)
        @test dt[2, :p50] ≈ median(l)
        @test all(0 .<= d .< 1)
    end

    @testset "annualise and period_percentiles" begin
        X = reshape(1.0:12.0, 2, 6)                     # rows = scenarios, cols = periods
        @test annualise(X, 2) == [4.0 12.0 20.0; 6.0 14.0 22.0]
        @test annualise(X, 4) == reshape([16.0, 20.0], 2, 1)
        pp = period_percentiles(X, [0.0, 1.0]; freq = 2)
        @test pp.period == [1, 2, 3]
        @test pp.p0 == [4.0, 12.0, 20.0] && pp.p100 == [6.0, 14.0, 22.0]
    end

    @test occursin("ex_kurtosis", sprint(io -> print_table(io, path_moments(sc, :a))))
end
