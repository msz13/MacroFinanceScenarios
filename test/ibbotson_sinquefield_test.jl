using Test
using Dates
using Statistics
using StatsBase: autocor, kurtosis
using TimeSeries
using MacroFinanceScenarios
using MacroFinanceScenarios.IbbotsonSinquefield
using MacroFinanceScenarios.EDA

@testset "IbbotsonSinquefield data" begin
    ta = load_shiller_annual(joinpath(@__DIR__, "..", "data", "ie_data.xlsx"))

    @testset "columns and sample" begin
        @test colnames(ta) == [:π, :rf, :r, :rb, :tp, :re, :xr, :cape, :dlcape, :fr, :erp]
        # Data2 starts in 1934-01, so 1935 is the first full Dec→Dec year; 2026 is partial.
        @test year.(timestamp(ta)) == collect(1935:2025)
        @test all(isfinite, values(ta))
    end

    @testset "identities" begin
        v(name) = values(ta[name])
        @test v(:re) .- v(:rf) ≈ v(:xr)
        @test v(:xr) .- mean(v(:dlcape)) ≈ v(:erp)
        @test v(:re) .- v(:dlcape) ≈ v(:fr)
        @test v(:rf) .- v(:π) ≈ v(:r)
        @test v(:rb) .- v(:rf) ≈ v(:tp)
        @test v(:dlcape)[2:end] ≈ diff(log.(v(:cape)))
    end

    @testset "subperiod" begin
        sub = subperiod(ta, (1975, 2025))
        @test year.(timestamp(sub)) == collect(1975:2025)
        @test values(sub) == values(ta)[year.(timestamp(ta)) .>= 1975, :]
        @test length(subperiod(ta, (1950, 1950))) == 1
    end
end

@testset "EDA tables" begin
    x = [0.1, -0.2, 0.3, 0.05, 0.0, 0.15, -0.1]
    y = [1.0, 2.0, 2.5, 4.0, 3.0, 6.0, 8.0]
    ta = TimeArray(Date.(2000:2006, 12, 31), hcat(x, y), [:x, :y])

    d = describe_series(ta)
    @test d.variable == ["x", "y"]
    @test d.mean ≈ [mean(x), mean(y)]
    @test d.std ≈ [std(x), std(y)]
    @test d.ex_kurtosis ≈ [kurtosis(x), kurtosis(y)]
    @test d.ar1 ≈ [autocor(x, [1])[1], autocor(y, [1])[1]]
    @test d.p50 ≈ [median(x), median(y)]
    @test d.min == [minimum(x), minimum(y)] && d.max == [maximum(x), maximum(y)]

    c = correlation_table(ta)
    @test c.variable == ["x", "y"]
    @test Matrix(c[:, 2:end]) ≈ cor(hcat(x, y))

    @test occursin("ex_kurtosis", sprint(io -> print_table(io, d)))
    @test occursin("<table", sprint(io -> print_table(io, c; backend = :html)))
end
