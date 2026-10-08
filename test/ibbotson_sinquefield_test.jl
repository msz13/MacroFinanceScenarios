using Test
using Dates
using Statistics
using LinearAlgebra: diag
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
        # rf + tp is the 10y yield at the start of the year; rb is the par-bond return on it.
        y10 = exp.(v(:rf) .+ v(:tp)) .- 1
        @test v(:rb)[1:end-1] ≈ log.(1 .+ vec(IbbotsonSinquefield.calculate_bond_returns(y10, 10, 1)))
        @test v(:dlcape)[2:end] ≈ diff(log.(v(:cape)))
    end

    @testset "subperiod" begin
        sub = subperiod(ta, (1975, 2025))
        @test year.(timestamp(sub)) == collect(1975:2025)
        @test values(sub) == values(ta)[year.(timestamp(ta)) .>= 1975, :]
        @test length(subperiod(ta, (1950, 1950))) == 1
    end
end

@testset "transform_columns" begin
    stamps = Date.(2000:2004, 12, 31)
    ta = TimeArray(stamps, [1.0 10.0 100.0; 2.0 20.0 200.0; 3.0 30.0 300.0;
                            4.0 40.0 400.0; 5.0 50.0 500.0], [:a, :b, :c])
    v(t, n) = values(t[n])
    demean(x) = x .- mean(x)

    @testset "pipeline on selected columns, remainder passes through" begin
        out = transform_columns(ta, [:a, :b] => [x -> 2 .* x, demean])
        @test colnames(out) == [:a, :b, :c]
        @test timestamp(out) == stamps
        @test v(out, :a) ≈ 2 .* v(ta, :a) .- 6
        @test v(out, :b) ≈ 2 .* v(ta, :b) .- 60
        @test v(out, :c) == v(ta, :c)
    end

    @testset "Except, single function and spec order" begin
        out = transform_columns(ta, :c => x -> log.(x), Except(:c) => demean, :a => x -> x ./ 2)
        @test v(out, :c) ≈ log.(v(ta, :c))
        @test v(out, :b) ≈ v(ta, :b) .- 30
        @test v(out, :a) ≈ (v(ta, :a) .- 3) ./ 2   # :a got both specs, in order
        @test values(transform_columns(ta, Except() => demean)) ≈ values(ta) .- mean(values(ta); dims = 1)
    end

    @testset "remainder = :drop" begin
        out = transform_columns(ta, :c => demean, :a => identity; remainder = :drop)
        @test colnames(out) == [:a, :c]   # original order
    end

    @testset "errors" begin
        @test_throws ArgumentError transform_columns(ta, :z => identity)
        @test_throws ArgumentError transform_columns(ta, :a => identity; remainder = :foo)
        @test_throws DimensionMismatch transform_columns(ta, :a => diff)
    end

    @testset "fitted steps and inverse" begin
        ct = fit_columns(ta, :c => log => exp, Except(:c) => [Affine(100, 0), Standardize()],
                         :b => Demean())
        z = transform_columns(ct, ta)
        @test v(z, :c) ≈ log.(v(ta, :c))
        @test mean(v(z, :a)) ≈ 0 atol = 1e-12
        @test std(v(z, :a)) ≈ 1
        @test values(inverse_transform_columns(ct, z)) ≈ values(ta)
        # Fitted means and scales are frozen: a new sample is transformed with them.
        new = TimeArray(stamps, values(ta) .+ 1, [:a, :b, :c])
        @test v(transform_columns(ct, new), :a) ≈ (100 .* (v(ta, :a) .+ 1) .- 300) ./ (100 * std(v(ta, :a)))
        # Any subset of the columns, in any order, can be inverted.
        @test values(inverse_transform_columns(ct, z[:c, :a])) ≈ values(ta[:c, :a])
    end

    @testset "inverse on Scenarios" begin
        ct = fit_columns(ta, :c => log => exp, [:a, :b] => Standardize())
        z = values(transform_columns(ct, ta))
        paths = cat(permutedims(z), 2 .* permutedims(z); dims = 3)   # 3 vars × 5 years × 2 scen
        sc = inverse_transform_columns(ct, Scenarios(paths, [:a, :b, :c], 2005))
        @test sc.names == [:a, :b, :c]
        @test sc.start_year == 2005
        @test sc[:a][:, 1] ≈ v(ta, :a)
        @test sc[:a][:, 2] ≈ mean(v(ta, :a)) .+ 2 .* std(v(ta, :a)) .* z[:, 1]
        @test sc[:c][:, 2] ≈ v(ta, :c) .^ 2
    end

    @testset "inverse errors" begin
        ct = fit_columns(ta, :a => x -> x .- mean(x); remainder = :drop)
        @test_throws ArgumentError inverse_transform_columns(ct, transform_columns(ct, ta))
        @test_throws ArgumentError inverse_transform_columns(ct, ta)   # :b, :c were dropped
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

@testset "IbbotsonSinquefield AR(1)" begin
    using Random: Xoshiro

    @testset "recovers c and φ on simulated data" begin
        rng = Xoshiro(42)
        c, φ, σ, n = 0.01, 0.6, 0.02, 20_000
        x = zeros(n)
        x[1] = c / (1 - φ)
        for t in 2:n
            x[t] = c + φ * x[t-1] + σ * randn(rng)
        end
        ar = fit_ar1(x)
        @test ar.c ≈ c atol = 3 * ar.se[1]
        @test ar.φ ≈ φ atol = 3 * ar.se[2]
        @test ar.σ ≈ σ rtol = 0.02
        @test unconditional_mean(ar) ≈ c / (1 - φ) rtol = 0.05
        @test unconditional_std(ar) ≈ σ / sqrt(1 - φ^2) rtol = 0.05
        @test ar.period == (1, n)
    end

    @testset "fit structure and TimeArray method" begin
        x = [0.03, 0.05, 0.02, 0.04, 0.06, 0.01, 0.03, 0.05]
        ta = TimeArray(Date.(2000:2007, 12, 31), hcat(x, reverse(x)), [:a, :b])
        ar = fit_ar1(ta, :a)
        @test ar.period == (2000, 2007)
        @test ar.years == collect(2001:2007)
        @test ar.fitted .+ ar.resid ≈ x[2:end]
        @test ar.fitted ≈ ar.c .+ ar.φ .* x[1:end-1]
        @test abs(sum(ar.resid)) < 1e-12                     # intercept ⇒ residuals sum to zero
        # Standard errors match the textbook OLS formula.
        X = [ones(7) x[1:end-1]]
        @test collect(ar.se) ≈ sqrt.(diag(ar.σ^2 * inv(X' * X)))
        @test ar.r2 ≈ cor(ar.fitted, x[2:end])^2
        @test fit_ar1(ta[:a]).φ == ar.φ
        @test_throws ArgumentError fit_ar1(ta)
    end
end

@testset "IbbotsonSinquefield simulation" begin
    using Random: Xoshiro
    using LinearAlgebra: I

    ar_π = AR1(0.01, 0.6, 0.02, (0.0, 0.0), 0.0, Float64[], Float64[], Int[], (1, 2))
    ar_r = AR1(0.002, 0.8, 0.015, (0.0, 0.0), 0.0, Float64[], Float64[], Int[], (1, 2))
    ar_tp = AR1(0.006, 0.6, 0.01, (0.0, 0.0), 0.0, Float64[], Float64[], Int[], (1, 2))
    c, A, names = is_coefficients(ar_π, ar_r, ar_tp; erp_target = 0.04)

    @testset "is_coefficients" begin
        @test names == [:π, :r, :tp, :erp]
        @test c == [0.01, 0.002, 0.006, 0.04]
        @test A == [0.6 0 0 0; 0 0.8 0 0; 0 0 0.6 0; 0 0 0 0]
        @test unconditional_mean(c, A) ≈ [unconditional_mean(ar_π), unconditional_mean(ar_r), 0.015, 0.04]
        c2, A2, _ = is_coefficients(ar_π, ar_r, ar_tp; erp_target = 0.04, tp_mean = 0.02)
        @test A2 == A
        @test unconditional_mean(c2, A2)[3] ≈ 0.02
    end

    # Fake history: VAR data with a non-zero residual mean, so demeaning matters.
    rng = Xoshiro(7)
    n_hist = 60
    Y = zeros(n_hist, 4)
    Y[1, :] = unconditional_mean(c, A)
    for t in 2:n_hist
        Y[t, :] = c .+ A * Y[t-1, :] .+ 0.01 .+ [0.02, 0.015, 0.01, 0.17] .* randn(rng, 4)
    end
    hist = TimeArray(Date.(1961:2020, 12, 31), Y, names)

    @testset "issm_bootstrap_matrix" begin
        yrs, U = issm_bootstrap_matrix(hist, c, A, names, (1971, 2020))
        @test yrs == collect(1972:2020)
        @test size(U) == (49, 4)
        @test all(abs.(mean(U; dims = 1)) .< 1e-14)
        raw = Y[12:end, :] .- c' .- Y[11:end-1, :] * A'
        @test U ≈ raw .- mean(raw; dims = 1)
        # Columns may come in any order of names.
        perm = [2, 1, 3, 4]
        _, U2 = issm_bootstrap_matrix(hist, c[perm], A[perm, perm], names[perm], (1971, 2020))
        @test U2 ≈ U[:, perm]
        @test_throws DimensionMismatch issm_bootstrap_matrix(hist, c, A[:, 1:3], names, (1971, 2020))
    end

    _, U = issm_bootstrap_matrix(hist, c, A, names, (1961, 2020))

    @testset "means match (I − A)⁻¹ c" begin
        A_off = copy(A)
        A_off[4, 2] = -0.5                                  # r → erp
        for Ax in (A, A_off)
            sc = simulate_var_bootstrapped_res(c, Ax, names, U, 50, 4000; y0 = :mean, rng = Xoshiro(1))
            @test size(sc.data) == (4, 50, 4000)
            m = vec(mean(sc.data; dims = (2, 3)))
            @test m ≈ unconditional_mean(c, Ax) atol = 2e-3
        end
        sc = simulate_var_bootstrapped_res(c, A, names, U, 50, 4000; y0 = :mean, rng = Xoshiro(1))
        @test mean(sc[:erp]) ≈ 0.04 atol = 2e-3
        @test mean(sc[:π]) ≈ 0.01 / 0.4 atol = 2e-3
    end

    @testset "U = 0 reproduces the deterministic recursion" begin
        y0 = [0.06, -0.01, 0.0, 0.0]
        sc = simulate_var_bootstrapped_res(c, A, names, zeros(5, 4), 10, 3; y0 = y0, start_year = 2026)
        y = copy(y0)
        for t in 1:10
            y = c .+ A * y
            @test all(sc.data[:, t, s] ≈ y for s in 1:3)
        end
        @test scenario_years(sc) == 2026:2035
    end

    @testset "y0 = :last and start_year from data" begin
        sc = simulate_var_bootstrapped_res(c, A, names, zeros(1, 4), 1, 1; data = hist)
        @test sc.start_year == 2021
        @test sc.data[:, 1, 1] ≈ c .+ A * Y[end, :]
        @test_throws ArgumentError simulate_var_bootstrapped_res(c, A, names, U, 5, 2)
        @test_throws ArgumentError simulate_var_bootstrapped_res(c, A, names, U, 5, 2; y0 = :foo)
        @test_throws DimensionMismatch simulate_var_bootstrapped_res(c, A[:, 1:3], names, U, 5, 2; y0 = :mean)
    end

    @testset "unstable A is rejected" begin
        A_bad = copy(A)
        A_bad[1, 1] = 1.0
        @test_throws ArgumentError simulate_var_bootstrapped_res(c, A_bad, names, U, 5, 2; y0 = :mean)
    end

    @testset "returns_from_components" begin
        sc = simulate_var_bootstrapped_res(c, A, names, U, 20, 5; y0 = :mean, rng = Xoshiro(3))
        nom = returns_from_components(sc)
        rl = returns_from_components(sc; real = true)
        @test nom.names == rl.names == [:rf, :rb, :re]
        @test nom.start_year == sc.start_year
        @test size(nom.data) == (3, 19, 5)
        rf = sc[:π] .+ sc[:r]
        @test nom[:rf] ≈ rf[1:end-1, :]
        @test nom[:re] ≈ (rf .+ sc[:erp])[1:end-1, :]
        y10 = exp.(rf .+ sc[:tp]) .- 1
        @test nom[:rb] ≈ log.(1 .+ IbbotsonSinquefield.calculate_bond_returns(y10, 10, 1))
        # Constant yield: the par bond returns its yield.
        flat = Scenarios(repeat([0.02, 0.01, 0.015, 0.04], 1, 4, 2), [:π, :r, :tp, :erp], 2026)
        @test returns_from_components(flat)[:rb] ≈ fill(0.045, 3, 2)
        @test rl.data ≈ nom.data .- reshape(sc[:π][1:end-1, :], 1, 19, 5)
    end
end
