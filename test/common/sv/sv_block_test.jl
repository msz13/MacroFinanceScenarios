using Test
using Distributions
using LinearAlgebra
using Random
using Statistics

isdefined(Main, :TCVAR) || include(joinpath(@__DIR__, "..", "..", "..", "src", "TCVAR", "TCVAR.jl"))

# TCVAR members are reached as `TCVAR.f` rather than via `using .TCVAR` — see the note in
# tcvar_test_utils.jl.
isdefined(TCVAR, :draw_stochastic_volatility) || error(
    "The TCVAR module loaded in this session predates the stochastic-volatility block. " *
    "Restart the Julia session (or REPL / IDE worker) and re-run.")

"""Simulate a log-volatility VAR(1) path `h_{0:T}` and residuals `e_t ~ N(0, diag(exp(h_t)))`."""
function simulate_volatility_path(n_time, μ, Φ, Ω; h0 = μ)
    n = length(μ)
    Ω_L = cholesky(Symmetric(Ω)).L

    h = zeros(n_time + 1, n)
    h[1, :] = h0
    for t in 2:(n_time + 1)
        h[t, :] = μ + Φ * (h[t-1, :] - μ) + Ω_L * randn(n)
    end

    residuals = [randn() * exp(h[t+1, i] / 2) for t in 1:n_time, i in 1:n]
    return h, residuals
end

@testset "common/sv/sv_block" begin

    mixture = TCVAR.KSC_MIXTURE

    @testset "shapes and argument checking" begin
        Random.seed!(1)
        n, n_time = 3, 40
        params = (μ = fill(-2.0, n), Φ = Matrix(0.9I, n, n), Ω = Matrix(0.05I, n, n))
        _, residuals = simulate_volatility_path(n_time, params.μ, params.Φ, params.Ω)
        h = repeat(params.μ', n_time + 1)

        h_new, indicators = TCVAR.draw_stochastic_volatility(residuals, h, params)

        @test size(h_new) == (n_time + 1, n)
        @test size(indicators) == (n_time, n)
        @test all(j -> 1 ≤ j ≤ 7, indicators)
        @test all(isfinite, h_new)

        # `h` carries the h_0 row, so it is one row longer than the residuals.
        @test_throws DimensionMismatch TCVAR.draw_stochastic_volatility(residuals, h[2:end, :], params)
        @test_throws ArgumentError TCVAR.draw_stochastic_volatility(residuals, h, params; offset = 0.0)

        bad_params = (μ = fill(-2.0, n + 1), Φ = params.Φ, Ω = params.Ω)
        @test_throws DimensionMismatch TCVAR.draw_stochastic_volatility(residuals, h, bad_params)
    end

    @testset "a zero residual is finite through the offset" begin
        # log(0) is what the KSC offset c̄ exists to prevent.
        Random.seed!(2)
        n, n_time = 2, 10
        params = (μ = zeros(n), Φ = Matrix(0.5I, n, n), Ω = Matrix(0.1I, n, n))
        residuals = zeros(n_time, n)
        h = zeros(n_time + 1, n)

        h_new, _ = TCVAR.draw_stochastic_volatility(residuals, h, params)
        @test all(isfinite, h_new)
    end

    @testset "indicator frequencies reproduce the mixture weights" begin
        # With h at the truth the measurement error y* − h is exactly log χ²₁, and the
        # marginal label frequencies are then the mixture weights themselves. Formed from
        # an exact χ²₁ rather than through `draw_stochastic_volatility` so that the KSC
        # offset — which floors the far left tail and depletes component 1 — does not
        # enter this particular comparison.
        Random.seed!(3)
        n_time = 200_000
        h = fill(0.9, n_time, 1)
        y_star = h .+ log.(rand(Chisq(1), n_time))

        indicators = TCVAR.draw_mixture_indicators(y_star, h)
        empirical = [count(==(j), indicators) / n_time for j in 1:7]

        @test empirical ≈ mixture.probabilities atol = 0.01
    end

    @testset "draw_log_volatilities de-means by the component and by μ" begin
        # Ω enormous, prior on h̃_0 diffuse: the state has no pull, so the smoothed mean
        # of h_t collapses onto the de-meaned observation y*_t − m_{s_t}. Averaging over
        # draws removes the Carter–Kohn sampling noise, which is what is left once the
        # prior is uninformative.
        Random.seed!(4)
        n, n_time = 2, 40
        params = (μ = [0.3, -0.7], Φ = Matrix(0.5I, n, n), Ω = Matrix(1e6I, n, n))
        y_star = randn(n_time, n)
        indicators = fill(4, n_time, n)          # the narrowest component, v² = 0.167

        n_repeats = 2_000
        average = zeros(n_time + 1, n)
        for _ in 1:n_repeats
            average .+= TCVAR.draw_log_volatilities(y_star, indicators, params;
                                                    h0_covariance = Matrix(1e6I, n, n))
        end
        average ./= n_repeats

        # Element-wise, not `≈`: over 80 cells the norm-based tolerance would be
        # dominated by the accumulated Monte Carlo error rather than by any one cell.
        @test maximum(abs.(average[2:end, :] .- (y_star .- mixture.means[4]))) < 0.06
    end

    @testset "a vanishing Ω pins h at μ" begin
        # The other end of the same seam: with no state innovations and a stable Φ, h̃
        # cannot leave its unconditional mean of zero, so h ≡ μ whatever the data say.
        Random.seed!(5)
        n, n_time = 2, 40
        params = (μ = [0.3, -0.7], Φ = Matrix(0.5I, n, n), Ω = Matrix(1e-10I, n, n))

        h_new = TCVAR.draw_log_volatilities(randn(n_time, n), rand(1:7, n_time, n), params)

        @test maximum(abs.(h_new .- repeat(params.μ', n_time + 1))) < 1e-3
    end

    @testset "initial-state covariance" begin
        Φ = [0.9 0.1; 0.0 0.8]
        Ω = [0.05 0.01; 0.01 0.04]

        P₀ = TCVAR.stationary_volatility_covariance(Φ, Ω)
        @test P₀ ≈ Φ * P₀ * Φ' + Ω

        # A random-walk volatility has no stationary distribution: the caller has to say
        # what the diffuse prior on h_0 is.
        unit_root = Matrix(1.0I, 2, 2)
        @test_throws ArgumentError TCVAR.stationary_volatility_covariance(unit_root, Ω)

        Random.seed!(6)
        params = (μ = zeros(2), Φ = unit_root, Ω = Ω)
        _, residuals = simulate_volatility_path(30, params.μ, params.Φ, params.Ω)
        h = zeros(31, 2)

        @test_throws ArgumentError TCVAR.draw_stochastic_volatility(residuals, h, params)
        @test size(TCVAR.draw_stochastic_volatility(residuals, h, params;
                                                    h0_covariance = Matrix(10.0I, 2, 2))[1]) == (31, 2)
    end

    @testset "recovers a simulated volatility path" begin
        # The end-to-end check: iterate the block on data simulated at known parameters
        # and see whether the posterior mean of h tracks the truth. The level ratio is
        # the assertion that matters — dropping the -1.2704 shift, or applying it twice,
        # leaves the path shape intact but scales exp(h/2) by 1.89 or 0.53.
        Random.seed!(20250819)
        n, n_time = 2, 250
        μ = [log(0.5^2), log(1.5^2)]
        Φ = Matrix(0.95I, n, n)
        Ω = [0.05 0.02; 0.02 0.04]
        params = (μ = μ, Φ = Φ, Ω = Ω)

        h_true, residuals = simulate_volatility_path(n_time, μ, Φ, Ω;
                                                     h0 = μ + [0.2, -0.3])

        n_burnin, n_kept = 400, 400
        h_draw = repeat(μ', n_time + 1)
        kept = zeros(n_kept, n_time + 1, n)
        for iteration in 1:(n_burnin + n_kept)
            h_draw, _ = TCVAR.draw_stochastic_volatility(residuals, h_draw, params)
            iteration > n_burnin && (kept[iteration - n_burnin, :, :] = h_draw)
        end

        posterior_mean = dropdims(mean(kept, dims = 1), dims = 1)

        for i in 1:n
            # The path is identified only up to the considerable noise of a single
            # squared observation per period, hence the loose correlation floor.
            @test cor(posterior_mean[:, i], h_true[:, i]) > 0.6

            # No systematic level bias in the standard deviation scale.
            level_ratio = mean(exp.(posterior_mean[:, i] ./ 2)) / mean(exp.(h_true[:, i] ./ 2))
            @test 0.8 < level_ratio < 1.25
        end

        # `compute_posterior_statistics` consumes exactly this layout, and the truth
        # should sit inside a 90% band most of the time.
        _, lower, upper = TCVAR.compute_posterior_statistics(kept)
        coverage = mean(lower .<= h_true .<= upper)
        @test coverage > 0.7
    end

end
