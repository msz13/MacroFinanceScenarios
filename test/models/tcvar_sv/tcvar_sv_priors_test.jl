using Test
using Distributions
using LinearAlgebra
using Statistics

isdefined(Main, :TCVAR) || include(joinpath(@__DIR__, "..", "..", "..", "src", "TCVAR", "TCVAR.jl"))
isdefined(Main, :tcvar_test_priors) || include(joinpath(@__DIR__, "..", "..", "tcvar_test_utils.jl"))

# TCVAR members are reached as `TCVAR.f` rather than via `using .TCVAR` — see the note in
# tcvar_test_utils.jl.
isdefined(TCVAR, :tcvar_sv_priors) || error(
    "The TCVAR module loaded in this session predates the TCVAR-SV priors. " *
    "Restart the Julia session (or REPL / IDE worker) and re-run.")

@testset "models/tcvar_sv/tcvar_sv_priors" begin

    n, nt, p = 3, 2, 2

    tc_keys() = tcvar_test_priors(; n = n, nt = nt, p = p)
    sv_keys() = TCVAR.sv_priors(n)

    @testset "the type" begin
        # A NamedTuple is invariant in its value-type parameter, so a tuple carrying a
        # FullNormal is not `isa TCVARSVPriors` even with the keys in the declared order.
        # Construction, not `isa`, is the validation — this is what the type is for.
        @test isconcretetype(TCVAR.TCVARSVPriors)
        @test fieldnames(TCVAR.TCVARSVPriors) ==
              (:initial_trend, :initial_cycle, :trend_covariance, :cycle_covariance,
               :cycle_β, :volatility_mean, :volatility_ar, :volatility_covariance,
               :simultaneity)
    end

    @testset "assembly from the two source tuples" begin
        tc, sv = tc_keys(), sv_keys()
        priors = TCVAR.tcvar_sv_priors(tc, sv)

        @test priors isa TCVAR.TCVARSVPriors
        @test keys(priors) == fieldnames(TCVAR.TCVARSVPriors)

        # Assembly only — every distribution is the one the source tuple supplied,
        # stored by reference rather than rebuilt.
        @test mean(priors.initial_cycle) == mean(tc.initial_cycle)
        @test priors.cycle_β === tc.cycle_β
        @test mean(priors.volatility_mean) == mean(sv.volatility_mean)
        @test mean(priors.simultaneity) == mean(sv.simultaneity)

        # Key order at the call site is free; the stored tuple is canonical.
        shuffled = NamedTuple{reverse(keys(tc))}(values(reverse(tc)))
        @test keys(TCVAR.tcvar_sv_priors(shuffled, sv_keys())) ==
              fieldnames(TCVAR.TCVARSVPriors)

        # An extra key is dropped by the two-tuple form ...
        extra = merge(tc, (nonsense = 1,))
        @test keys(TCVAR.tcvar_sv_priors(extra, sv_keys())) ==
              fieldnames(TCVAR.TCVARSVPriors)
        # ... and is a loud MethodError in the keyword form, which is why that one is
        # the recommended entry point.
        @test_throws MethodError TCVAR.tcvar_sv_priors(; extra..., sv_keys()...)

        # A missing key still throws, naming the field.
        @test_throws FieldError TCVAR.tcvar_sv_priors(Base.structdiff(tc, (cycle_β = 1,)),
                                                      sv_keys())
    end

    @testset "the keyword form is the same function" begin
        tc, sv = tc_keys(), sv_keys()
        from_keywords = TCVAR.tcvar_sv_priors(; tc..., sv...)
        from_tuples   = TCVAR.tcvar_sv_priors(tc, sv)

        @test keys(from_keywords) == keys(from_tuples)
        @test all(mean(from_keywords[k]) == mean(from_tuples[k])
                  for k in (:initial_trend, :initial_cycle, :volatility_mean, :simultaneity))
    end

    @testset "internal shape validation" begin
        tc, sv = tc_keys(), sv_keys()

        # n and p are read off cycle_β, so a mismatched initial_cycle is caught here.
        wrong_cycle = merge(tc, (initial_cycle = MvNormal(zeros(n * p + 1),
                                                          Matrix(1.0I, n * p + 1, n * p + 1)),))
        @test_throws DimensionMismatch TCVAR.tcvar_sv_priors(wrong_cycle, sv)

        wrong_Σc = merge(tc, (cycle_covariance = InverseWishart(10.0, Matrix(1.0I, n + 1, n + 1)),))
        @test_throws DimensionMismatch TCVAR.tcvar_sv_priors(wrong_Σc, sv)

        wrong_Ω = merge(sv, (volatility_covariance = InverseWishart(20.0, Matrix(1.0I, n + 1, n + 1)),))
        @test_throws DimensionMismatch TCVAR.tcvar_sv_priors(tc, wrong_Ω)

        wrong_μ = merge(sv, (volatility_mean = MvNormal(zeros(n + 1), Matrix(1.0I, n + 1, n + 1)),))
        @test_throws DimensionMismatch TCVAR.tcvar_sv_priors(tc, wrong_μ)

        wrong_A₀ = merge(sv, (simultaneity = MvNormal(zeros(2), Matrix(1.0I, 2, 2)),))
        @test_throws DimensionMismatch TCVAR.tcvar_sv_priors(tc, wrong_A₀)

        # n_trends has to agree with itself across the two trend keys.
        wrong_trend = merge(tc, (initial_trend = MvNormal(zeros(nt + 1),
                                                          Matrix(1.0I, nt + 1, nt + 1)),))
        @test_throws DimensionMismatch TCVAR.tcvar_sv_priors(wrong_trend, sv)

        @test_throws ArgumentError TCVAR.tcvar_sv_priors(tc, sv; ar_structure = :block)
    end

    @testset "ar_structure selects the volatility_ar length" begin
        @test length(TCVAR.sv_priors(n).volatility_ar) == n
        @test length(TCVAR.sv_priors(n; ar_structure = :full).volatility_ar) == n^2
        # The :full prior is the same AR(1) belief lifted to vec(Φᵀ): centred on the
        # diagonal matrix, every off-diagonal spillover shrunk towards zero.
        @test mean(TCVAR.sv_priors(n; ar_structure = :full).volatility_ar) ==
              vec(Matrix(0.8I, n, n))
        @test_throws ArgumentError TCVAR.sv_priors(n; ar_structure = :block)

        full = TCVAR.tcvar_sv_priors(tc_keys(), TCVAR.sv_priors(n; ar_structure = :full);
                                     ar_structure = :full)
        @test length(full.volatility_ar) == n^2

        # A diagonal prior under :full (and vice versa) is a length error, not silence.
        @test_throws DimensionMismatch TCVAR.tcvar_sv_priors(tc_keys(), sv_keys();
                                                             ar_structure = :full)
        @test_throws DimensionMismatch TCVAR.tcvar_sv_priors(
            tc_keys(), TCVAR.sv_priors(n; ar_structure = :full))
    end
end
