using Test
using Distributions
using LinearAlgebra
using Random
using Statistics

isdefined(Main, :TCVAR) || include(joinpath(@__DIR__, "..", "..", "src", "TCVAR", "TCVAR.jl"))

# TCVAR members are reached as `TCVAR.f` rather than via `using .TCVAR` — see the note in
# tcvar_test_utils.jl.
isdefined(TCVAR, :lyapunov_covariance) || error(
    "The TCVAR module loaded in this session predates lyapunov_covariance. " *
    "Restart the Julia session (or REPL / IDE worker) and re-run.")

@testset "common/linalg" begin

    @testset "lyapunov_covariance" begin
        F = [0.7 0.2; -0.1 0.5]
        Q = [0.4 0.1; 0.1 0.3]

        P = TCVAR.lyapunov_covariance(F, Q)

        # Defining property: P solves the discrete Lyapunov equation.
        @test P ≈ F * P * F' + Q
        @test P ≈ P'
        @test all(eigvals(Symmetric(P)) .> 0)

        # It is the unconditional covariance, so iterating the recursion converges to it.
        iterated = zeros(2, 2)
        for _ in 1:500
            iterated = F * iterated * F' + Q
        end
        @test P ≈ iterated

        # Zero dynamics leave the innovation covariance untouched.
        @test TCVAR.lyapunov_covariance(zeros(2, 2), Q) ≈ Q

        # Scalar case has the closed form q / (1 - f²).
        @test TCVAR.lyapunov_covariance(fill(0.9, 1, 1), fill(0.19, 1, 1))[1] ≈ 0.19 / (1 - 0.81)
    end

    @testset "lyapunov_covariance behind its callers" begin
        # `initial_cycle_prior` and `stationary_cycle_covariance` are the two existing
        # call sites; both must still return a covariance that solves their own Lyapunov
        # equation after the extraction.
        Σ_prior, β_prior, c₀_prior = TCVAR.var_priors(0.4, 2, [1.0, 0.5]; δ = zeros(2))

        A = TCVAR.prior_var_coeff(β_prior)              # n × n*p, oldest-lag-first
        n, p = β_prior.n, β_prior.p
        F = vcat(hcat(zeros(n * (p - 1), n), I(n * (p - 1))), A)
        Q = zeros(n * p, n * p)
        Q[end-n+1:end, end-n+1:end] = mean(Σ_prior)

        @test Matrix(cov(c₀_prior)) ≈ F * Matrix(cov(c₀_prior)) * F' + Q

        model = TCVAR.tc_var(Matrix(1.0I, 2, 2); p = 2)
        TCVAR.update_tc_var!(model, [0.1 0.0 0.5 0.2; 0.0 0.1 -0.1 0.4],
                             diagm([0.01, 0.02]), [0.5 0.1; 0.1 0.3], 2, 2, 2)

        P = TCVAR.stationary_cycle_covariance(model, 2)
        Tc = model.T[3:end, 3:end]
        Qc = model.Q[3:end, 3:end]
        @test P ≈ Tc * P * Tc' + Qc
    end

end
