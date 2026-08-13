using Distributions, LinearAlgebra

const N = 6  # number of gap variables

sv_priors = (
    # Average level of each log-variance process.
    # Variance is the ergodic one implied by the drawn σ² and ρ, so this is
    # a closure evaluated at the current draws.
    h̄ = (ρ, σ²) -> Normal(log(0.1^2), sqrt(σ² / (1 - ρ^2))),

    # AR(1) lag coefficient of each log-variance process (Clark & Ravazzolo, 2015)
    ρ = Normal(0.8, 0.2),

    # Covariance of shocks to the gap volatilities (ηₜ, eq. 9):
    # IW, ν = N + 11, E[·] = 0.2²·I_N
    Ω_gap = InverseWishart(float(N + 11), Matrix(0.04 * (N + 11 - N - 1) * I, N, N)),
)