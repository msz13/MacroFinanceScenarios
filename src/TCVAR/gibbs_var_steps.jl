"""
    MinnesotaPrior{T}

Conjugate Normal–Inverse-Wishart ("Minnesota") prior for a reduced-form BVAR,
following Giannone, Lenza & Primiceri (2015), *Prior Selection for Vector
Autoregressions*, REStat.

Model (stacked form):

    Y = X Φ + E,     vec(E) ~ N(0, Σ ⊗ I_T)

where each row of `X` is `[ yₜ₋₁' … yₜ₋ₚ' , 1 ]` (p lags, then the intercept),
so `X` has `k = n*p + 1` columns and `Φ` is `k × n`.

Prior:

    Σ      ~ IW(Ψ, d)
    Φ | Σ  ~ MN(Φ₀, Ω, Σ)          # vec(Φ) | Σ ~ N(vec(Φ₀), Σ ⊗ Ω)

Minnesota structure:
* `Φ₀` is zero except the own-first-lag of each equation, set to `δᵢ`
  (random walk when `δᵢ = 1`).
* `Ω` is diagonal; the entry for the coefficient on lag `s` of variable `j` is
  `(d-n-1) * λ² / (s² * ψⱼ)`. This makes the *marginal* prior variance of that
  coefficient (in equation `i`) equal to `λ²/s² * ψᵢ/ψⱼ` — the classic
  Minnesota shrinkage: tighter for higher lags, scaled by relative residual
  variances. The `(d-n-1)` factor cancels `E[Σᵢᵢ] = ψᵢ/(d-n-1)`, so the
  marginal coefficient variance does not depend on `d`.
* The intercept gets a loose prior variance `ω_c`.

# Fields
- `n::Int`  number of endogenous variables
- `p::Int`  number of lags
- `k::Int`  regressors per equation (`= n*p + 1`)
- `λ::T`    overall tightness hyperparameter (→0 dogmatic, →∞ flat)
- `Φ₀`      `k × n` prior coefficient mean
- `Ω`       `k × k` diagonal prior row-covariance
- `Ψ`       `n × n` Inverse-Wishart scale (diagonal)
- `d::T`    Inverse-Wishart degrees of freedom
"""
struct MinnesotaPrior{T<:Real}
    n::Int
    p::Int
    k::Int
    λ::T
    Φ₀::Matrix{T}
    Ω::Diagonal{T,Vector{T}}
    Ψ::Diagonal{T,Vector{T}}
    d::T
end

"""
    MinnesotaPrior(λ, n, p, ψ, d; δ = ones(n), ω_c = 1e6)

Build the Giannone–Lenza–Primiceri Minnesota prior.

# Arguments
- `λ`      : tightness hyperparameter (`> 0`).
- `n`, `p` : number of variables and lags.
- `ψ`      : length-`n` diagonal of the IW scale matrix `Ψ` (the prior scale of
             `diag(Σ)`; with `d = n+2` it equals the prior mean of `diag(Σ)`).
- `d`      : IW degrees of freedom (`> n+1` so that `E[Σ]` exists; GLP default
             is `n+2`).

# Keywords
- `δ`   : length-`n` prior mean of each variable's own first lag
          (default `1` ⇒ random-walk prior; use `0` for white-noise prior).
- `ω_c` : prior row-variance assigned to the intercept (default `1e6`, flat).
"""
function MinnesotaPrior(λ::Real, n::Integer, p::Integer,
                        ψ::AbstractVector{<:Real}, d::Real;
                        δ::AbstractVector{<:Real} = ones(n),
                        ω_c::Real = 1e6)
    n ≥ 1          || throw(ArgumentError("n must be ≥ 1"))
    p ≥ 1          || throw(ArgumentError("p must be ≥ 1"))
    λ > 0          || throw(ArgumentError("λ must be > 0"))
    length(ψ) == n || throw(DimensionMismatch("ψ must have length n = $n"))
    length(δ) == n || throw(DimensionMismatch("δ must have length n = $n"))
    all(>(0), ψ)   || throw(ArgumentError("all ψⱼ must be > 0"))
    d > n + 1      || throw(ArgumentError("d must be > n+1 for E[Σ] to exist"))

    T = float(promote_type(typeof(λ), eltype(ψ), typeof(d),
                           eltype(δ), typeof(ω_c)))
    λ_, d_, ωc_ = T(λ), T(d), T(ω_c)
    ψ_, δ_      = collect(T, ψ), collect(T, δ)

    k = n * p + 1
    scale = d_ - n - 1                      # (d-n-1); equals 1 when d = n+2

    # ---- prior mean Φ₀: own first lag = δᵢ, everything else 0 ----
    Φ₀ = zeros(T, k, n)
    @inbounds for i in 1:n
        Φ₀[i, i] = δ_[i]
    end

    # ---- diagonal of Ω (row-covariance) ----
    # regressor order: [lag1 vars 1..n, lag2 vars 1..n, …, lagp vars 1..n, const]
    ω = Vector{T}(undef, k)
    @inbounds for s in 1:p, j in 1:n
        ω[(s - 1) * n + j] = scale * λ_^2 / (s^2 * ψ_[j])
    end
    ω[k] = ωc_                              # loose intercept

    return MinnesotaPrior{T}(n, p, k, λ_, Φ₀, Diagonal(ω), Diagonal(ψ_), d_)
end

"""
    coef_cov(prior, Σ) -> Symmetric

Full conditional prior covariance of `vec(Φ)` given `Σ`, i.e. `Σ ⊗ Ω`.
"""
coef_cov(pr::MinnesotaPrior, Σ::AbstractMatrix) =
    Symmetric(kron(Matrix(Σ), Matrix(pr.Ω)))
    

function prepare_var_data(Y::Matrix{Float64}, p::Int, X::Union{Matrix{Float64},Vector{Float64}} = Matrix{Float64}(undef, 0, 0), add_intercept::Bool = false)
    T, n = size(Y)
    Y_lagged = zeros(T - p, n * p)
    for t in (p + 1):T
        Y_lagged[t - p, :] = Y[t-p:t-1, :]'
    end

    predictors = Y_lagged

    if !isempty(X)
        if size(X, 1) != T
            error("The number of rows in X must be equal to the number of rows in Y.")
        end
        X_subset = X[p+1:end, :]
        predictors = hcat(predictors, X_subset)
    end

    if add_intercept
        intercept = ones(T - p, 1)
        predictors = hcat(intercept, predictors)
    end

    return Y[p+1:end, :], predictors
end


posterior_beta_coefficient_mean(Y, X, beta_mean, Ω_inv) = inv(X'X + Ω_inv)*(X'Y + Ω_inv*beta_mean)


#posterior_beta_coefficient_var(X, Σ, Ω_inv) = kron(Σ, inv(X'X + Ω_inv)) do usuniecia

#Ω_inv prior of beta coefficient variance

function draw_beta(X, Σ, beta_posterior, Ω_inv)

    n = size(Σ, 1)      # number of equations
    k = size(X, 2)      # number of predictors (n * p)
    m = n * k           # length of vec(β)

    beta_var = kron(Σ, inv(X'X + Ω_inv)) + I(m) * 1e-5

    F = eigen(Hermitian(beta_var))
    λ = max.(F.values, 0.0)
    L = F.vectors * Diagonal(sqrt.(λ))

    return vec(beta_posterior) + L * randn(m)

end


function covariance_posterior_dist(Y, X, β_posterior_μ, posterior_df, variance_prior, β_prior_μ, Ω_inv)

    ε =  Y - X * β_posterior_μ

    β_diff = β_posterior_μ - β_prior_μ

    S = ε' * ε + β_diff' * Ω_inv * β_diff  + variance_prior

    return InverseWishart(posterior_df, collect(Hermitian(S)))

end



"""
    is_stationary(var_coeff, n, p)

Check VAR(p) stationarity via the companion matrix eigenvalues.
`var_coeff` is the n × n*p companion bottom block in the oldest-lag-first ordering
used by [`tc_var`](@ref) (i.e. `B'` for the regression `Y = X·B`).
Returns true if all eigenvalues of the companion matrix have modulus < 1.
"""
function is_stationary(var_coeff::AbstractMatrix, n::Int, p::Int)
    if p == 1
        return all(abs.(eigvals(var_coeff)) .< 1.0)
    end
    companion = vcat(
        hcat(zeros(n * (p - 1), n), I(n * (p - 1))),
        var_coeff)
    return all(abs.(eigvals(companion)) .< 1.0)
end


"""
    sample_var_params(data,p, β_mean, Ω_inv)

    data: observations
    p: number of lags
    β_priormean: prior mean of beta coefficients
    Ω_inv: inversion prior variance of beta coefficients
    S: prior covariance scale
    df: posterior covariance distribution degrees of freedom
"""
function sample_var_params(data, p, β_prior_μ, Ω_inv, S, df; max_draws::Int = 100)

    Y, X = prepare_var_data(data, p)
    n = size(Y, 2)

    β_hat = posterior_beta_coefficient_mean(Y, X, β_prior_μ, Ω_inv)

    Σ = rand(covariance_posterior_dist(Y, X, β_hat, df, S, β_prior_μ, Ω_inv))

    β = draw_beta(X, Σ, β_hat, Ω_inv)

    # Companion bottom block A = B' (n × n*p) in oldest-lag-first ordering.
    var_coeff(β) = collect(reshape(β, n * p, n)')

    draws = 1
    while !is_stationary(var_coeff(β), n, p) && draws < max_draws
        β = draw_beta(X, Σ, β_hat, Ω_inv)
        draws += 1
    end

    return β, Σ

end
  
 