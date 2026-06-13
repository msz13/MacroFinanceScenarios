"""
Port of FRBNY-DSGE rstarBrookings2017 tvar/MainModel1.m to Julia.

State-space model with common trends (π*, r*, ts*) and a VAR cycle:
    y_t = C * s_t + ε_t,       ε_t ~ N(0, R)
    s_t = A * s_{t-1} + η_t,   η_t ~ N(0, Q)

State vector s_t = [π*_t, r*_t, ts*_t, y_t, y_{t-1}, …, y_{t-p+1}]
                   |— trends (r=3) —|  |———— VAR companion (n·p) ————|

Estimation uses a Gibbs sampler that alternates:
  1. Draw states via Carter–Kohn smoother
  2. Draw VAR coefficients via Bayesian VAR
  3. Draw trend covariance via Inverse-Wishart

Data: first 5 series from DelNegro.xlsx (Pi, EPi, BILL, EBILL, TBlong)
"""
module MainModel1

using LinearAlgebra
using Random
using Dates
using Statistics
using XLSX

include("utils.jl")
include("kalman_filter.jl")
include("carter_kohn.jl")
include("bvar.jl")
include("covariance_draw.jl")

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

"""
    load_data(path) -> (dates, Y, mnem)

Read the Excel data file and return:
  - `dates` : Vector{Date}  of quarterly dates
  - `Y`     : T × ncols  Float64 matrix (NaN where missing)
  - `mnem`  : Vector{String}  column names
"""
function load_data(path::String)
    wb   = XLSX.readxlsx(path)
    sh   = wb[XLSX.sheetnames(wb)[1]]
    data = sh[:]

    mnem  = String.(data[1, 2:end])
    dates = Date.(data[2:end, 1])

    nrows = size(data, 1) - 1
    ncols = size(data, 2) - 1
    Y     = fill(NaN, nrows, ncols)
    for i in 1:nrows, j in 1:ncols
        v = data[i+1, j+1]
        if !ismissing(v) && v !== nothing
            Y[i, j] = Float64(v)
        end
    end
    return dates, Y, mnem
end

# ─────────────────────────────────────────────────────────────────────────────
# Main estimation
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_model1(; data_path, ndraws, seed) -> NamedTuple

Estimate Model 1 (π + Yields) via Gibbs sampling.

Keyword arguments:
  - `data_path` : path to DelNegro.xlsx (default: "data/DelNegro.xlsx")
  - `ndraws`    : total MCMC draws (default: 100_000)
  - `seed`      : RNG seed (0 = random; default: 0)

Returns a NamedTuple with posterior draws (post burn-in):
  - `common_trends` : T × r × M  (r = 3: π*, r*, ts*)
  - `cycles`        : T × n × M
  - `trends`        : T × n × M  (observation-space trend)
  - `trends_real`   : T × n × M  (real trends, π-deflated)
  - `AA`, `QQ`, `CC`, `RR` : model matrices per draw
  - `loglik`        : M-vector of log-likelihoods
  - `SS0`           : r × M  initial trend draws
  - `dates`, `Y`, `y`, `mnem`  : data
"""
function run_model1(;
    data_path :: String = "data/DelNegro.xlsx",
    ndraws    :: Int    = 100_000,
    seed      :: Int    = 0,
)
    seed != 0 && Random.seed!(seed)

    # ── Load data ────────────────────────────────────────────────────────────
    dates_all, Y_all, mnem_all = load_data(data_path)

    first_year = 1960
    last_year  = 2016
    select_idx = 1:5          # Pi, EPi, BILL, EBILL, TBlong

    t0 = findfirst(d -> year(d) == first_year, dates_all)
    t1 = findlast( d -> year(d) == last_year,  dates_all)

    dates = dates_all[t0:t1]
    Y     = Y_all[t0:t1, select_idx]
    mnem  = mnem_all[select_idx]
    y     = copy(Y)
    T, n  = size(y)

    # Apply missingness rules matching the MATLAB code
    t70  = findlast(d -> year(d) == 1970, dates)
    tzlb = findlast(d -> year(d) == 2008, dates)

    bill_col = findfirst(==("BILL"), mnem)
    if !isnothing(bill_col) && !isnothing(tzlb)
        y[tzlb:end, bill_col] .= NaN       # BILL set to NaN after ZLB (2008)
    end
    y[1:t70, 2] .= NaN                     # inflation expectations NaN before 1970

    # ── Model dimensions ─────────────────────────────────────────────────────
    p = 4    # VAR lags for the cycle
    r = 3    # number of common trends: π*, r*, ts*

    # Observation loadings: Ctr maps trends to observations
    #        π*  r*  ts*
    Ctr = [
        1   0   0;   # 1: Inflation
        1   0   0;   # 2: Inflation Expectations
        1   1   0;   # 3: Short rate (3m T-Bill)
        1   1   0;   # 4: Expected short rate
        1   1   1;   # 5: Long rate (20y T-Bond)
    ]
    # Cycle loadings: Ccyc = [I_n | 0 … 0]  (observe only current cycle)
    Ccyc = [I(n) zeros(n, n*(p-1))]

    C = [Ctr Ccyc]    # n × (r + n·p)

    # ── Prior hyperparameters ─────────────────────────────────────────────────
    df0tr  = 100
    SC0tr  = [2.0, 1.0, 1.0].^2 / 400   # prior variance for trend innovations
    S0tr   = [2.0, 0.5, 1.0]             # prior mean for initial trends
    P0tr   = Diagonal(ones(r))

    # Cycle prior scaling (Minnesota-style)
    Psi    = [2.0, 1.0, 1.0, 0.5, 1.0]  # per-variable variance scale

    # ── State-space matrices ─────────────────────────────────────────────────
    Atr  = I(r)
    Acyc = zeros(n*p, n*p)
    if p > 1
        Acyc[n+1:end, 1:n*(p-1)] = I(n*(p-1))
    end

    A         = zeros(r + n*p, r + n*p)
    A[1:r, 1:r] = Matrix(Atr)
    A[r+1:end, r+1:end] = Acyc

    R = I(n) * 1e-12    # near-zero measurement noise (state-space identification)

    Q0cyc = zeros(n*p, n*p)
    Q0cyc[1:n, 1:n] = Diagonal(Psi)
    Q0tr  = Diagonal(SC0tr)

    Q = zeros(r + n*p, r + n*p)
    Q[1:r,   1:r]   = Matrix(Q0tr)
    Q[r+1:end, r+1:end] = Q0cyc

    b0 = zeros(n*p, n)   # prior mean for VAR coefficients (zero)

    S0 = [S0tr; zeros(n*p)]
    P0 = zeros(r + n*p, r + n*p)
    P0[1:r,   1:r]   = Matrix(P0tr)
    P0[r+1:end, r+1:end] = Diagonal(kron(ones(p), Psi))

    notrend = findall(SC0tr .< 1e-6)   # trends with near-zero variance (none in Model 1)

    # ── Storage ──────────────────────────────────────────────────────────────
    p_acc      = fill(NaN, ndraws)
    States     = fill(NaN, T, r + n*p, ndraws)
    Trends     = fill(NaN, T, n, ndraws)
    TrendsReal = fill(NaN, T, n, ndraws)
    LogLik_arr = fill(NaN, ndraws)
    SS0_arr    = fill(NaN, r, ndraws)
    AA_arr     = fill(NaN, r + n*p, r + n*p, ndraws)
    QQ_arr     = fill(NaN, r + n*p, r + n*p, ndraws)
    CC_arr     = fill(NaN, n, r + n*p, ndraws)
    RR_arr     = fill(NaN, n, n, ndraws)

    # ── Gibbs sampler ────────────────────────────────────────────────────────
    t_start = time()
    A  = Matrix{Float64}(A)
    Q  = Matrix{Float64}(Q)
    R  = Matrix{Float64}(Matrix(R))
    C  = Matrix{Float64}(C)
    S0 = Vector{Float64}(S0)
    P0 = Matrix{Float64}(P0)

    for jm in 1:ndraws

        # Step 1a: Kalman filter
        kf     = kalman_filter(y, C, R, A, Q, S0, P0)
        loglik = kf.loglik

        # Step 1b: Metropolis step for initial trend level (skipped when notrend is empty)
        if !isempty(notrend)
            S0_new              = copy(S0)
            S0_new[notrend]    .= S0[notrend] .+ randn(length(notrend))
            kf_new              = kalman_filter(y, C, R, A, Q, S0_new, P0)
            loglik_new          = kf_new.loglik
            log_rat             = loglik_new - loglik
            acc                 = min(exp(log_rat), 1.0)
            if rand() <= acc
                S0     = S0_new
                loglik = loglik_new
                kf     = kf_new
            end
            p_acc[jm] = acc
        end

        # Step 2: Carter–Kohn backward simulation smoother
        kc   = carter_kohn(kf)

        # Step 3: Draw VAR coefficients from cycle states
        # Build augmented cycle series including p pre-sample lags from S0 draw
        Ycyc = kc.S[:, r+1:r+n]               # T × n
        for jp in 1:p
            pre_row = kc.S0[r+(jp-1)*n+1:r+jp*n]'   # 1 × n
            Ycyc    = [pre_row; Ycyc]                  # prepend
        end
        # Ycyc is now (T+p) × n

        beta, sigma = bvar(Ycyc, p, b0, Psi, 0.2, true)
        A[r+1:r+n,  r+1:end] = beta'
        Q[r+1:r+n,  r+1:r+n] = sigma

        # Step 4: Draw trend innovation covariance
        Ytr  = [kc.S0[1:r]'; kc.S[:, 1:r]]   # (T+1) × r
        SCtr = covariance_draw(diff(Ytr, dims=1), df0tr, Diagonal(SC0tr))
        Q[1:r, 1:r] = SCtr

        # Update initial covariance from stationary distribution of cycle block
        ns      = r + n*p
        vecP0f  = (I(ns^2) - kron(A, A)) \ Q[:]
        P0full  = reshape(vecP0f, ns, ns)
        P0[r+1:end, r+1:end] = P0full[r+1:end, r+1:end]

        # Store
        States[:, :, jm]     = kc.S
        Trends[:, :, jm]     = kc.S[:, 1:r] * C[:, 1:r]'
        TrendsReal[:, :, jm] = kc.S[:, 2:r] * C[:, 2:r]'
        LogLik_arr[jm]       = loglik
        SS0_arr[:, jm]       = S0[1:r]
        AA_arr[:, :, jm]     = A
        QQ_arr[:, :, jm]     = Q
        CC_arr[:, :, jm]     = C
        RR_arr[:, :, jm]     = R

        if mod(jm, 50) == 0
            elapsed = time() - t_start
            @info "Draw $jm / $ndraws  ($(round(elapsed; digits=1)) s elapsed)"
            if !isempty(notrend) && jm <= 1000
                @info "  Acceptance rate so far: $(mean(filter(!isnan, p_acc[1:jm])))"
            end
        end
    end

    # ── Discard burn-in ───────────────────────────────────────────────────────
    skip    = 1
    discard = ceil(Int, ndraws / 2)
    idx     = discard+1:ndraws

    common_trends = States[:, 1:r, idx]
    cycles        = States[:, r+1:r+n, idx]

    return (
        common_trends = common_trends,
        cycles        = cycles,
        trends        = Trends[:, :, idx],
        trends_real   = TrendsReal[:, :, idx],
        AA            = AA_arr[:, :, idx],
        QQ            = QQ_arr[:, :, idx],
        CC            = CC_arr[:, :, idx],
        RR            = RR_arr[:, :, idx],
        loglik        = LogLik_arr[idx],
        SS0           = SS0_arr[:, idx],
        p_acc         = p_acc[idx],
        ndraws        = ndraws,
        discard       = discard,
        SC0tr         = SC0tr,
        S0tr          = S0tr,
        P0tr          = P0tr,
        df0tr         = df0tr,
        Psi           = Psi,
        dates         = dates,
        Y             = Y,
        y             = y,
        mnem          = mnem,
    )
end

end # module


