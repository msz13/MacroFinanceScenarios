using StatsBase
using PrettyTables
using TimeSeries


# max_drawdown_and_length, annualise and print_percentiles (now period_percentiles) moved
# to src/ScenariosEvaluation/evaluation.jl.

function returns_summarystats(data::TimeArray,t)
    names = colnames(data)
    returns = transpose(values(data))
    n_assets = size(returns)[1]
    n_digits = 4

    
    stats = [ Dict(
        :mean => round(mean(returns[i,:]) * t, digits=n_digits), 
        :std => round(std(returns[i,:]) * t^0.5, digits=n_digits),
        :median => round(median(returns[i,:]) * t, digits=n_digits),
        :skewness => round(skewness(returns[i,:]), digits=n_digits),
        :kurtosis => round(kurtosis(returns[i,:]), digits=n_digits),
        :autocor => round(autocor(returns[i,:],[1])[1], digits=n_digits),
        :p25th => round(percentile(returns[i,:],25) * t, digits=n_digits),
        :p75th => round(percentile(returns[i,:],75) * t, digits=n_digits),
        :min => round(minimum(returns[i,:]) * t, digits=n_digits),
        :max => round(maximum(returns[i,:]) * t, digits=n_digits),
        :sr => round((mean(returns[i,:]) * t)/(std(returns[i,:]) * t^0.5), digits=n_digits)        
        ) for i in 1:n_assets ]
        

    short_stats = pretty_table(stats, backend = Val(:html), row_labels = names)
    return short_stats
end

function cor_returns(returns:: TimeArray)
    col = colnames(returns)
    corr = cor(values(returns))
    return pretty_table(corr, column_labels=col, backend = Val(:html), row_labels=col)
end

function sum_returns_between_periods(scenarios::Matrix{Float64}, periods::Vector{Int})
    # Validate inputs
    n_periods, n_scenarios = size(scenarios)
    length(periods) < 2 && error("At least two periods are required")
    all(1 .<= periods .<= n_periods) || error("Invalid period indices")
    issorted(periods) || error("Periods must be sorted in ascending order")
    
    # Initialize output matrix: rows = number of period intervals, cols = number of assets
    n_intervals = length(periods) - 1
    result = zeros(Float64, n_intervals, n_scenarios)
    
    #Sum returns for each interval and asset
    for i in 1:n_intervals
        start_idx = periods[i]
        end_idx = periods[i+1]
        result[i, :] = sum(scenarios[start_idx:end_idx, :], dims=1)
    end
    
    return result
end


function cum_returns_in_periods(scenarios, periods, freq, annualise=false)
    
    n_assets, n_steps, n_scenarios = size(scenarios)
    n_periods = length(periods)

    result = zeros(Float64, n_assets, n_periods, n_scenarios)

    for a in 1:n_assets
        cum_ret  = cumsum(scenarios[a, :, :], dims=1)
        result[a, :, :] =  cum_ret[freq * periods, :] 
        if annualise
            result[a, :, :] = result[a, :, :] ./ periods
        end 
    end

    return result

end

function print_scenarios_summary(scenarios:: Array{Float64, 3}, assets_names, periods)
    n_assets, n_periods, _ =  size(scenarios)
   
    means = zeros(n_periods, n_assets)
    stds = zeros(n_periods, n_assets)
    skew = zeros(n_periods, n_assets)
    kurt = zeros(n_periods, n_assets)

    for a in 1:n_assets
    
        means[:,a] = mean(scenarios[a,:,:], dims=2)
        stds[:,a] = std(scenarios[a, :,:], dims=2)
        
        for t in 1:n_periods
        skew[t,a] = skewness(scenarios[a, t, :])
        end
        
        for t in 1:n_periods
        kurt[t,a] = kurtosis(scenarios[a, t,:])
        end 

    end

    pretty_table(round.(means, digits=4), backend = Val(:html), column_labels=assets_names, row_labels = periods, title="Means")
    pretty_table(round.(stds, digits=4), backend = Val(:html), column_labels=assets_names, row_labels = periods, title="Standard devations")
    pretty_table(round.(skew, digits=4), backend = Val(:html), column_labels=assets_names, row_labels = periods, title="Skewness")
    pretty_table(round.(kurt, digits=4), backend = Val(:html), column_labels=assets_names, row_labels = periods, title="Kurtosis")

end


function print_scenarios_percentiles(scenarios, perc, periods_names, title="")
    years = size(scenarios, 2)
    simulation_perc = zeros(years, length(perc))

    for t in 1:years
        simulation_perc[t,:] = quantile(scenarios[:,t],perc)
    end
    pretty_table(round.(simulation_perc, digits=4); backend = :html, column_labels=perc, row_labels=periods_names, title=title)
end

function girf(B::Matrix{Float64}, Σ::Matrix{Float64}, h::Int, shock_var::Int, shock_value)
    """
    Generalized Impulse Response Function (GIRF) for VAR models
    
    Parameters:
    -----------
    B : Matrix{Float64}
        VAR coefficient matrix of size (K, K*p) where K is number of variables
        and p is the lag order. Contains [A₁ A₂ ... Aₚ]
    Σ : Matrix{Float64}
        Residual covariance matrix of size (K, K)
    h : Int
        Number of periods (horizon) for impulse response
    shock_var : Int
        Index of the variable to be shocked (1 to K)
    
    Returns:
    --------
    Matrix{Float64}
        GIRF matrix of size (h+1, K) where each row represents the response
        at time t, and each column represents a different variable
    """
    
    K = size(Σ, 1)  # Number of variables
    p = size(B, 2) ÷ K  # Lag order
    
    # Initialize GIRF matrix
    girf_matrix = zeros(h + 1, K)
    
    # Shock size: one standard deviation shock scaled by covariance
    σⱼ = sqrt(Σ[shock_var, shock_var])
    eⱼ = zeros(K)
    eⱼ[shock_var] = shock_value
    
    # Generalized impulse: Σ * eⱼ / σⱼ
    shock = Σ * eⱼ / σⱼ
    
    # Period 0: immediate impact
    girf_matrix[1, :] = shock
    
    # Compute companion form if p > 1
    if p > 1
        # Companion matrix
        F = zeros(K * p, K * p)
        F[1:K, :] = B
        F[K+1:end, 1:K*(p-1)] = I(K * (p - 1))
        
        # Extended shock vector
        shock_extended = vcat(shock, zeros(K * (p - 1)))
        
        # Iterate through horizons
        state = shock_extended
        for t in 1:h
            state = F * state
            girf_matrix[t + 1, :] = state[1:K]
        end
    else
        # Simple VAR(1) case
        state = shock
        for t in 1:h
            state = B * state
            girf_matrix[t + 1, :] = state
        end
    end
    
    return girf_matrix
end

function calculate_equity_returns(div_growth, dp)
    
    pd_growth = diff(-dp, dims=1)

    return  pd_growth .+ div_growth[2:end,:] ./100 + log.(1 .+ exp.(dp[2:end,:]))
       

end


# calculate_bond_returns lives in src/common/bond_returns.jl (shared with IbbotsonSinquefield)