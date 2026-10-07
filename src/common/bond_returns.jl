"""
yelds_scenarios: matrix of yelds returns 
T: maturity
t: frequency of analises
"""
function calculate_bond_returns(yelds_scenarios, T, t)

    yt = yelds_scenarios[2:end,:]
    ytm1 = yelds_scenarios[1:end-1,:]

    A = ytm1 ./ t
    C = 1 ./( (1 .+ yt ./2).^(2*(T-1 ./ t)))
    B = ytm1 ./ yt .* (1 .- C)
    
    return A .+ B .+ C .-1 
end
