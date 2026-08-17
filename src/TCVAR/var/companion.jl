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
