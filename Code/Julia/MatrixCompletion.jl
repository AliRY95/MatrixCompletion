using LinearAlgebra
using CSV, SparseArrays
using DataFrames
using Arpack, StatsBase
using Random
using JuMP
using Convex
using SCS


function create_matrices( m, r )
    U = randn( m, r )
    V = randn( m, r )
    N = randn( m, m ) * 0.1
    omega = sprand( Bool, m, m, 0.1 )
    B = omega .* ( U * V' + N )
    return omega, B
end

function matrix_completion( omega, B, l, eps )
    R = omega .* B
    Q = zeros( size( B ) )
    k = 1
    Z = zeros( size( B ) )
    tau = 0
    while true
        Z = omega .* R
        svdZ = svds( Z, nsv = 1 )[1]
        tau = svdZ.S
        deltaR = omega .* ( svdZ.S .* svdZ.U * svdZ.V' ) - Q
        
        rho = dot( deltaR, R )
        
        if rho < eps
            break
        end
        
        theta = min( 1, rho / norm( deltaR ) ^ 2 )
                
        R -= theta * deltaR
        Q += theta * deltaR
        k += 1
    end
    
    svdZ = svds( Z, nsv = l )[1]
    return  svdZ.U, svdZ.S, svdZ.V, tau
end

m = 200 
r = round( Int, m / 100 ) 
omega, B = create_matrices( m, r )
U, S, V, tau = matrix_completion( omega, B, 2, 1e-10 )  




################################################################################
S = Convex.Variable(2,2)
objective = 0.5 * opnorm((omega .* (U * S * V' - B)))  

problem = minimize( objective, tr(S) <= tau, isposdef(S), S == diagm(diag(S)) )

solve!(problem, SCS.Optimizer)  
optimized_S = Convex.evaluate(S)


squared_diffs = omega .* (U * optimized_S * V')

rmsd(squared_diffs, B; normalize=false)


l,k,p=svd(U * optimized_S * V')
frobenius_norm = sqrt(sum(k .^ 2))
threshold = 0.9 * frobenius_norm

cumulative_sum = cumsum(k .^ 2)
rank_90 = findfirst(cumulative_sum .>= threshold^2)
################################################################################
# objMin = 10.e10
# sMin = 0.
# for s = LinRange( 0., tau[1], 1000 )
#     obj = norm( omega .* ( s .* U * V' - B ) )
#     if obj < objMin
#         global objMin = obj
#         global sMin = s
#     end
# end
# frobenius_norm = sqrt( sum( sMin .^ 2 ) )
# threshold = 0.9 * frobenius_norm
# cumulative_sum = cumsum( sMin .^ 2 )
# rank_90 = findfirst( cumulative_sum .>= threshold ^ 2 )
# println( "tau = ", tau[1], " sMin = ", sMin, " Rank = ", rank_90 )
