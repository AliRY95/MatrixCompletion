using LinearAlgebra
using CSV, SparseArrays
using DataFrames
using Arpack, StatsBase
using Random
using JuMP
using Convex
using SCS , ECOS, Optim
using Plots

function createRandomMatrix(rowN, colN, sparsity)
    trueRank = ceil(Int, rowN / 100)
    U = randn(rowN, trueRank)
    V = randn(colN, trueRank)
    N = randn(rowN, colN)

    omega = ones(Bool, rowN, colN)
    numZeros = ceil(Int, (1 - sparsity) * rowN * colN)
    omega[1:numZeros] .= false
    shuffle!(omega)

    sparseData = sparse(omega .* (U * V' + 0.1 * N))
    return sparseData
end


function solver(U, V, B, Omega, τ)
    m, n = size(U)
    model = Model(SCS.Optimizer)
    @variable(model, S[1:n, 1:n], PSD)

    for i in 1:n
        for j in 1:n
            if i != j
                @constraint(model, S[i, j] == 0)
            end
        end
    end

    USVt = U * Diagonal(S) * V'

    error_term = Omega .* (USVt - B)
    sum_of_squares = sum(error_term[i, j]^2 for i in 1:m, j in 1:n)

    @objective(model, Min, 0.5 * sum_of_squares)
    @constraint(model, tr(S) <= τ)

    optimize!(model)

    xv = value.(S)
    return xv
end

function matrix_completion(Ω, B, ℓ, ε, τ, maxit)
    R = Ω .* B
    Q = zeros(size(B))
    k = 1
    Z=zeros(size(B))
    ρ = 1e7
    rhos=[]
    while k < maxit
        Z = Ω .* R
        u, σ, v = svds(Z, nsv = 1)[1]

        ΔR = Ω .* ( τ * u * v') - Q
        
        ρ = dot(ΔR, R)

        if ρ < ε
            break
        end
        
        θ = min(1, ρ / (norm(ΔR))^2)
                
        R -= θ * ΔR
        Q += θ * ΔR
        k += 1
        push!(rhos,ρ)
    end

    U,Σ,V = svds(Z, nsv = ℓ)[1]

    # function objective(S::Vector{Float64})
    #     0.5 * norm(Ω .* (U * Diagonal(S) * V' - B))^2
    # end

    # lb = zeros(ℓ)
    # ub = τ * ones(ℓ)

    # S0 = zeros(ℓ)
    # opt_result = optimize(objective, lb, ub, S0, Fminbox())
    # S = Optim.minimizer(opt_result)
    S = solver(U, V, B, Ω, τ)
    return  U,Σ,V,k,rhos
end


Dat = createRandomMatrix(2000, 2000, 0.1)
Mask = sparse(Dat .!= 0)

@time u,s,v,iter,resid= matrix_completion(Mask, Dat, 1, 1e-10, 30, 100)

l,k,p=svd(u * s * v')
frobenius_norm = sqrt(sum(k .^ 2))
threshold = 0.9 * frobenius_norm1
cumulative_sum = cumsum(k .^ 2)
rank_90 = findfirst(cumulative_sum .>= threshold^2)

NMAE = sum(abs.(Mask .*(u* s * v') .- Dat))/(sum(Mask)*(maximum(Dat)- minimum(Dat)))


Dual_nmae=[]

for τ in [1, 5, 10, 15, 20, 50]
    for m in [100, 300, 500, 1000, 2000]
        Dat = createRandomMatrix(m, m, 0.1)
        Mask = sparse(Dat .!= 0)
        u,s,v,iter,resid= matrix_completion(Mask, Dat, 1, 1e-10, τ, 1000)
        push!(Dual_nmae, sum(abs.(Mask .*(u* s * v') .- Dat))/(sum(Mask)*(maximum(Dat)- minimum(Dat))))
    end
end


##### MovieLens Dataset

df = CSV.File("ratings_matrix_MovieLens.csv") |> DataFrame

matrix = transpose(Array(df[:, 3:end]))

# Binarizing the values 
Data_binarized = ifelse.(matrix .> 3.5, 1, ifelse.(matrix .> 0, -1, 0))

# Construct the mask matrix
mask_matrix = convert(Array{Int, 2}, Data_binarized .!= 0)


u_m,s_m,v_m,iter_m,resid_m= matrix_completion(mask_matrix, Data_binarized, 1, 1e-10, 50, 1000)

######### Another setup for the same problem


function matrix_completion(Ω, B, ℓ, ε, τ, maxit)
    R = Ω .* B
    Q = zeros(size(B))
    k = 1
    Z=zeros(size(B))
    ρ = 1e7
    rhos=[]
    while k < maxit
        Z = Ω .* R
        u, σ, v = svds(Z, nsv = 1)[1]

        ΔR = Ω .* ( τ * u * v') - Q
        
        ρ = dot(ΔR, R)

        if ρ < ε
            break
        end
        
        θ = min(1, ρ / (norm(ΔR))^2)
                
        R -= θ * ΔR
        Q += θ * ΔR
        k += 1
        push!(rhos,ρ)
    end

    U,Σ,V = svds(Z, nsv = ℓ)[1]




    indices = findall(!iszero, B)
    observedData_obj = B[indices]
    UVT = zeros(length(indices), ℓ)
    for i = 1:ℓ
        temp = U[:, i] * V[:, i]'
        UVT[:, i] = temp[indices]
    end
    
    # Linear constraint trace(S) < tau

    A = zeros(ℓ, ℓ)

    for i = 1:ℓ
        A[i, i] = 1
    end
    
    for i = 1:ℓ - 1
        A[i + 1, i] = -1
    end
    
    A[1, :] .= 1
    
    b = zeros(ℓ)
    b[1] = τ
    
    model = Model(Ipopt.Optimizer)
    @variable(model, S[1:ℓ] >= 0)
    @objective(model, Min, 0.5 * sum((UVT * S - observedData_obj).^2))
    @constraint(model, A * S .<= b)
    
    # Initial guess
    set_start_value.(S, zeros(ℓ))
    set_start_value(S[1], τ)
    
    # Solve the problem
    optimize!(model)
    
    # Get the solution
    S_opt = value.(S)
    return  U,S_opt,V,k,rhos
end