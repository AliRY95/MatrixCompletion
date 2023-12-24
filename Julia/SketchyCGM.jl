using LinearAlgebra
using CSV, SparseArrays
using DataFrames
using Arpack, StatsBase
using Random
using JuMP
using Convex
using SCS , ECOS, Plots, HDF5


function create_E(B)
    E = findall(!iszero, B)
    return E
end


function orth(Y)
    F = qr(Y)
    return F.Q
end

function A_linear_map(X, E) 
    return [X[i] for i in E]
end

#Adjoint of A, places the observed entries from a vector back into a matrix
function A_adjoint(z, E, m, n)
    X = zeros(m, n)
    for (index, value) in zip(E, z)
        X[index] = value
    end
    return X
end

function SketchyCGM(mask, Data, ϵ,r, α,maxit) 
    EE= create_E(mask)
    m,n=size(mask)
    ℓ = 4*r + 3
    k = 2*r + 1
    Ω = randn(n,k)
    Ψ = randn(ℓ,m)
    Y = zeros(m,k)
    W = zeros(ℓ,n)
    z = zeros(size(EE)[1])
    t=0
    Q = zeros(m,m)
    U = zeros(m,r)
    V =zeros(n,r)
    Σ= zeros(r)
    res=[]
    while t < maxit
        fz = A_linear_map(Data, EE)-z
        u, σ, v = svds(A_adjoint(fz, EE, m, n), nsv = 1)[1]
        h = A_linear_map(-α * u * v', EE)
        if dot(z-h,fz) < ϵ
            break
        end
        push!(res,dot(z-h,fz) )
        η = 2/(t+2)
        z = (1-η)*z + η*h
        Y = (1- η)*Y + η*(-α*u)*(v'*Ω)
        W = (1- η)*W + η*(Ψ*-α*u)*v'
        Q = orth(Y)
        B = (Ψ*Q)\W
        U,Σ,V = svds(B, nsv = r)[1]
        t += 1
    end
    return Q*U, Σ, V, t, res
end


Sketchy_nmae=[]

for τ in [1, 5, 10, 15, 20, 50]
    for m in [100, 300, 500, 1000, 2000]
        Dat = createRandomMatrix(m, m, 0.1)
        Mask = sparse(Dat .!= 0)
        u,s,v,iter,resid= SketchyCGM(Mask, Dat, 1e-10,1, τ, 1000)
        push!(Sketchy_nmae, sum(abs.(Mask .*(u* s * v') .- Dat))/(sum(Mask)*(maximum(Dat)- minimum(Dat))))
        print(m)
    end
end

#########
##MovieLens data

df = CSV.File("ratings_matrix_MovieLens.csv") |> DataFrame

matrix = transpose(Array(df[:, 3:end]))

# Binarizing the values 
Data_binarized = ifelse.(matrix .> 3.5, 1, ifelse.(matrix .> 0, -1, 0))

mask = convert(Array{Int, 2}, Data_binarized .!= 0)

uu,ss,vv,it= SketchyCGM(mask, Data_binarized, 1e-10, 1, 20,5000)

pred = uu * Diagonal(ss) * vv'

Pred_binarized = ifelse.(pred .> 0, 1, -1)


