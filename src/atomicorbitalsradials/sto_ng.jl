struct STO_NG
    # ----------------- metadata 
    meta::Dict{String, Any}
end

STO_NG(; meta = Dict{String, Any}()) = STO_NG(meta)

function evaluate(basis::STO_NG, ξ::Vector{Matrix{Float64}}, x::AbstractVector{<: Number}) 
    ζ, D = ξ[1], ξ[2]
    N, M = size(ζ)
    nX = length(x)
    P = zeros(eltype(x), nX, N)

    @inbounds begin 
        for n = 1:N
            for m = 1:M
                @simd ivdep for i = 1:nX 
                    P[i,n] += D[n,m] * exp(-ζ[n, m] * x[i]^2)
                end
            end
        end
    end

    return P # D[n,m] * exp(-[n, m] * x[i]^2)
end

function evaluate_ed(basis::STO_NG, ξ::Vector{Matrix{Float64}}, x::AbstractVector{<: Number})
    ζ, D = ξ[1], ξ[2]
    N, M = size(ζ)
    nX = length(x)
    P = zeros(eltype(x), nX, N)
    dP = zeros(eltype(x), nX, N)

    @inbounds begin 
        for n = 1:N
            for m = 1:M
                @simd ivdep for i = 1:nX 
                    Z = D[n,m] * exp(-ζ[n, m] * x[i]^2)
                    P[i,n] += Z
                    dP[i,n] += -2 * ζ[n, m] * x[i] * Z
                end
            end
        end
    end

    return P, dP 
end 

function evaluate_ed2(basis::STO_NG, ξ::Vector{Matrix{Float64}}, x::AbstractVector{<: Number})
    ζ, D = ξ[1], ξ[2]
    N, M = size(ζ)
    nX = length(x)
    P = zeros(eltype(x), nX, N)
    dP = zeros(eltype(x), nX, N)
    ddP = zeros(eltype(x), nX, N)

    @inbounds begin 
        for n = 1:N
            for m = 1:M
                @simd ivdep for i = 1:nX 
                    Z = D[n,m] * exp(-ζ[n, m] * x[i]^2)
                    dZ = -2 * ζ[n, m] * x[i] * Z
                    P[i,n] += Z
                    dP[i,n] += dZ
                    ddP[i,n] += -2 * ζ[n, m] * Z -2 * ζ[n, m] * x[i] * dZ
                end
            end
        end
    end

    return P, dP, ddP 
end 