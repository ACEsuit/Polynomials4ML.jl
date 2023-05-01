struct SlaterBasis
    # ----------------- metadata 
    meta::Dict{String, Any}
end

SlaterBasis(; meta = Dict{String, Any}()) = SlaterBasis(meta)

function evaluate(basis::SlaterBasis, ζ::AbstractVector{<: Number}, x::AbstractVector{<: Number}) 
    N = length(ζ)
    nX = length(x)
    P = zeros(eltype(x), nX, N)

    @inbounds begin 
        for n = 1:N
            @simd ivdep for i = 1:nX 
                P[i,n] = exp(-ζ[n] * x[i])
            end
        end
    end
    return P 
end

function evaluate_ed(basis::SlaterBasis, ζ::AbstractVector{<: Number}, x::AbstractVector{<: Number})
    N = length(ζ)
    nX = length(x)
    P = zeros(eltype(x), nX, N)
    dP = zeros(eltype(x), nX, N)

    @inbounds begin 
        for n = 1:N
            @simd ivdep for i = 1:nX 
                P[i, n] = exp(-ζ[n] * x[i])
                dP[i, n] = -ζ[n] * P[i,n]
            end
        end
    end

   return P, dP 
end 

function evaluate_ed2(basis::SlaterBasis, ζ::AbstractVector{<: Number}, x::AbstractVector{<: Number})
    N = length(ζ)
    nX = length(x)
    P = zeros(eltype(x), nX, N)
    dP = zeros(eltype(x), nX, N)
    ddP = zeros(eltype(x), nX, N)
    
    @inbounds begin 
        for n = 1:N
            @simd ivdep for i = 1:nX 
                P[i, n] = exp(-ζ[n] * x[i])
                dP[i, n] = -ζ[n] * P[i, n]
                ddP[i, n] = -ζ[n] * dP[i, n]
            end
        end
    end
   return P, dP, ddP 
end 