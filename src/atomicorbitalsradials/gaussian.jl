struct GaussianBasis
    # ----------------- metadata 
    meta::Dict{String, Any}
end
 
GaussianBasis(; meta = Dict{String, Any}()) = GaussianBasis(meta)

function evaluate(basis::GaussianBasis, ζ::AbstractVector{<: Number}, x::AbstractVector{<: Number}) 
    N = length(ζ)
    nX = length(x)
    P = zeros(eltype(x), nX, N)

    @inbounds begin 
        for n = 1:N
            @simd ivdep for i = 1:nX 
                P[i,n] = exp(-ζ[n] * x[i]^2)
            end
        end
    end

    return P 
end

function evaluate_ed(basis::GaussianBasis, ζ::AbstractVector{<: Number}, x::AbstractVector{<: Number})
    N = length(ζ)
    nX = length(x)
    P = zeros(eltype(x), nX, N)
    dP = zeros(eltype(x), nX, N)

    @inbounds begin 
        for n = 1:N
            @simd ivdep for i = 1:nX 
                P[i,n] = exp(-ζ[n] * x[i]^2)
                dP[i,n] = -2 * ζ[n] * x[i] * P[i, n]
            end
        end
    end
    return P, dP 
end 

function evaluate_ed2(basis::GaussianBasis, ζ::AbstractVector{<: Number}, x::AbstractVector{<: Number})
    N = length(ζ)
    nX = length(x)
    P = zeros(eltype(x), nX, N)
    dP = zeros(eltype(x), nX, N)
    ddP = zeros(eltype(x), nX, N)

    @inbounds begin 
        for n = 1:N
            @simd ivdep for i = 1:nX 
                P[i, n] = exp(-ζ[n] * x[i]^2)
                dP[i, n] = -2 * ζ[n] * x[i] * P[i, n]
                ddP[i, n] = -2 * ζ[n] * P[i, n] -2 * ζ[n] * x[i] * dP[i, n]
            end
        end
    end
   return P, dP, ddP 
end 