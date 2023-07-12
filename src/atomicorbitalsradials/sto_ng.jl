struct STO_NG{T} <: ScalarPoly4MLBasis
    ζ::Tuple{Matrix{T}, Matrix{T}}
    # ----------------- metadata 
    @reqfields
end

STO_NG(ζ) = STO_NG(ζ, _make_reqfields()...)

Base.length(basis::STO_NG) = length(basis.ζ[1])

_valtype(::STO_NG, T::Type{<: Real}) = T
_valtype(::STO_NG, T::Type{<: Hyper{<:Real}}) = T

function evaluate!(P, basis::STO_NG, x::AbstractVector) 
    ζ, D = basis.ζ[1], basis.ζ[2]
    N, M = size(ζ)
    nX = length(x)
    fill!(P, zero(eltype(P)))
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

function evaluate_ed!(P, dP, basis::STO_NG, x::AbstractVector{<: Real})
    ζ, D = basis.ζ[1], basis.ζ[2]
    N, M = size(ζ)
    nX = length(x)
    fill!(P, zero(eltype(P)))
    fill!(dP, zero(eltype(dP)))
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

function evaluate_ed2!(P, dP, ddP, basis::STO_NG, x::AbstractVector{<: Real})
    ζ, D = basis.ζ[1], basis.ζ[2]
    N, M = size(ζ)
    nX = length(x)
    fill!(P, zero(eltype(P)))
    fill!(dP, zero(eltype(dP)))
    fill!(ddP, zero(eltype(ddP)))
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