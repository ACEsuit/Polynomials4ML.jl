mutable struct SlaterBasis <: ScalarPoly4MLBasis
    ζ::AbstractVector
    # ----------------- metadata 
    @reqfields
end

SlaterBasis(ζ) = SlaterBasis(ζ, _make_reqfields()...)

Base.length(basis::SlaterBasis) = length(basis.ζ)

_valtype(::SlaterBasis, T::Type{<: Real}) = T
_valtype(::SlaterBasis, T::Type{<: Hyper{<:Real}}) = T
function evaluate!(P, basis::SlaterBasis, x::AbstractVector) 
    N = size(P, 2)
    nX = length(x)

    @inbounds begin 
        for n = 1:N
            @simd ivdep for i = 1:nX 
                P[i,n] = exp(-basis.ζ[n] * x[i])
            end
        end
    end
    return P 
end

function evaluate_ed!(P, dP, basis::SlaterBasis, x)
    N = size(P, 2)
    nX = length(x)

    @inbounds begin 
        for n = 1:N
            @simd ivdep for i = 1:nX 
                P[i, n] = exp(-basis.ζ[n] * x[i])
                dP[i, n] = -basis.ζ[n] * P[i,n]
            end
        end
    end

   return P, dP 
end 

function evaluate_ed2!(P, dP, ddP, basis::SlaterBasis, x)
    N = size(P, 2)
    nX = length(x)
    
    @inbounds begin 
        for n = 1:N
            @simd ivdep for i = 1:nX 
                P[i, n] = exp(-basis.ζ[n] * x[i])
                dP[i, n] = -basis.ζ[n] * P[i, n]
                ddP[i, n] = -basis.ζ[n] * dP[i, n]
            end
        end
    end
   return P, dP, ddP 
end 