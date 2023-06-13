struct GaussianBasis <: ScalarPoly4MLBasis
    ζ::AbstractVector
    # ----------------- metadata 
    @reqfields
end
 
GaussianBasis(ζ) = GaussianBasis(ζ, _make_reqfields()...)

Base.length(basis::GaussianBasis) = length(basis.ζ)

_valtype(::GaussianBasis, T::Type{<: Real}) = T
_valtype(::GaussianBasis, T::Type{<: Hyper{<:Real}}) = T
function evaluate!(P, basis::GaussianBasis, x::AbstractVector) 
    N = size(P, 2)
    nX = length(x)

    @inbounds begin 
        for n = 1:N
            @simd ivdep for i = 1:nX 
                P[i,n] = exp(-basis.ζ[n] * x[i]^2)
            end
        end
    end

    return P 
end

function evaluate_ed!(P, dP, basis::GaussianBasis, x)
    N = length(basis.ζ)
    nX = length(x)

    @inbounds begin 
        for n = 1:N
            @simd ivdep for i = 1:nX 
                P[i,n] = exp(-basis.ζ[n] * x[i]^2)
                dP[i,n] = -2 * basis.ζ[n] * x[i] * P[i, n]
            end
        end
    end
    return P, dP 
end 

function evaluate_ed2!(P, dP, ddP, basis::GaussianBasis, x)
    N = length(basis.ζ)
    nX = length(x)

    @inbounds begin 
        for n = 1:N
            @simd ivdep for i = 1:nX 
                P[i, n] = exp(-basis.ζ[n] * x[i]^2)
                dP[i, n] = -2 * basis.ζ[n] * x[i] * P[i, n]
                ddP[i, n] = -2 * basis.ζ[n] * P[i, n] -2 * basis.ζ[n] * x[i] * dP[i, n]
            end
        end
    end
   return P, dP, ddP 
end 