struct GaussianBasis <: AbstractPoly4MLBasis
    # ----------------- metadata 
    @reqfields
end
 
GaussianBasis() = GaussianBasis(_make_reqfields()...)

_valtype(::GaussianBasis, T::Type{<: Real}) = T

function evaluate(basis::GaussianBasis, ζ::AbstractVector{<: Number}, x::AbstractVector{<: Number}) 
    N = length(ζ)
    nX = length(x)
    P = acquire!(basis.pool, :P, (nX, N), eltype(x))
    fill!(P, 0)

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
    P = acquire!(basis.pool, :P, (nX, N), eltype(x))
    dP = acquire!(basis.pool, :dP, (nX, N), eltype(x))
    fill!(P, 0)
    fill!(dP, 0)

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

    P = acquire!(basis.pool, :P, (nX, N), eltype(x))
    dP = acquire!(basis.pool, :dP, (nX, N), eltype(x))
    ddP = acquire!(basis.pool, :ddP, (nX, N), eltype(x))
    fill!(P, 0)
    fill!(dP, 0)
    fill!(ddP, 0)

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