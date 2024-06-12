mutable struct GaussianBasis{T} <: AbstractP4MLBasis
    ζ::Vector{T}
    # ----------------- metadata 
    @reqfields
end
 
GaussianBasis(ζ::Vector{T}) where {T} = GaussianBasis(ζ, _make_reqfields()...)

Base.length(basis::GaussianBasis) = length(basis.ζ)

Base.show(io::IO, basis::GaussianBasis) = 
    print(io, "GaussianBasis(", length(basis), ")")

_valtype(::GaussianBasis, T::Type{<: Real}) = T
_valtype(::GaussianBasis, T::Type{<: Hyper{<:Real}}) = T


function evaluate!(P, basis::GaussianBasis, x::AbstractVector)
    N = size(P, 2)
    nX = length(x)

    @inbounds begin 
        for n = 1:N
            @simd ivdep for i = 1:nX 
                P[i, n] = exp(-basis.ζ[n] * x[i]^2)
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

function evaluate_ed_dp!(P, dP, dpP, basis::GaussianBasis, x)
    N = length(basis.ζ)
    nX = length(x)

    @inbounds begin 
        for n = 1:N
            @simd ivdep for i = 1:nX 
                P[i,n] = exp(-basis.ζ[n] * x[i]^2)
                dP[i,n] = -2 * basis.ζ[n] * x[i] * P[i, n]
                dpP[i,n] = - x[i]^2 * P[i,n]
            end
        end
    end
    return P, dP, dpP
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

function ChainRulesCore.rrule(::typeof(evaluate), basis::GaussianBasis{T}, R::AbstractVector{<: Real}) where {T}
    A, dR, dζ = evaluate_ed_dp(basis, R)
    #dζ = pb_params(basis.Dn.ζ, basis, R)

    function pb(∂A)
        ∂R = similar(R)
        ∂ζ = similar(basis.Dn.ζ)
        @assert size(∂A) == (length(R), length(basis))
        for i = 1:length(R)
            ∂R[i] = dot(@view(∂A[i, :]), @view(dR[i, :]))
        end
        for i = 1:length(basis.Dn.ζ)
            ∂ζ[i] = dot(@view(∂A[:, i]), @view(dζ[:, i]))
        end
        return NoTangent(), ∂ζ, ∂R
    end
    
    return A, pb
end
