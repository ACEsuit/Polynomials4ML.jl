mutable struct SlaterBasis{N, T} <: AbstractP4MLBasis
    ζ::SVector{N, T}
end

function SlaterBasis(ζ::AbstractVector) 
    N = length(ζ); T = eltype(ζ)
    return SlaterBasis{N, T}(SVector{N, T}(ζ))
end

Base.length(basis::SlaterBasis) = length(basis.ζ)

Base.show(io::IO, basis::SlaterBasis) = print(io, "SlaterBasis($(length(basis)))")

_valtype(::SlaterBasis, T::Type{<: Real}) = T

_static_params(basis::SlaterBasis) = (ζ = basis.ζ,)

_evaluate!(P, dP, basis::SlaterBasis, x)  = 
    _evaluate!(P, dP, basis, x, _static_params(basis), nothing)


function _evaluate!(P, dP, basis::SlaterBasis, x::AbstractVector, ps, st)
    ζ = ps.ζ
    N = size(P, 2)
    nX = length(x)
    WITHGRAD = !isnothing(dP)

    @inbounds begin 
        for n = 1:N
            @simd ivdep for i = 1:nX 
                P_in = exp(- ζ[n] * x[i])
                P[i, n] = P_in
                if WITHGRAD
                    dP[i, n] = - ζ[n] * P_in
                end
            end
        end
    end

   return P, dP 
end 

#=
function evaluate_ed_dp!(P, dP, dpP, basis::SlaterBasis, x)
    N = length(basis.ζ)
    nX = length(x)

    @inbounds begin 
        for n = 1:N
            @simd ivdep for i = 1:nX 
                P[i, n] = exp(-basis.ζ[n] * x[i])
                dP[i, n] = -basis.ζ[n] * P[i,n]
                dpP[i, n] = -x[i] * P[i,n]
            end
        end
    end

    return P, dP, dpP
end 


function ChainRulesCore.rrule(::typeof(evaluate), basis::SlaterBasis{T}, R::AbstractVector{<: Real}) where {T}
    A, dR, dζ = evaluate_ed_dp(basis, R)

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

=#