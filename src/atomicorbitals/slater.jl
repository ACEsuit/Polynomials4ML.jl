mutable struct SlaterBasis{N, T} <: AbstractP4MLBasis
    ζ::SVector{N, T}
end

function SlaterBasis(ζ::AbstractVector) 
    N = length(ζ); T = eltype(ζ)
    return SlaterBasis{N, T}(SVector{N, T}(ζ))
end

function _rand_slater_basis(n1 = 5, n2 = 3, T = Float64)
    Pn = legendre_basis(n1+1)
    spec = [(n1 = n1, n2 = n2, l = l) for n1 = 1:n1 for n2 = 1:n2 for l = 0:n1-1] 
    ζ = rand(length(spec))
    Dn = SlaterBasis(ζ)
    return AtomicOrbitalsRadials(Pn, Dn, spec) 
end

Base.length(basis::SlaterBasis) = length(basis.ζ)

Base.show(io::IO, basis::SlaterBasis) = print(io, "SlaterBasis($(length(basis)))")

_valtype(::SlaterBasis, T::Type{<: Real}) = T

_valtype(::SlaterBasis, T::Type{<: Real}, 
         ps::Union{Nothing, @NamedTuple{}}, st) = T

_valtype(::SlaterBasis, T::Type{<: Real}, 
         ps, st) = promote_type(T, eltype(ps.ζ), )


_static_params(basis::SlaterBasis) = (ζ = basis.ζ,)

_init_luxparams(basis::SlaterBasis) = ( ζ = Vector(basis.ζ), )


_evaluate!(P, dP, basis::SlaterBasis, x)  = 
    _evaluate!(P, dP, basis, x, _static_params(basis), nothing)


function _evaluate!(P, dP, basis::SlaterBasis, x::AbstractVector, ps, st)
    ζ = ps.ζ
    N = size(P, 2)
    nX = length(x)
    WITHGRAD = !isnothing(dP)

    @inbounds begin 
        for n = 1:N
            @simd ivdep for j = 1:nX 
                P_jn = exp(- ζ[n] * x[j])
                P[j, n] = P_jn
                if WITHGRAD
                    dP[j, n] = - ζ[n] * P_jn
                end
            end
        end
    end

   return P, dP 
end 


function pullback_ps(∂P, basis::SlaterBasis, x::BATCH, ps, st)
    ζ = ps.ζ
    N = length(ζ)
    nX = length(x)

    ∂ζ = fill!(similar(ζ), 0)

    @inbounds for n = 1:N
        ζₙ = ζ[n]
        @simd ivdep for j = 1:nX 
            # P[j,n] = p_jn => ∂P[j,n]*P[j,n] = ∂P[j,n] * p_jn
            p_jn = exp(- ζ[n] * x[j])
            ∂ζ[n] += ∂P[j,n] * p_jn * (-x[j])
        end
    end

    return (ζ = ∂ζ,)
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