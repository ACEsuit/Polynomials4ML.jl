
mutable struct GaussianBasis{N, T} <: AbstractP4MLBasis
    ζ::SVector{N, T}
end
 
function GaussianBasis(ζ::AbstractVector) 
    N = length(ζ); T = eltype(ζ)
    return GaussianBasis{N, T}(SVector{N, T}(ζ))
end

function _rand_gaussian_basis(N1 = 5, N2 = 3, T = Float64)
    Pn = legendre_basis(N1+1)
    spec = [(n1 = n1, n2 = n2, l = l) for n1 = 1:N1 for n2 = 1:N2 for l = 0:N1-1] 
    ζ = rand(length(spec))
    Dn = GaussianBasis(ζ)
    return AtomicOrbitalsRadials(Pn, Dn, spec) 
end

Base.length(basis::GaussianBasis) = length(basis.ζ)

Base.show(io::IO, basis::GaussianBasis) = 
    print(io, "GaussianBasis(", length(basis), ")")

_valtype(::GaussianBasis, T::Type{<: Real}) = T

_static_params(basis::GaussianBasis) = (ζ = basis.ζ,)

_evaluate!(P, dP, basis::GaussianBasis, x)  = 
    _evaluate!(P, dP, basis, x, _static_params(basis), nothing)

function _evaluate!(P, dP, basis::GaussianBasis, x::AbstractVector, ps, st)
    ζ = ps.ζ
    N = length(ζ)
    nX = length(x)
    WITHGRAD = !isnothing(dP)

    @inbounds begin 
        for n = 1:N
            @simd ivdep for j = 1:nX 
                p_jn = exp(- ζ[n] * x[j]^2)
                P[j,n] = p_jn
                if WITHGRAD
                    dP[j,n] = -2 * ζ[n] * x[j] * p_jn
                end
            end
        end
    end

    return nothing 
end

#=
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
=#
