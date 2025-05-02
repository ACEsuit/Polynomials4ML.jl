mutable struct STO_NG{TSMAT} <: AbstractP4MLBasis
    ζ::TSMAT
    D::TSMAT
end

function STO_NG(ζ::AbstractMatrix, D::AbstractMatrix) 
    LEN, K = size(ζ)
    T = eltype(ζ)
    @assert size(D) == (LEN, K)
    sζ = SMatrix{LEN, K, T}(ζ)
    sD = SMatrix{LEN, K, T}(D)
    return STO_NG{typeof(ζ)}(sζ, sD)
end

function _rand_sto_basis(n1 = 4, n2 = 2, K = 4, T = Float64)
    Pn = legendre_basis(n1+1)
    spec = [(n1 = n1, n2 = n2, l = l) for n1 = 1:n1 for n2 = 1:n2 for l = 0:n1-1] 
    ζ = rand(length(spec), K) .= 0.5
    D = rand(length(spec), K) .= 0.5
    Dn = STO_NG(ζ, D)
    return AtomicOrbitalsRadials(Pn, Dn, spec) 
end


Base.length(basis::STO_NG) = size(basis.ζ, 1)

Base.show(io::IO, basis::STO_NG) = print(io, "STO_NG$(size(basis.ζ))")

_valtype(::STO_NG, T::Type{<: Real}) = T

_valtype(::STO_NG, T::Type{<: Real}, 
         ps::Union{Nothing, @NamedTuple{}}, st) = T

_valtype(::STO_NG, T::Type{<: Real}, 
         ps, st) = promote_type(T, eltype(ps.ζ), eltype(ps.D))

_static_params(basis::STO_NG) = (ζ = basis.ζ, D = basis.D)

_init_luxparams(basis::STO_NG) = 
            ( ζ = Matrix(basis.ζ), D = Matrix(basis.D) )


_evaluate!(P, dP, basis::STO_NG, x)  = 
    _evaluate!(P, dP, basis, x, _static_params(basis), nothing)

function _evaluate!(P, dP, basis::STO_NG, x::AbstractVector, ps, st) 
    ζ, D = ps.ζ, ps.D
    N, K = size(ζ)
    nX = length(x)
    WITHGRAD = !isnothing(dP)

    fill!(P, zero(eltype(P)))
    if WITHGRAD
        fill!(dP, zero(eltype(dP)))
    end

    @inbounds begin 
        for n = 1:N, m = 1:K
            @simd ivdep for i = 1:nX 
                a = D[n, m] * exp(-ζ[n, m] * x[i]^2)
                P[i, n] += a 
                if WITHGRAD
                    dP[i, n] += -2 * ζ[n, m] * x[i] * a
                end
            end
        end
    end

    return nothing 
end


function pullback_ps(∂P, basis::STO_NG, x::BATCH, ps, st)
    ζ, D = ps.ζ, ps.D
    N, K = size(ζ)
    nX = length(x)

    ∂ζ = fill!(similar(ζ), 0)
    ∂D = fill!(similar(D), 0)

    @inbounds for n = 1:N, m = 1:K
        @simd ivdep for j = 1:nX 
            # P[i, n] += D[n, m] * exp(-ζ[n, m] * x[j]^2)
            a1 = exp(-ζ[n, m] * x[j]^2)
            ∂ζ[n, m] += ∂P[j, n] * D[n, m] * a1 * (-x[j]^2)
            ∂D[n, m] += ∂P[j, n] * a1
        end
    end

    return (ζ = ∂ζ, D = ∂D)
end



#=
function evaluate_ed2!(P, dP, ddP, basis::STO_NG, x::AbstractVector{<: Real})
    ζ, D = basis.ζ[1], basis.ζ[2]
    N = size(ζ, 1)
    nX = length(x)
    fill!(P, zero(eltype(P)))
    fill!(dP, zero(eltype(dP)))
    fill!(ddP, zero(eltype(ddP)))
    @inbounds begin 
        for n = 1:N
            for m = 1:length(D[n])
                @simd ivdep for i = 1:nX 
                    Z = D[n][m] * exp(-ζ[n][m] * x[i]^2)
                    dZ = -2 * ζ[n][m] * x[i] * Z
                    P[i,n] += Z
                    dP[i,n] += dZ
                    ddP[i,n] += -2 * ζ[n][m] * Z -2 * ζ[n][m] * x[i] * dZ
                end
            end
        end
    end

    return P, dP, ddP 
end 

=#