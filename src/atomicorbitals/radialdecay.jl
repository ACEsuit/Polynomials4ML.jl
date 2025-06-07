struct GaussianDecay <: AbstractDecayFunction
end

struct SlaterDecay <: AbstractDecayFunction
end

# Gaussian: f(x) = x^2, df/dx = 2x
(f::GaussianDecay)(x) = x^2
df(f::GaussianDecay, x) = 2x

# Slater: f(x) = x, df/dx = 1
(f::SlaterDecay)(x) = x
df(f::SlaterDecay, x::T) where T = one(T)

"""
    construct_basis(RadialDecay, ζ_raw, D_raw, decay)

Construct a `RadialDecay` object from raw matrix data `ζ_raw`, `D_raw` 
and a `DecayFunction(f, df)` representing the decay form and its derivative.
All input is converted to statically-sized `SMatrix` for efficiency.
"""
function construct_basis(::Type{RadialDecay}, ζ_raw, D_raw, decay::AbstractDecayFunction)
    LEN, K = size(ζ_raw)
    T = promote_type(eltype(ζ_raw), eltype(D_raw))
    ζ = SMatrix{LEN, K, T}(ζ_raw)
    D = SMatrix{LEN, K, T}(D_raw)
    return RadialDecay{typeof(ζ), typeof(decay)}(ζ, D, decay)
end

"""
    construct_basis(RadialDecay, ζ_vec, D_vec, decay)

Overload for vector input. Converts to N×1 matrix and calls the full constructor.
"""
function construct_basis(::Type{RadialDecay}, ζ_vec::AbstractVector, D_vec::AbstractVector, decay::AbstractDecayFunction)
    @assert length(ζ_vec) == length(D_vec)
    N = length(ζ_vec)
    ζ = reshape(ζ_vec, N, 1)
    D = reshape(D_vec, N, 1)
    return construct_basis(RadialDecay, ζ, D, decay)
end

"""
    construct_basis(RadialDecay, ζ_vec, decay)

Simplified overload where D is taken as all ones.
"""
function construct_basis(::Type{RadialDecay}, ζ_vec::AbstractVector, decay::AbstractDecayFunction)
    N = length(ζ_vec)
    ζ = reshape(ζ_vec, N, 1)
    D = ones(eltype(ζ_vec), N, 1)
    return construct_basis(RadialDecay, ζ, D, decay)
end

Base.length(basis::RadialDecay) = size(basis.ζ, 1)

_valtype(::RadialDecay, T::Type{<: Real}) = T

_valtype(::RadialDecay, T::Type{<: Real}, 
         ps::Union{Nothing, @NamedTuple{}}, st) = T

_valtype(::RadialDecay, T::Type{<: Real}, 
         ps, st) = promote_type(T, eltype(ps.ζ), eltype(ps.D))

_static_params(basis::RadialDecay) = (ζ = basis.ζ, D = basis.D)

_init_luxparams(basis::RadialDecay) = 
            ( ζ = Matrix(basis.ζ), D = Matrix(basis.D) )

_evaluate!(P, dP, basis::RadialDecay, x)  = 
    _evaluate!(P, dP, basis, x, _static_params(basis), nothing)

function _evaluate!(P, dP, basis::RadialDecay, x::AbstractVector, ps, st) 
    ζ, D = ps.ζ, ps.D
    N, K = size(ζ)
    nX = length(x)
    WITHGRAD = !isnothing(dP)

    fill!(P, zero(eltype(P)))
    if WITHGRAD
        fill!(dP, zero(eltype(dP)))
    end

    decay = basis.decay

    @inbounds begin 
        for n = 1:N, m = 1:K
            @simd ivdep for i = 1:nX 
                fx = decay(x[i])
                a = D[n, m] * exp(-ζ[n, m] * fx)
                P[i, n] += a 
                if WITHGRAD
                    dfx = df(decay, x[i])
                    dP[i, n] += - ζ[n, m] * dfx * a
                end
            end
        end
    end

    return nothing 
end

function pullback_ps(∂P, basis::RadialDecay, x::BATCH, ps, st)
    ζ, D = ps.ζ, ps.D
    decay = basis.decay
    N, K = size(ζ)
    nX = length(x)

    ∂ζ = fill!(similar(ζ), 0)
    ∂D = fill!(similar(D), 0)

    @inbounds for n = 1:N, m = 1:K
        @simd ivdep for j = 1:nX 
            fx  = decay(x[j])           # f(x[j])
            dfx = df(decay, x[j])       # df/dx
            a1  = exp(-ζ[n, m] * fx)
            ∂ζ[n, m] += ∂P[j, n] * D[n, m] * a1 * (-fx)
            ∂D[n, m] += ∂P[j, n] * a1
        end
    end
    return (ζ = ∂ζ, D = ∂D)
end


function construct_gaussian(ζ_vec::AbstractVector)
    return construct_basis(RadialDecay, ζ_vec, GaussianDecay())
end

function construct_slater(ζ_vec::AbstractVector)
    return construct_basis(RadialDecay, ζ_vec, SlaterDecay())
end

function construct_sto_ng(ζ::AbstractMatrix, D::AbstractMatrix)
    return construct_basis(RadialDecay, ζ, D, GaussianDecay())
end

function _rand_gaussian_basis(N1 = 4, N2 = 3, T = Float64)
    Pn = legendre_basis(N1 + 1)
    spec_list = [(n1 = n1, n2 = n2, l = l) for n1 in 1:N1, n2 in 1:N2, l in 0:N1-1]
    spec = SVector{length(spec_list)}(spec_list)
    ζ = rand(T, length(spec))
    Dn = construct_gaussian(ζ)
    return AtomicOrbitalsRadials{length(spec), typeof(Pn), typeof(Dn)}(Pn, Dn, spec)
end

function _rand_slater_basis(N1 = 4, N2 = 3, T = Float64)
    Pn = legendre_basis(N1 + 1)
    spec_list = [(n1 = n1, n2 = n2, l = l) for n1 in 1:N1, n2 in 1:N2, l in 0:N1-1]
    spec = SVector{length(spec_list)}(spec_list)
    ζ = rand(T, length(spec))
    Dn = construct_slater(ζ)
    return AtomicOrbitalsRadials{length(spec), typeof(Pn), typeof(Dn)}(Pn, Dn, spec)
end

function _rand_sto_basis(n1 = 4, n2 = 2, K = 4, T = Float64)
    Pn = legendre_basis(n1 + 1)
    spec_list = [(n1 = n1, n2 = n2, l = l) for n1 in 1:n1, n2 in 1:n2, l in 0:n1-1]
    spec = SVector{length(spec_list)}(spec_list)
    ζ = fill(T(0.5), length(spec), K)
    D = fill(T(0.5), length(spec), K)
    Dn = construct_sto_ng(ζ, D)
    return AtomicOrbitalsRadials{length(spec), typeof(Pn), typeof(Dn)}(Pn, Dn, spec)
end
