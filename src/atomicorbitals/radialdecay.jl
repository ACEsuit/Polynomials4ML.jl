using SpecialFunctions

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
function construct_basis(ζ_raw, D_raw, decay::AbstractDecayFunction, spec_list)
    LEN, K = size(ζ_raw)
    T = promote_type(eltype(ζ_raw), eltype(D_raw))
    ζ = SMatrix{LEN, K, T}(ζ_raw)
    D = SMatrix{LEN, K, T}(D_raw)
    spec = SVector{length(spec_list)}(spec_list)
    return RadialDecay{length(spec), typeof(ζ), typeof(decay)}(ζ, D, decay, spec)
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

natural_indices(basis::RadialDecay) = basis.spec

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
                    dP[i, n] += -ζ[n, m] * dfx * a
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

function _rand_basis(N1=4, N2=3; 
    K::Int=1, 
    T::Type=Float64, 
    decay_type::AbstractDecayFunction=GaussianDecay(),
    ζinit = () -> rand(T, N1 * N2 * N1^2, K), 
    Dinit = () -> ones(T, N1 * N2 * N1^2, K))

    Pn = MonoBasis(N1 + 1)
    Ylm = real_solidharmonics(N1 - 1)
    spec_list = [(n1=n1, n2=n2, l=l, m=m) for n1 in 1:N1, n2 in 1:N2, l in 0:N1-1 for m in -l:l]
    spec = SVector{length(spec_list)}(spec_list)
    spec_ln = unique((n1=s.n1, n2=s.n2, l=s.l) for s in spec)
    Dn = construct_basis(ζinit(), Dinit(), decay_type, spec_ln)
    specidx = _specidx(spec, Pn, Dn, Ylm)

    return AtomicOrbitals{length(spec), typeof(Pn), typeof(Dn), typeof(Ylm)}(Pn, Dn, Ylm, spec, specidx)
end

_rand_gaussian_basis(N1=4, N2=3, T=Float64) = _rand_basis(N1, N2; T=T)

_rand_slater_basis(N1=4, N2=3, T=Float64) = _rand_basis(N1, N2; T=T, decay_type = SlaterDecay())

_rand_sto_basis(N1=4, N2=2, K=4, T=Float64) = _rand_basis(N1, N2; T=T, K=K, 
        ζinit = () -> rand(T, N1 * N2 * N1^2, K),
        Dinit = () -> rand(T, N1 * N2 * N1^2, K))

function _invmap(a::AbstractVector)
    inva = Dict{eltype(a), Int}()
    for i = 1:length(a) 
       inva[a[i]] = i 
    end
    return inva 
end

function _specidx(spec, Pn, Dn, Ylm)
    specidx = Vector{Tuple{Int, Int, Int}}(undef, length(spec))

    spec_Ylm = natural_indices(Ylm); inv_Ylm = _invmap(spec_Ylm)
    spec_Pn = natural_indices(Pn); inv_Pn = _invmap(spec_Pn)
    spec_Dn = natural_indices(Dn); inv_Dn = _invmap(spec_Dn)
    for (z, b) in enumerate(spec)
        specidx[z] = (inv_Pn[(n = b.n1, )], inv_Dn[(n1 = b.n1, n2 = b.n2, l = b.l)], inv_Ylm[(l=b.l, m=b.m)])
    end  
    return specidx
end
