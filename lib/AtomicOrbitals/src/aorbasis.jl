abstract type AbstractDecayFunction end

const NT_NNL = NamedTuple{(:n1, :n2, :l), Tuple{Int, Int, Int}}
const NT_NNLM = NamedTuple{(:n1, :n2, :l, :m), Tuple{Int, Int, Int, Int}}

struct RadialDecay{LEN, TSMAT, DF<:AbstractDecayFunction} <: AbstractP4MLBasis
    ζ::TSMAT
    D::TSMAT
    decay::DF
    spec::SVector{LEN, NT_NNL}
end

mutable struct AtomicOrbitals{LEN, TP, TD, TY}  <: AbstractP4MLBasis
   Pn::TP
   Dn::TD
   Ylm::TY
   spec::SVector{LEN, NT_NNLM}
   specidx::Vector{Tuple{Int64, Int64, Int64}}
end

function AtomicOrbitals(Pn, Dn, Ylm, spec::AbstractVector{NT_NNLM}, specidx)
    LEN = length(spec)
    return AtomicOrbitals{LEN, typeof(Pn), typeof(Dn), typeof(Ylm)}(Pn, Dn, Ylm, SVector{LEN, NT_NNLM}(spec), specidx)
end

Base.length(basis::AtomicOrbitals) = length(basis.spec)

natural_indices(basis::AtomicOrbitals) = basis.spec

# the angular `Ylm` is a SpheriCart harmonics basis (param-free) used via the
# ACEbase interface, so its value type is derived here rather than through the
# P4ML `_valtype` machinery.
_ylm_valtype(::Union{SolidHarmonics, SphericalHarmonics},
             ::Type{<: SVector{3, S}}) where {S} = S
_ylm_valtype(::Union{ComplexSolidHarmonics, ComplexSphericalHarmonics},
             ::Type{<: SVector{3, S}}) where {S} = Complex{S}

_valtype(basis::AtomicOrbitals, T::Type{<: SVector{3, S}}) where {S} =
        promote_type(_valtype(basis.Pn, S), _valtype(basis.Dn, S), _ylm_valtype(basis.Ylm, T))

_valtype(basis::AtomicOrbitals, T::Type{<: SVector{3, S}},
            ps::Union{Nothing, @NamedTuple{}}, st) where {S} =
        promote_type(_valtype(basis.Pn, S), _valtype(basis.Dn, S), _ylm_valtype(basis.Ylm, T))

_valtype(basis::AtomicOrbitals, T::Type{<: SVector{3, S}}, ps, st) where {S} =
        promote_type(_valtype(basis.Pn, S, ps.Dn, st.Dn),
                     _valtype(basis.Dn, S, ps.Dn, st.Dn),
                     _ylm_valtype(basis.Ylm, T))

_generate_input(basis::AtomicOrbitals) = @SVector randn(3)

Base.show(io::IO, basis::AtomicOrbitals) = 
        print(io, "AtomicOrbitals($(basis.Pn), $(typeof(basis.Dn.decay).name.name), $(basis.Ylm))")

# Type of atomic orbital type basis sets         

include("radialdecay.jl")

# _static_params is used to extract parameters from the basis set when 
# the basis is evaluated with the old parameter-free convention. In that case, 
# the internally stored parameters are used. 
#
# _init_luxparams is used to initialize parameters in the lux style, as a 
# NamedTuple. This is used when the basis as a learnable Lux layer.

_static_params(basis::AbstractP4MLBasis) = NamedTuple() 

# `Ylm` carries no P4ML parameters/state (it is a bare SpheriCart basis)
_static_params(basis::AtomicOrbitals) =
        (Pn = _static_params(basis.Pn), Dn = _static_params(basis.Dn), Ylm = NamedTuple())

_init_luxparams(rng::Random.AbstractRNG, l::AtomicOrbitals) =
        ( Pn = _init_luxparams(rng, l.Pn),
          Dn = _init_luxparams(rng, l.Dn),
          Ylm = NamedTuple())

_init_luxstate(rng::Random.AbstractRNG, l::AtomicOrbitals) =
        ( Pn = _init_luxstate(rng, l.Pn),
          Dn = _init_luxstate(rng, l.Dn),
          Ylm = NamedTuple())

# -------- Evaluation Code 

_evaluate!(Rnlm, dRnlm, basis::AtomicOrbitals, X) = 
            _evaluate!(Rnlm, dRnlm, basis, X, 
                       _static_params(basis), 
                       (Pn = nothing, Dn = nothing, Ylm = nothing))

function _evaluate!(Rnl, dRnl, basis::AtomicOrbitals, X::AbstractVector{<: SVector{3}}, 
                     ps, st)
    nR = length(X)
    WITHGRAD = !isnothing(dRnl)

    fill!(Rnl, zero(eltype(Rnl)))
    WITHGRAD && fill!(dRnl, zero(eltype(dRnl)))

    @no_escape begin
        TR = eltype(eltype(X))
        R = @alloc(TR, nR)
        map!(norm, R, X)
        # `Ylm` is a SpheriCart basis used via the ACEbase in-place interface;
        # buffers are taken from the Bumper stack to keep this allocation-free.
        TY = _ylm_valtype(basis.Ylm, eltype(X))
        nY = length(basis.Ylm)
        if WITHGRAD
            # this is a hack that circumvents an unexplained allocation in
            # the @withalloc macro
            T = promote_type(eltype(Rnl), TR)
            Pn = @alloc(T, nR, length(basis.Pn))
            dPn = @alloc(T, nR, length(basis.Pn))
            _evaluate!(Pn, dPn, basis.Pn, R, ps.Pn, st.Pn)
            Dn = @alloc(T, nR, length(basis.Dn))
            dDn = @alloc(T, nR, length(basis.Dn))
            _evaluate!(Dn, dDn, basis.Dn, R, ps.Dn, st.Dn)
            Ylm = @alloc(TY, nR, nY)
            dYlm = @alloc(SVector{3, TY}, nR, nY)
            evaluate_ed!(Ylm, dYlm, basis.Ylm, X)
        else
            Pn = @withalloc evaluate!(basis.Pn, R, ps.Pn, st.Pn)   # Pn(r)
            Dn = @withalloc evaluate!(basis.Dn, R, ps.Dn, st.Dn)   # Dn(r)  (ζ are the parameters -> reorganize the Lux way)
            Ylm = @alloc(TY, nR, nY)
            evaluate!(Ylm, basis.Ylm, X)
            dPn = nothing
            dDn = nothing
            dYlm = nothing
        end
        for (i, b) in enumerate(basis.specidx)
            @simd ivdep for j = 1:nR
                Rnl[j, i] = Pn[j, b[1]] * Dn[j, b[2]] * Ylm[j, b[3]]
                if WITHGRAD 
                    drj = X[j] / R[j]
                    dRnl[j, i] = ( dPn[j, b[1]] * drj * Dn[j, b[2]] * Ylm[j, b[3]] + 
                                    Pn[j, b[1]] * dDn[j, b[2]] * drj * Ylm[j, b[3]] + 
                                    Pn[j, b[1]] * Dn[j, b[2]] * dYlm[j, b[3]])
                end
            end
        end
    end

    return nothing 
end

function pullback_ps(∂Rnl, basis::AtomicOrbitals, X::AbstractVector{<: SVector{3}},
                     ps::NamedTuple, st)
    TR = eltype(eltype(X))
    T = promote_type(eltype(∂Rnl), TR) 
    nR = length(X)
    R = zeros(T, nR)
    map!(norm, R, X)

    # Rnl = output of evaluate(basis, X, ...)
    Pn = evaluate(basis.Pn, R, ps.Pn, st.Pn)
    Dn = evaluate(basis.Dn, R, ps.Dn, st.Dn)
    Ylm = evaluate(basis.Ylm, X)   # SpheriCart basis, param-free
    ∂Pn = zeros(T, size(Pn))
    ∂Dn = zeros(T, size(Dn))

    for (i, b) in enumerate(basis.specidx)
        @simd ivdep for j = 1:nR
            #             Rnl[j, i] =             Pn[j, b.n1] * Dn[j, i]
            # ∂Rnl[j,i] * Rnl[j, i] = ∂Rnl[j,i] * Pn[j, b.n1] * Dn[j, i]
            ∂Pn[j, b[1]] += ∂Rnl[j, i] * Dn[j, b[2]] * Ylm[j, b[3]]
            ∂Dn[j, b[2]] += ∂Rnl[j, i] * Pn[j, b[1]] * Ylm[j, b[3]]
        end
    end

    ∂p_Pn = pullback_ps(∂Pn, basis.Pn, R, ps.Pn, st.Pn)
    ∂p_Dn = pullback_ps(∂Dn, basis.Dn, R, ps.Dn, st.Dn)
    # Ylm has no parameters
    return (Pn = ∂p_Pn, Dn = ∂p_Dn, Ylm = NamedTuple())
end
