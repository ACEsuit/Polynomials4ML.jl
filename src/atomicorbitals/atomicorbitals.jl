using LinearAlgebra: dot 

export AtomicOrbitalsRadials

abstract type AbstractDecayFunction end

struct RadialDecay{TSMAT, DF<:AbstractDecayFunction} <: AbstractP4MLBasis
    ζ::TSMAT
    D::TSMAT
    decay::DF
end

const NT_NNL = NamedTuple{(:n1, :n2, :l), Tuple{Int, Int, Int}}

# AORBAS = Union{AtomicOrbitalsRadials, GaussianBasis, SlaterBasis, STO_NG}

mutable struct AtomicOrbitalsRadials{LEN, TP, TD}  <: AbstractP4MLBasis
   Pn::TP
   Dn::TD
   spec::SVector{LEN, NT_NNL}
end

function AtomicOrbitalsRadials(Pn, Dn, spec::AbstractVector{NT_NNL})
    LEN = length(spec)
    return AtomicOrbitalsRadials{LEN, typeof(Pn), typeof(Dn)}(Pn, Dn, SVector{LEN, NT_NNL}(spec))
end

Base.length(basis::AtomicOrbitalsRadials) = length(basis.spec)

natural_indices(basis::AtomicOrbitalsRadials) = basis.spec

_valtype(basis::AtomicOrbitalsRadials, T::Type{<: Real}) = 
        promote_type(_valtype(basis.Pn, T), _valtype(basis.Dn, T))

_valtype(basis::AtomicOrbitalsRadials, T::Type{<: Real}, 
            ps::Union{Nothing, @NamedTuple{}}, st) = 
        promote_type(_valtype(basis.Pn, T), _valtype(basis.Dn, T))

_valtype(basis::AtomicOrbitalsRadials, T::Type{<: Real}, ps, st) = 
        promote_type(_valtype(basis.Pn, T, ps.Dn, st.Dn), 
                     _valtype(basis.Dn, T, ps.Dn, st.Dn))

_generate_input(basis::AtomicOrbitalsRadials) = 0.1 + 0.9 * rand()

Base.show(io::IO, basis::AtomicOrbitalsRadials) = 
        print(io, "AtomicOrbitalsRadials($(basis.Pn), $(basis.Dn))")

# Type of atomic orbital type basis sets         

include("radialdecay.jl")

# _static_params is used to extract parameters from the basis set when 
# the basis is evaluated with the old parameter-free convention. In that case, 
# the internally stored parameters are used. 
#
# _init_luxparams is used to initialize parameters in the lux style, as a 
# NamedTuple. This is used when the basis as a learnable Lux layer.

_static_params(basis::AbstractP4MLBasis) = NamedTuple() 

_static_params(basis::AtomicOrbitalsRadials) = 
        (Pn = _static_params(basis.Pn), Dn = _static_params(basis.Dn), )

_init_luxparams(rng::Random.AbstractRNG, l::AtomicOrbitalsRadials) = 
        ( Pn = _init_luxparams(rng, l.Pn), 
          Dn = _init_luxparams(rng, l.Dn), )

_init_luxstate(rng::Random.AbstractRNG, l::AtomicOrbitalsRadials) = 
        ( Pn = _init_luxstate(rng, l.Pn), 
          Dn = _init_luxstate(rng, l.Dn), )          

# -------- Evaluation Code 

_evaluate!(Rnl, dRnl, basis::AtomicOrbitalsRadials, X) = 
            _evaluate!(Rnl, dRnl, basis, X, 
                       _static_params(basis), 
                       (Pn = nothing, Dn = nothing))

function _evaluate!(Rnl, dRnl, basis::AtomicOrbitalsRadials, R::BATCH, 
                     ps, st)
    nR = length(R)
    WITHGRAD = !isnothing(dRnl)

    fill!(Rnl, zero(eltype(Rnl)))
    WITHGRAD && fill!(dRnl, zero(eltype(Rnl)))

    @no_escape begin 
        if WITHGRAD
            # this is a hack that circumvents an unexplained allocation in 
            # the @withalloc macro 
            T = promote_type(eltype(Rnl), eltype(R))
            Pn = @alloc(T, nR, length(basis.Pn))
            dPn = @alloc(T, nR, length(basis.Pn))
            _evaluate!(Pn, dPn, basis.Pn, R, ps.Pn, st.Pn)
            Dn = @alloc(T, nR, length(basis.Dn))
            dDn = @alloc(T, nR, length(basis.Dn))
            _evaluate!(Dn, dDn, basis.Dn, R, ps.Dn, st.Dn)
            # Pn, dPn = @withalloc evaluate_ed!(basis.Pn, R)
            # Dn, dDn = @withalloc evaluate_ed!(basis.Dn, R)
        else 
            Pn = @withalloc evaluate!(basis.Pn, R, ps.Pn, st.Pn)   # Pn(r)
            Dn = @withalloc evaluate!(basis.Dn, R, ps.Dn, st.Dn)   # Dn(r)  (ζ are the parameters -> reorganize the Lux way)
            dPn = nothing
            dDn = nothing
        end
        for (i, b) in enumerate(basis.spec)
            @simd ivdep for j = 1:nR
                Rnl[j, i] = Pn[j, b.n1] * Dn[j, i]
                if WITHGRAD 
                    dRnl[j, i] = ( dPn[j, b.n1] *  Dn[j, i] + 
                                    Pn[j, b.n1] * dDn[j, i] )
                end
            end
        end
    end

    return nothing 
end


function pullback_ps(∂Rnl, basis::AtomicOrbitalsRadials, X::BATCH,
                     ps::NamedTuple, st)
    T = promote_type(eltype(∂Rnl), eltype(X)) 
    nR = length(X)

    # Rnl = output of evaluate(basis, X, ...)
    Pn = evaluate(basis.Pn, X, ps.Pn, st.Pn)
    Dn = evaluate(basis.Dn, X, ps.Dn, st.Dn)
    ∂Pn = zeros(T, size(Pn))
    ∂Dn = zeros(T, size(Dn))

    for (i, b) in enumerate(basis.spec)
        @simd ivdep for j = 1:nR
            #             Rnl[j, i] =             Pn[j, b.n1] * Dn[j, i]
            # ∂Rnl[j,i] * Rnl[j, i] = ∂Rnl[j,i] * Pn[j, b.n1] * Dn[j, i]
            ∂Pn[j, b.n1] += ∂Rnl[j, i] * Dn[j, i]
            ∂Dn[j, i] += ∂Rnl[j, i] * Pn[j, b.n1]
        end 
    end 

    ∂p_Pn = pullback_ps(∂Pn, basis.Pn, X, ps.Pn, st.Pn)
    ∂p_Dn = pullback_ps(∂Dn, basis.Dn, X, ps.Dn, st.Dn)
    return (Pn = ∂p_Pn, Dn = ∂p_Dn,)
end

# ---------------------------------------- 
#  gradient w.r.t. parameters 
#  We won't keep this for now; maybe we will need it again so we kee it 
#  commented out until we can test that it is no longer needed. 
#=
function evaluate_ed_dp! end 

evaluate_ed_dp(basis, x) = _with_safe_alloc(evaluate_ed_dp!, basis, x) 


function evaluate_ed_dp!(Rnl, dRnl, dpRnl, basis::AtomicOrbitalsRadials, R::AbstractVector)
    nR = length(R)
    fill!(Rnl, zero(eltype(Rnl)))
    fill!(dRnl, zero(eltype(Rnl)))
    fill!(dpRnl, zero(eltype(Rnl))) 

    @no_escape begin 
        Pn, dPn = @withalloc evaluate_ed!(basis.Pn, R)
        Dn, dDn, dpDn = @withalloc evaluate_ed_dp!(basis.Dn, R)
        for (i, b) in enumerate(basis.spec)
            for j = 1:nR
                Rnl[j, i] += Pn[j, b.n1] * Dn[j, i]
                dRnl[j, i] += dPn[j, b.n1] * Dn[j, i]
                dRnl[j, i] += Pn[j, b.n1] * dDn[j, i]
                dpRnl[j, i] += Pn[j, b.n1] * dpDn[j, i]
            end
        end
    end 
    return Rnl, dRnl, dpRnl
end
=#

# ------------------------------------------------------- 

# function whatalloc(::typeof(evaluate_ed_dp!), basis::AORBAS, r::Number)
#     Nb = length(basis)
#     TV = _valtype(basis, r)
#     TG = _gradtype(basis, r)
#     return (TV, Nb), (TG, Nb), (TG, Nb)
# end

# function whatalloc(::typeof(evaluate_ed_dp!), basis::AORBAS, R::AbstractVector)
#     Nb = length(basis); Nr = length(R)
#     TV = _valtype(basis, R)
#     TG = _gradtype(basis, R)
#     return (TV, Nr, Nb), (TG, Nr, Nb), (TG, Nr, Nb)
# end


# function evaluate!(Rnl, basis::AORBAS, r::Number)
#     Rnl_ = reshape(Rnl, (1, length(basis)))
#     evaluate!(Rnl_, basis, [r,])
#     return Rnl 
# end

# function evaluate_ed!(Rnl, dRnl, basis::AORBAS, r::Number)
#     Rnl_ = reshape(Rnl, (1, length(basis)))
#     dRnl_ = reshape(dRnl, (1, length(basis)))
#     evaluate_ed!(Rnl_, dRnl_, basis, [r,])
#     return Rnl, dRnl 
# end

# function evaluate_ed_dp!(Rnl, dRnl, dpRnl, basis::AORBAS, r::Number)
#     Rnl_ = reshape(Rnl, (1, length(basis)))
#     dpRnl_ = reshape(dpRnl, (1, length(basis)))
#     dRnl_ = reshape(dRnl, (1, length(basis)))
#     evaluate_ed_dp!(Rnl_, dRnl_, dpRnl_, basis, [r,])
#     return Rnl, dRnl, dpRnl 
# end

# function evaluate_ed2!(Rnl, dRnl, ddRnl, basis::AORBAS, r::Number)
#     Rnl_ = reshape(Rnl, (1, length(basis)))
#     dRnl_ = reshape(dRnl, (1, length(basis)))
#     ddRnl_ = reshape(ddRnl, (1, length(basis)))
#     evaluate_ed2!(Rnl_, dRnl_, ddRnl_, basis, [r,])
#     return Rnl, dRnl, ddRnl 
# end


# --------------------- connect with Lux 
#=
struct AORLayer{TP, TD, TI} <: AbstractLuxLayer 
    basis::AtomicOrbitalsRadials{TP, TD, TI}
end

struct STOLayer{TP, TD, TI} <: AbstractLuxLayer 
    basis::AtomicOrbitalsRadials{TP, TD, TI}
end

lux(basis::AtomicOrbitalsRadials) = begin
    if basis.Dn isa Union{GaussianBasis,SlaterBasis}
        return AORLayer(basis)
    elseif basis.Dn isa STO_NG
        return STOLayer(basis)
    end
end

initialparameters(rng::AbstractRNG, l::AORLayer) = ( ζ = l.basis.Dn.ζ, )
 
initialstates(rng::AbstractRNG, l::AORLayer) = NamedTuple()
 
function evaluate(l::AORLayer, X, ps, st)
    l.basis.Dn.ζ = ps[1]
    B = evaluate(l.basis, X)
    return B, st 
end 

(l::AORLayer)(X, ps, st) = evaluate(l, X, ps, st)
 
initialparameters(rng::AbstractRNG, l::STOLayer) = NamedTuple()
 
initialstates(rng::AbstractRNG, l::STOLayer) = ( ζ = l.basis.Dn.ζ, )
 
function evaluate(l::STOLayer, X::AbstractArray, ps, st)
    l.basis.Dn.ζ::Tuple{Matrix{Float64}, Matrix{Float64}} = st[1]
    B = evaluate(l.basis, X)
    return B, st 
end 

(l::STOLayer)(X, ps, st) = evaluate(l, X, ps, st)


function ChainRulesCore.rrule(::typeof(evaluate), l::AORLayer, R::AbstractVector{<: Real}, ps, st)
    A, dR, dζ = evaluate_ed_dp(l.basis, R)
    ∂R = similar(R)
    ∂ζ = similar(l.basis.Dn.ζ)
    function pb(∂A)
        @assert size(∂A[1]) == (length(R), length(l.basis))
        for i = 1:length(R)
            ∂R[i] = dot(@view(∂A[1][i, :]), @view(dR[i, :]))
        end
        for i = 1:length(l.basis.Dn.ζ)
            ∂ζ[i] = dot(@view(∂A[1][:, i]), @view(dζ[:, i]))
        end
        return NoTangent(), NoTangent(), ∂R, (ζ = ∂ζ,), NoTangent()
    end
    return (A, NamedTuple()), pb
end 

function ChainRulesCore.rrule(::typeof(evaluate), l::STOLayer, R::AbstractVector{<: Real}, ps, st)
    A, dR = evaluate_ed(l.basis, R)
    ∂R = similar(R)
    function pb(∂A)
        @assert size(∂A[1]) == (length(R), length(l.basis))
        for i = 1:length(R)
            ∂R[i] = dot(@view(∂A[1][i, :]), @view(dR[i, :]))
        end
        return NoTangent(), NoTangent(), ∂R, NoTangent(), NoTangent()
    end
    return (A, NamedTuple()), pb
end 



# The following code is used to compute the derivative with respect to zeta by Hyperduals.

#_alloc_dp(basis::ExponentialType, X) = 
#      acquire!(basis.tmp, _outsym(X), _out_size(basis, X), promote_type(eltype(basis.ζ)) )

#_alloc_dp(basis::AtomicOrbitalsRadials, X) = 
#      acquire!(basis.tmp, _outsym(X), _out_size(basis, X), promote_type(eltype(basis.Dn.ζ)) )

#function eval_dp!(Rnl, basis::AtomicOrbitalsRadials, R::AbstractVector)
#    nR = length(R)
#    Pn = evaluate(basis.Pn, R)
#    D = Polynomials4ML._alloc_dp(basis.Dn, R)
#    Dn = evaluate!(D, basis.Dn, R) 

#    fill!(Rnl, zero(eltype(Rnl)))
    
#    for (i, b) in enumerate(basis.spec)
#        for j = 1:nR
#            Rnl[j, i] = Pn[j, b.n1] * Dn[j, i]
#        end
#    end

#    release!(Pn); release!(Dn)

#    return Rnl 
#end

#function expontype(Dn::GaussianBasis, ζ)
#    hζ = [ Hyper(ζ[i], 1, 1, 0) for i = 1:length(ζ) ] 
#    return GaussianBasis(hζ)
#end

#function expontype(Dn::SlaterBasis, ζ)
#    hζ = [ Hyper(ζ[i], 1, 1, 0) for i = 1:length(ζ) ] 
#    return SlaterBasis(hζ)
#end

#eps1(h::Hyper{<:Real}) = h.epsilon1

#function pb_params(ζ::AbstractVector, basis::AtomicOrbitalsRadials, R::AbstractVector{<: Real})
#    Dn = expontype(basis.Dn, ζ)
#    bRnl = AtomicOrbitalsRadials(basis.Pn, Dn, basis.spec) 
#    Rnl = _alloc_dp(bRnl, R)
#    eval_dp!(Rnl, bRnl, R)
#    dζ = eps1.(Rnl)
#    return copy(dζ)
#end

=#