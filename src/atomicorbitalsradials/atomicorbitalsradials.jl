export AtomicOrbitalsRadials, GaussianBasis, SlaterBasis, STO_NG
using ChainRulesCore
using ChainRulesCore: NoTangent
using HyperDualNumbers: Hyper

import LuxCore 
import LuxCore: initialparameters, initialstates, AbstractExplicitLayer
using Random: AbstractRNG


const NLM{T} = NamedTuple{(:n1, :n2, :l, :m), Tuple{T, T, T, T}}
const NL{T} = NamedTuple{(:n1, :n2, :l), Tuple{T, T, T}}

struct AtomicOrbitalsRadials{TP, TD, TI}  <: ScalarPoly4MLBasis
   Pn::TP
   Dn::TD
   spec::Vector{NL{TI}}
   # ----------------- metadata 
   @reqfields
end

AtomicOrbitalsRadials(Pn, Dn, spec) = 
        AtomicOrbitalsRadials(Pn, Dn, spec, _make_reqfields()...)

Base.length(basis::AtomicOrbitalsRadials) = length(basis.spec)

_valtype(basis::AtomicOrbitalsRadials, T::Type{<: Real}) = T
_valtype(basis::AtomicOrbitalsRadials, T::Type{<: Hyper{<:Real}}) = T

# -------- Evaluation Code 

# TODO: (Jerry?) this kind of construction could be used for all  bases? 
#       file an issue on this.

function evaluate!(Rnl, basis::AtomicOrbitalsRadials, R::AbstractVector)
    nR = length(R)
    Pn = evaluate(basis.Pn, R)           # Pn(r)
    Dn = evaluate(basis.Dn, R)           # Dn(r)  (ζ are the parameters -> reorganize the Lux way)

    fill!(Rnl, 0)
    
    for (i, b) in enumerate(basis.spec)
        for j = 1:nR
            Rnl[j, i] = Pn[j, b.n1] * Dn[j, i]
        end
    end

    release!(Pn); release!(Dn)

    return Rnl 
end

function evaluate_ed!(Rnl, dRnl, basis::AtomicOrbitalsRadials, R::AbstractVector)
    nR = length(R)
    Pn, dPn = evaluate_ed(basis.Pn, R)
    Dn, dDn = evaluate_ed(basis.Dn, R)

    fill!(Rnl, 0); fill!(dRnl, 0); 

    for (i, b) in enumerate(basis.spec)
        for j = 1:nR
            Rnl[j, i] += Pn[j, b.n1] * Dn[j, i]
            dRnl[j, i] += dPn[j, b.n1] * Dn[j, i]
            dRnl[j, i] += Pn[j, b.n1] * dDn[j, i]
        end
    end

    release!(Pn); release!(dPn); release!(Dn); release!(dDn)

    return Rnl, dRnl 
end


function evaluate_ed2!(Rnl, dRnl, ddRnl, basis::AtomicOrbitalsRadials, R::AbstractVector)
    nR = length(R)
    Pn, dPn, ddPn = evaluate_ed2(basis.Pn, R)
    Dn, dDn, ddDn = evaluate_ed2(basis.Dn, R)

    fill!(Rnl, 0); fill!(dRnl, 0); fill!(ddRnl, 0)

    for (i, b) in enumerate(basis.spec)
        for j = 1:nR
            Rnl[j, i] += Pn[j, b.n1] * Dn[j, i]
            dRnl[j, i] += dPn[j, b.n1] * Dn[j, i] + Pn[j, b.n1] * dDn[j, i]
            ddRnl[j, i] += ddPn[j, b.n1] * Dn[j, i] + 2 * dPn[j, b.n1] * dDn[j, i] + Pn[j, b.n1] * ddDn[j, i]
        end
    end

    release!(Pn); release!(dPn); release!(ddPn); 
    release!(Dn); release!(dDn); release!(ddDn)

    return Rnl, dRnl, ddRnl
end

natural_indices(basis::AtomicOrbitalsRadials) = copy(basis.spec)
degree(basis::AtomicOrbitalsRadials, b::NamedTuple) = b.n1

include("gaussian.jl")
include("slater.jl")
include("sto_ng.jl")

const ExponentialType = Union{GaussianBasis, SlaterBasis, STO_NG}


function evaluate!(Rnl, basis::Union{AtomicOrbitalsRadials, ExponentialType}, r::Number)
    Rnl_ = reshape(Rnl, (1, length(basis)))
    evaluate!(Rnl_, basis, [r,])
    return Rnl 
end

function evaluate_ed!(Rnl, dRnl, basis::Union{AtomicOrbitalsRadials, ExponentialType}, r::Number)
    Rnl_ = reshape(Rnl, (1, length(basis)))
    dRnl_ = reshape(dRnl, (1, length(basis)))
    evaluate_ed!(Rnl_, dRnl_, basis, [r,])
    return Rnl 
end

function evaluate_ed2!(Rnl, dRnl, ddRnl, basis::Union{AtomicOrbitalsRadials, ExponentialType}, r::Number)
    Rnl_ = reshape(Rnl, (1, length(basis)))
    dRnl_ = reshape(dRnl, (1, length(basis)))
    ddRnl_ = reshape(ddRnl, (1, length(basis)))
    evaluate_ed2!(Rnl_, dRnl_, ddRnl_, basis, [r,])
    return Rnl 
end

# --------------------- connect with Lux 
struct AORLayer <: AbstractExplicitLayer 
    basis::AtomicOrbitalsRadials
end

lux(basis::AtomicOrbitalsRadials) = AORLayer(basis)
 
initialparameters(rng::AbstractRNG, l::AORLayer) = ( ζ = l.basis.Dn.ζ, )
 
initialstates(rng::AbstractRNG, l::AORLayer) = NamedTuple()
 
# This should be removed later and replace by ObejctPools
(l::AORLayer)(X, ps, st) = 
       evaluate(l.basis, X), st