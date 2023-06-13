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
function evaluate(l::AORLayer, X, ps, st)
    B = evaluate(l.basis, X)
    return B, st 
end 

(l::AORLayer)(X, ps, st) = evaluate(l, X, ps, st)


# The following code is used to compute the derivative with respect to zeta by Hyperduals.

_alloc_dp(basis::ExponentialType, X) = 
      acquire!(basis.pool, _outsym(X), _out_size(basis, X), promote_type(eltype(basis.ζ)) )

_alloc_dp(basis::AtomicOrbitalsRadials, X) = 
      acquire!(basis.pool, _outsym(X), _out_size(basis, X), promote_type(eltype(basis.Dn.ζ)) )

function eval_dp!(Rnl, basis::AtomicOrbitalsRadials, R::AbstractVector)
    nR = length(R)
    Pn = evaluate(basis.Pn, R)   
    D = Polynomials4ML._alloc_dp(basis.Dn, R)
    Dn = evaluate!(D, basis.Dn, R) 

    fill!(Rnl, 0)
    
    for (i, b) in enumerate(basis.spec)
        for j = 1:nR
            Rnl[j, i] = Pn[j, b.n1] * Dn[j, i]
        end
    end

    release!(Pn); release!(Dn)

    return Rnl 
end

function expontype(Dn::GaussianBasis, ζ)
    hζ = [ Hyper(ζ[i], 1, 1, 0) for i = 1:length(ζ) ] 
    return GaussianBasis(hζ)
end

function expontype(Dn::SlaterBasis, ζ)
    hζ = [ Hyper(ζ[i], 1, 1, 0) for i = 1:length(ζ) ] 
    return SlaterBasis(hζ)
end

eps1(h::Hyper{<:Real}) = h.epsilon1

function pb_params(∂ζ, ζ, basis::AtomicOrbitalsRadials, R::AbstractVector{<: Real})
    Dn = expontype(basis.Dn, ζ)
    bRnl = AtomicOrbitalsRadials(basis.Pn, Dn, basis.spec) 
    Rnl = _alloc_dp(bRnl, R)
    eval_dp!(Rnl, bRnl, R)
    ∂ζ = eps1.(Rnl)
    return ∂ζ
end

function ChainRulesCore.rrule(::typeof(evaluate), basis::AtomicOrbitalsRadials, R::AbstractVector{<: Real})
    A, dR = evaluate_ed(basis, R)
    dζ = pb_params(similar(A), basis.Dn.ζ, basis, R)

    ∂R = similar(R)
    ∂ζ = similar(basis.Dn.ζ)
    function pb(∂A)
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

function ChainRulesCore.rrule(::typeof(evaluate), l::AORLayer, R::AbstractVector{<: Real}, ps, st)
    A = l(R, ps, st)
    dζ = pb_params(similar(A[1]), l.basis.Dn.ζ, l.basis, R)
    ∂ζ = similar(l.basis.Dn.ζ)
    function pb(∂A)
        @assert size(∂A[1]) == (length(R), length(l.basis))
        for i = 1:length(l.basis.Dn.ζ)
            ∂ζ[i] = dot(@view(∂A[1][:, i]), @view(dζ[:, i]))
         end
        return NoTangent(), NoTangent(), NoTangent(), (ζ = ∂ζ,), NoTangent()
    end
    return A, pb
end 