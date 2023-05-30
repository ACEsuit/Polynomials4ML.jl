export AtomicOrbitalsRadials, GaussianBasis, SlaterBasis, STO_NG
using ChainRulesCore
using ChainRulesCore: NoTangent

const NLM{T} = NamedTuple{(:n1, :n2, :l, :m), Tuple{T, T, T, T}}
const NL{T} = NamedTuple{(:n1, :n2, :l), Tuple{T, T, T}}

struct AtomicOrbitalsRadials{TP, TD, TI, TZ}  <: AbstractPoly4MLBasis
   Pn::TP
   Dn::TD
   spec::Vector{NL{TI}}
   ζ::Vector{TZ}   # later : this does into a parameters named-tuple 
   # ----------------- metadata 
   @reqfields
end

AtomicOrbitalsRadials(Pn, Dn, spec, ζ) = 
        AtomicOrbitalsRadials(Pn, Dn, spec, ζ, _make_reqfields()...)

Base.length(basis::AtomicOrbitalsRadials) = length(basis.spec)

_valtype(basis::AtomicOrbitalsRadials, T::Type{<: Real}) = T

# -------- Evaluation Code 

# TODO: (Jerry?) this kind of construction could be used for all  bases? 
#       file an issue on this.

function evaluate!(Rnl, basis::AtomicOrbitalsRadials, r::Number)
    Rnl_ = reshape(Rnl, (1, length(basis)))
    evaluate!(Rnl_, basis, [r,])
    return Rnl 
end


function evaluate!(Rnl, basis::AtomicOrbitalsRadials, R::AbstractVector{<: Real})
    nR = length(R)
    Pn = evaluate(basis.Pn, R)           # Pn(r)
    Dn = evaluate(basis.Dn, basis.ζ, R)  # Dn(r)  (ζ are the parameters -> reorganize the Lux way)

    fill!(Rnl, 0)
    
    for (i, b) in enumerate(basis.spec)
        for j = 1:nR
            Rnl[j, i] = Pn[j, b.n1] * Dn[j, i]
        end
    end

    release!(Pn); release!(Dn)

    return Rnl 
end

function evaluate_ed!(Rnl, dRnl, basis::AtomicOrbitalsRadials, R)
    nR = length(R)
    Pn, dPn = evaluate_ed(basis.Pn, R)
    Dn, dDn = evaluate_ed(basis.Dn, basis.ζ, R)

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


function evaluate_ed2!(Rnl, dRnl, ddRnl, basis::AtomicOrbitalsRadials, R)
    nR = length(R)
    Pn, dPn, ddPn = evaluate_ed2(basis.Pn, R)
    Dn, dDn, ddDn = evaluate_ed2(basis.Dn, basis.ζ, R)

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

# not test
function ChainRulesCore.rrule(::typeof(evaluate), basis::AtomicOrbitalsRadials, R::AbstractVector{<: Real})
   A  = evaluate(basis, R)
   ∂R = similar(R)
   dR = evaluate_ed(basis, R)[2]
   function pb(∂A)
        @assert size(∂A) == (length(R), length(basis))
        for i = 1:length(R)
            ∂R[i] = dot(@view(∂A[i, :]), @view(dR[i, :]))
        end
        return NoTangent(), NoTangent(), ∂R
   end
   return A, pb
end

include("gaussian.jl")
include("slater.jl")
include("sto_ng.jl")

const ExponentialType = Union{GaussianBasis, SlaterBasis, STO_NG}

evaluate(basis::ExponentialType, ζ::Number, r::Number) = evaluate(basis, [ζ,], [r,])[:]
evaluate(basis::ExponentialType, ζ::Number, r::Vector) = evaluate(basis, [ζ,], r)
evaluate(basis::ExponentialType, ζ::Vector, r::Number) = evaluate(basis, ζ, [r,])
evaluate_ed(basis::ExponentialType, ζ::Number, r::Number) = evaluate_ed(basis, [ζ,], [r,])[:]
evaluate_ed(basis::ExponentialType, ζ::Number, r::Vector) = evaluate_ed(basis, [ζ,], r)
evaluate_ed(basis::ExponentialType, ζ::Vector, r::Number) = evaluate_ed(basis, ζ, [r,])
evaluate_ed2(basis::ExponentialType, ζ::Number, r::Number) = evaluate_ed2(basis, [ζ,], [r,])[:]
evaluate_ed2(basis::ExponentialType, ζ::Number, r::Vector) = evaluate_ed2(basis, [ζ,], r)
evaluate_ed2(basis::ExponentialType, ζ::Vector, r::Number) = evaluate_ed2(basis, ζ, [r,])

natural_indices(basis::AtomicOrbitalsRadials) = copy(basis.spec)
degree(basis::AtomicOrbitalsRadials, b::NamedTuple) = b.n1