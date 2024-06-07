using LinearAlgebra: dot 

export AtomicOrbitalsRadials, GaussianBasis, SlaterBasis, STO_NG

const NLM{T} = NamedTuple{(:n1, :n2, :l, :m), Tuple{T, T, T, T}}
const NL{T} = NamedTuple{(:n1, :n2, :l), Tuple{T, T, T}}

mutable struct AtomicOrbitalsRadials{TP, TD, TI}  <: AbstractP4MLBasis
   Pn::TP
   Dn::TD
   spec::Vector{NL{TI}}
   # ----------------- metadata 
   @reqfields
end

AtomicOrbitalsRadials(Pn, Dn, spec) = 
        AtomicOrbitalsRadials(Pn, Dn, spec, _make_reqfields()...)

Base.length(basis::AtomicOrbitalsRadials) = length(basis.spec)

natural_indices(basis::AtomicOrbitalsRadials) = copy(basis.spec)
degree(basis::AtomicOrbitalsRadials, b::NamedTuple) = b.n1

_valtype(basis::AtomicOrbitalsRadials, T::Type{<: Real}) = T
_valtype(basis::AtomicOrbitalsRadials, T::Type{<: Hyper{<:Real}}) = T

# -------- Evaluation Code 


function evaluate!(Rnl, basis::AtomicOrbitalsRadials, R::AbstractVector)
    nR = length(R)
    Pn = evaluate(basis.Pn, R)           # Pn(r)
    Dn = evaluate(basis.Dn, R)           # Dn(r)  (ζ are the parameters -> reorganize the Lux way)

    fill!(Rnl, zero(eltype(Rnl)))
    
    for (i, b) in enumerate(basis.spec)
        for j = 1:nR
            Rnl[j, i] = Pn[j, b.n1] * Dn[j, i]
        end
    end

    return Rnl 
end

function evaluate_ed!(Rnl, dRnl, basis::AtomicOrbitalsRadials{TP, TD, TI}, R::AbstractVector) where {TP, TD, TI}
    nR = length(R)
    Pn, dPn = evaluate_ed(basis.Pn, R)
    Dn, dDn = evaluate_ed(basis.Dn, R)

    fill!(Rnl, zero(eltype(Rnl))); fill!(dRnl, zero(eltype(Rnl))); 

    for (i, b) in enumerate(basis.spec)
        for j = 1:nR
            Rnl[j, i] += Pn[j, b.n1] * Dn[j, i]
            dRnl[j, i] += dPn[j, b.n1] * Dn[j, i]
            dRnl[j, i] += Pn[j, b.n1] * dDn[j, i]
        end
    end

    return Rnl, dRnl 
end

function evaluate_ed2!(Rnl, dRnl, ddRnl, basis::AtomicOrbitalsRadials, R::AbstractVector)
    nR = length(R)
    Pn, dPn, ddPn = evaluate_ed2(basis.Pn, R)
    Dn, dDn, ddDn = evaluate_ed2(basis.Dn, R)

    fill!(Rnl, zero(eltype(Rnl))); fill!(dRnl, zero(eltype(dRnl))); fill!(ddRnl, zero(eltype(ddRnl)))

    for (i, b) in enumerate(basis.spec)
        for j = 1:nR
            Rnl[j, i] += Pn[j, b.n1] * Dn[j, i]
            dRnl[j, i] += dPn[j, b.n1] * Dn[j, i] + Pn[j, b.n1] * dDn[j, i]
            ddRnl[j, i] += ddPn[j, b.n1] * Dn[j, i] + 2 * dPn[j, b.n1] * dDn[j, i] + Pn[j, b.n1] * ddDn[j, i]
        end
    end

    return Rnl, dRnl, ddRnl
end


# ---------------------------------------- 
#  gradient w.r.t. parameters 

function evaluate_ed_dp! end 

evaluate_ed_dp(basis, x) = _with_safe_alloc(evaluate_ed_dp!, basis, x) 


function evaluate_ed_dp!(Rnl, dRnl, dpRnl, basis::AtomicOrbitalsRadials, R::AbstractVector)
    nR = length(R)
    Pn, dPn = evaluate_ed(basis.Pn, R)
    Dn, dDn, dpDn = evaluate_ed_dp(basis.Dn, R)

    fill!(Rnl, zero(eltype(Rnl))); fill!(dRnl, zero(eltype(Rnl))); fill!(dpRnl, zero(eltype(Rnl))); 

    for (i, b) in enumerate(basis.spec)
        for j = 1:nR
            Rnl[j, i] += Pn[j, b.n1] * Dn[j, i]
            dRnl[j, i] += dPn[j, b.n1] * Dn[j, i]
            dRnl[j, i] += Pn[j, b.n1] * dDn[j, i]
            dpRnl[j, i] += Pn[j, b.n1] * dpDn[j, i]
        end
    end

    return Rnl, dRnl, dpRnl
end


# ------------------------------------------------------- 
include("gaussian.jl")
include("slater.jl")
include("sto_ng.jl")

AORBAS = Union{AtomicOrbitalsRadials, GaussianBasis, SlaterBasis, STO_NG}

function whatalloc(::typeof(evaluate_ed_dp!), basis::AORBAS, r::Number)
    Nb = length(basis)
    TV = _valtype(basis, r)
    TG = _gradtype(basis, r)
    return (TV, Nb), (TG, Nb), (TG, Nb)
end

function whatalloc(::typeof(evaluate_ed_dp!), basis::AORBAS, R::AbstractVector)
    Nb = length(basis); Nr = length(R)
    TV = _valtype(basis, R)
    TG = _gradtype(basis, R)
    return (TV, Nr, Nb), (TG, Nr, Nb), (TG, Nr, Nb)
end


function evaluate!(Rnl, basis::AORBAS, r::Number)
    Rnl_ = reshape(Rnl, (1, length(basis)))
    evaluate!(Rnl_, basis, [r,])
    return Rnl 
end

function evaluate_ed!(Rnl, dRnl, basis::AORBAS, r::Number)
    Rnl_ = reshape(Rnl, (1, length(basis)))
    dRnl_ = reshape(dRnl, (1, length(basis)))
    evaluate_ed!(Rnl_, dRnl_, basis, [r,])
    return Rnl, dRnl 
end

function evaluate_ed_dp!(Rnl, dRnl, dpRnl, basis::AORBAS, r::Number)
    Rnl_ = reshape(Rnl, (1, length(basis)))
    dpRnl_ = reshape(dpRnl, (1, length(basis)))
    dRnl_ = reshape(dRnl, (1, length(basis)))
    evaluate_ed_dp!(Rnl_, dRnl_, dpRnl_, basis, [r,])
    return Rnl, dRnl, dpRnl 
end

function evaluate_ed2!(Rnl, dRnl, ddRnl, basis::AORBAS, r::Number)
    Rnl_ = reshape(Rnl, (1, length(basis)))
    dRnl_ = reshape(dRnl, (1, length(basis)))
    ddRnl_ = reshape(ddRnl, (1, length(basis)))
    evaluate_ed2!(Rnl_, dRnl_, ddRnl_, basis, [r,])
    return Rnl, dRnl, ddRnl 
end


# --------------------- connect with Lux 
struct AORLayer{TP, TD, TI} <: AbstractExplicitLayer 
    basis::AtomicOrbitalsRadials{TP, TD, TI}
end

struct STOLayer{TP, TD, TI} <: AbstractExplicitLayer 
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