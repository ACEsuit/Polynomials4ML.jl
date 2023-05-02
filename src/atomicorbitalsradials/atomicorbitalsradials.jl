export AtomicOrbitalsRadials, GaussianBasis, SlaterBasis, STO_NG

const NLM{T} = NamedTuple{(:n1, :n2, :l, :m), Tuple{T, T, T, T}}
const NL{T} = NamedTuple{(:n1, :n2, :l), Tuple{T, T, T}}

struct AtomicOrbitalsRadials{TP, TD, TI, TZ} 
   Pn::TP
   Dn::TD
   spec::Vector{NL{TI}}
   ζ::Vector{TZ}
   # ----------------- metadata 
   meta::Dict{String, Any}
end

AtomicOrbitalsRadials(Pn, Dn, spec, ζ; meta = Dict{String, Any}()) = 
        AtomicOrbitalsRadials(Pn, Dn, spec, ζ, meta)

Base.length(basis::AtomicOrbitalsRadials) = length(basis.spec)

# -------- Evaluation Code 
_alloc(basis::AtomicOrbitalsRadials, r::T) where {T <: Real} = 
      zeros(T, length(basis))

_alloc(basis::AtomicOrbitalsRadials, rr::Vector{T}) where {T <: Real} = 
      zeros(T, length(rr), length(basis))

evaluate(basis::AtomicOrbitalsRadials, r::Number) = evaluate(basis, [r,])[:]

function evaluate(basis::AtomicOrbitalsRadials, R::AbstractVector{<: Real})
    nR = length(R)
    Pn = Polynomials4ML.evaluate(basis.Pn, R) # Pn(r)
    Dn = evaluate(basis.Dn, basis.ζ, R) # Dn(r)
    Rnl = _alloc(basis, R) 

    for (i, b) in enumerate(basis.spec)
        for j = 1:nR
            Rnl[j, i] = Pn[j, b.n1] * Dn[j, i]
        end
    end

    return Rnl 
end

function evaluate_ed(basis::AtomicOrbitalsRadials, R)
    nR = length(R) 
    Pn, dPn = Polynomials4ML.evaluate_ed(basis.Pn, R)
    Dn, dDn = evaluate_ed(basis.Dn, basis.ζ, R)
    Rnl = _alloc(basis, R)
    dRnl = _alloc(basis, R)

    for (i, b) in enumerate(basis.spec)
        for j = 1:nR
            Rnl[j, i] += Pn[j, b.n1] * Dn[j, i]
            dRnl[j, i] += dPn[j, b.n1] * Dn[j, i]
            dRnl[j, i] += Pn[j, b.n1] * dDn[j, i]
        end
    end
    return Rnl, dRnl 
end


function evaluate_ed2(basis::AtomicOrbitalsRadials, R)
    nR = length(R) 
    Pn, dPn, ddPn = Polynomials4ML.evaluate_ed2(basis.Pn, R)
    Dn, dDn, ddDn = evaluate_ed2(basis.Dn, basis.ζ, R)
    Rnl = _alloc(basis, R)
    dRnl = _alloc(basis, R)
    ddRnl = _alloc(basis, R)

    for (i, b) in enumerate(basis.spec)
        for j = 1:nR
            Rnl[j, i] += Pn[j, b.n1] * Dn[j, i]
            dRnl[j, i] += dPn[j, b.n1] * Dn[j, i] + Pn[j, b.n1] * dDn[j, i]
            ddRnl[j, i] += ddPn[j, b.n1] * Dn[j, i] + 2 * dPn[j, b.n1] * dDn[j, i] + Pn[j, b.n1] * ddDn[j, i]
        end
    end
    return Rnl, dRnl, ddRnl
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
