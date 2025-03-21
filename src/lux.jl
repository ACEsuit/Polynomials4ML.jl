
import LuxCore 
import LuxCore: initialparameters, initialstates, AbstractLuxLayer
using Random: AbstractRNG
using ChainRulesCore


"""
lux(basis) : convert a basis / embedding object into a lux layer. This assumes 
that the basis accepts a number or short vector as input and produces an output 
that is a vector. It also assumes that batched operations are implemented, 
as well as some other functionality. 
"""
function lux(basis::AbstractP4MLLayer; 
               name = String(nameof(typeof(basis))), 
               meta = Dict{String, Any}("name" => name),
            )
   @assert haskey(meta, "name")
   return PolyLuxLayer(basis, meta)
end

"""
a fall-back method for `initalparameters` that all AbstractP4MLBasis
should overload 
"""
_init_luxparams(rng::AbstractRNG, l::Any) = _init_luxparams(l)
_init_luxparams(l) = NamedTuple() 

_init_luxstate(rng::AbstractRNG, l) = _init_luxstate(l)
_init_luxstate(l) = NamedTuple() 


# ---------- PolyLuxLayer
# the simplest lux layer implementation

# WARNING: All PolyLuxLayer are assumed to be not containing trainable parameters, so that they fallback to rrule 
# interface without ps and st, all trainable PolyLayers should overload Polynomials4ML.lux with wanted function 
# and return another non-PolyLuxLayer type
struct PolyLuxLayer{TB} <: AbstractLuxLayer
   basis::TB
   meta::Dict{String, Any}
end

function Base.show(io::IO, l::PolyLuxLayer)
   print(io, "PolyLuxLayer($(l.meta["name"]))")
end

Base.length(l::PolyLuxLayer) = length(l.basis)

initialparameters(rng::AbstractRNG, l::PolyLuxLayer) = _init_luxparams(rng, l)

initialstates(rng::AbstractRNG, l::PolyLuxLayer) = _init_luxstate(rng, l)

(l::PolyLuxLayer)(args...) = evaluate(l, args...)

# general fallback of evaluate interface if we dont have trainble parameters in PolyLuxLayer
evaluate!(out, l::PolyLuxLayer, X, args...) = evaluate!(out, l.basis, X)

# lux evaluation interface
evaluate(l::PolyLuxLayer, X, ps, st) = evaluate(l.basis, X), st 

# Fallback of all PolyLuxLayer if no specific rrule is defined
# I use the usual rrule interface here since pb with temp array seems dangerous 
function ChainRulesCore.rrule(::typeof(LuxCore.apply), l::PolyLuxLayer, X, ps, st)
   val, inner_pb = ChainRulesCore.rrule(evaluate, l.basis, X)
   return (val, st), Δ -> (inner_pb(Δ[1])..., ZeroTangent(), NoTangent())
end
