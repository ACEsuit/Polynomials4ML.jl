
import LuxCore 
import LuxCore: initialparameters, initialstates, AbstractExplicitLayer
using Random: AbstractRNG

using ChainRulesCore

"""
lux(basis) : convert a basis / embedding object into a lux layer. This assumes 
that the basis accepts a number or short vector as input and produces an output 
that is a vector. It also assumes that batched operations are implemented, 
as well as some other functionality. 
"""
function lux(basis::AbstractPoly4MLBasis; 
               use_cache = true,
               name = String(nameof(typeof(basis))), 
               meta = Dict{String, Any}("name" => name),
            )
   @assert haskey(meta, "name")
   return PolyLuxLayer(basis, meta, use_cache)
end
"""
a fall-back method for `initalparameters` that all AbstractPoly4MLBasis
should overload 
"""
_init_luxparams(rng::AbstractRNG, l::Any) = _init_luxparams(l)
_init_luxparams(l) = NamedTuple() 

_init_luxstate(rng::AbstractRNG, l) = _init_luxstate(l)
_init_luxstate(l) = _init_default_luxstate(l.use_cache)
_init_default_luxstate(use_cache) = ( use_cache ?  (pool =  ArrayPool(FlexArrayCache), ) : (pool =  ArrayPool(FlexArray), ))



# ---------- PolyLuxLayer
# the simplest lux layer implementation

# WARNING: All PolyLuxLayer are assumed to be not containing trainable parameters, so that they fallback to rrule 
# interface without ps and st, all trainable PolyLayers should overload Polynomials4ML.lux with wanted function 
# and return another non-PolyLuxLayer type
struct PolyLuxLayer{TB} <: AbstractExplicitLayer
   basis::TB
   meta::Dict{String, Any}
   use_cache::Bool
end

function Base.show(io::IO, l::PolyLuxLayer)
   print(io, "PolyLuxLayer($(l.meta["name"]))")
end

Base.length(l::PolyLuxLayer) = length(l.basis)

initialparameters(rng::AbstractRNG, l::PolyLuxLayer) = _init_luxparams(rng, l)

initialstates(rng::AbstractRNG, l::PolyLuxLayer) = _init_luxstate(rng, l)

(l::PolyLuxLayer)(args...) = evaluate(l, args...)

# general fallback of evaluate and pullback interface
evaluate!(out, basis::AbstractPoly4MLBasis, X, ps, st) = evaluate!(out, basis, X)

# lux evaluation interface
function evaluate(l::PolyLuxLayer, X, ps, st)
   out = acquire!(st.pool, _outsym(X), _out_size(l.basis, X), _valtype(l.basis, X))
   evaluate!(out, l.basis, X, ps, st)
   return out, st
end

# Discuss: This only uses the usual eval interface with ArrayCache. Can we use tmp array in pb too?
function ChainRulesCore.rrule(::typeof(LuxCore.apply), l::PolyLuxLayer, X, ps, st)
   val, inner_pb = ChainRulesCore.rrule(evaluate, l.basis, X)
   return (val, st), Δ -> (inner_pb(Δ[1])..., NoTangent(), NoTangent())
end

## === 

## Backup: interface before we migrate to non-allocating lux layers

# function evaluate(l::PolyLuxLayer, X, ps, st)
   
#    # TODO: after we make sure we want to migrate to HyperDualNumbers in any cases we can ignore_derivatives from ChainRulesCore
#    #B = ChainRulesCore.ignore_derivatives() do 
#    #   evaluate(l.basis, X)
#    #end
#    B = evaluate(l.basis, X)
#    return B, st 
# end 
