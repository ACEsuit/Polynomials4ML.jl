
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
function lux(basis::AbstractP4MLBasis, label::Symbol = Symbol(""))
   return PolyLuxLayer{typeof(basis), label}(basis)
end

"""
a fall-back method for `initalparameters` that all AbstractP4MLBasis
should overload 
"""
_init_luxparams(rng::AbstractRNG, l::Any) = _init_luxparams(l)
_init_luxparams(l) = NamedTuple() 

_init_luxstate(rng::AbstractRNG, l::Any) = _init_luxstate(l)
_init_luxstate(l) = NamedTuple() 


# ---------- PolyLuxLayer
# the simplest lux layer implementation

# WARNING: All PolyLuxLayer are assumed to be not containing trainable parameters, so that they fallback to rrule 
# interface without ps and st, all trainable PolyLayers should overload Polynomials4ML.lux with wanted function 
# and return another non-PolyLuxLayer type
struct PolyLuxLayer{TB, LAB} <: AbstractLuxLayer
   basis::TB
end

function Base.show(io::IO, l::PolyLuxLayer{TB, LAB}) where {TB, LAB}
   print(io, "lux($LAB, $(l.basis))")
end

function Base.show(io::IO, l::PolyLuxLayer{TB, Symbol("")}) where {TB}
   print(io, "lux($(l.basis))")
end


Base.length(l::PolyLuxLayer) = length(l.basis)

initialparameters(rng::AbstractRNG, l::PolyLuxLayer) = 
      _init_luxparams(rng, l.basis)

initialstates(rng::AbstractRNG, l::PolyLuxLayer) = 
      _init_luxstate(rng, l.basis)

(l::PolyLuxLayer)(X, ps, st) = evaluate(l.basis, X, ps, st), st
