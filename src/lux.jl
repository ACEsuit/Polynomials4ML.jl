
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
               name = String(nameof(typeof(basis))), 
               meta = Dict{String, Any}("name" => name), 
            )
   @assert haskey(meta, "name")
   return PolyLuxLayer(basis, meta)
end

"""
a fall-back method for `initalparameters` that all AbstractPoly4MLBasis
should overload 
"""
_init_luxparams(rng::AbstractRNG, basis::Any) = _init_luxparams(basis)
_init_luxparams(basis) = NamedTuple() 

_init_luxstate(rng::AbstractRNG, basis) = _init_luxstate(basis)
_init_luxstate(basis) = _init_default_luxstate()
_init_default_luxstate() = ( tmp = ArrayPool(FlexArray), 
                           cache = ArrayPool(FlexArrayCache) )



# ---------- PolyLuxLayer
# the simplest lux layer implementation 



struct PolyLuxLayer{TB} <: AbstractExplicitLayer
   basis::TB
   meta::Dict{String, Any}
end

function Base.show(io::IO, l::PolyLuxLayer)
   print(io, "PolyLuxLayer($(l.meta["name"]))")
end


Base.length(l::PolyLuxLayer) = length(l.basis)

initialparameters(rng::AbstractRNG, l::PolyLuxLayer) = _init_luxparams(rng, l.basis)

initialstates(rng::AbstractRNG, l::PolyLuxLayer) = _init_luxstate(rng, l.basis)

(l::PolyLuxLayer)(args...) = evaluate(l, args...)

function evaluate(l::PolyLuxLayer, X, ps, st)
   
   # TODO: after we make sure we want to migrate to HyperDualNumbers in any cases we can ignore_derivatives from ChainRulesCore
   #B = ChainRulesCore.ignore_derivatives() do 
   #   evaluate(l.basis, X)
   #end
   B = evaluate(l.basis, X)
   return B, st 
end 


