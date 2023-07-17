import ChainRulesCore: rrule
using LuxCore
using Random

struct LinearLayer <: AbstractExplicitLayer 
   in_dim::Integer
   out_dim::Integer
end
 
function (l::LinearLayer)(x::AbstractMatrix, ps, st)
   return ps.W * parent(x), st
end
 
# Jerry: Maybe we should use Glorot Uniform if we have no idea about what we should use?
LuxCore.initialparameters(rng::AbstractRNG, l::LinearLayer) = ( W = randn(rng, l.out_dim, l.in_dim), )
LuxCore.initialstates(rng::AbstractRNG, l::LinearLayer) = NamedTuple()
 
function rrule(::typeof(LuxCore.apply), l::LinearLayer, x::AbstractMatrix, ps, st)
   val = l(x, ps, st)
   function pb(A)
      return NoTangent(), NoTangent(), ps.W' * A[1], (W = A[1] * x',), NoTangent()
   end
   return val, pb
end