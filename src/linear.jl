import ChainRulesCore: rrule
using LuxCore
using Random

struct LinearLayer{FEATFIRST} <: AbstractExplicitLayer
   in_dim::Integer
   out_dim::Integer
end
 
LinearLayer(in_dim::Int, out_dim::Int; feature_first = false) = LinearLayer{feature_first}(in_dim, out_dim)

(l::LinearLayer)(x::AbstractMatrix, ps, st) = ps.W * parent(x), st
(l::LinearLayer{true})(x::AbstractMatrix, ps, st) = ps.W * parent(x), st
(l::LinearLayer{false})(x::AbstractMatrix, ps, st) = parent(x) * transpose(ps.W), st
 
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

function rrule(::typeof(LuxCore.apply), l::LinearLayer{false}, x::AbstractMatrix, ps, st)
   val = l(x, ps, st)
   function pb(A)
      return NoTangent(), NoTangent(), A[1] * ps.W, (W = A[1]' * x,), NoTangent()
   end
   return val, pb
end
