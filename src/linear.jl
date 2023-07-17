import ChainRulesCore: rrule

struct LinearLayer <: AbstractExplicitLayer 
   in_dim::Integer
   out_dim::Integer
end
 
function (l::LinearLayer)(x::AbstractMatrix, ps, st)
   return parent(x) * ps.W, st
end
 
# Jerry: Maybe we should use Glorot Uniform if we have no idea about what we should use?
LuxCore.initialparameters(rng::AbstractRNG, l::LinearLayer) = ( W = randn(rng, l.out_dim, l.in_dim), )
LuxCore.initialstates(rng::AbstractRNG, l::LinearLayer) = NamedTuple()
 
function rrule(::typeof(Lux.apply), l::LinearLayer, x::AbstractMatrix, ps, st)
   val = l(x, ps, st)
   function pb(A)
      return NoTangent(), NoTangent(), A[1] * ps.W', (W = x' * A[1],), NoTangent()
   end
   return val, pb
end