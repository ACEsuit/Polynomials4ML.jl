import ChainRulesCore: rrule
using LuxCore
using Random
using LinearAlgebra: mul!

"""
`struct LinearLayer` : This lux layer returns `W * x` if `feature_first` is true, otherwise it returns `x * transpose(W)`, where `W` is the weight matrix`
where 
* `x::AbstractMatrix` of size `(in_dim, N)` or `(N, in_dim)`, where `in_dim = feature dimension`, `N = batch size`
* `W::AbstractMatrix` of size `(out_dim, in_dim)`

### Constructor 
```julia 
LinearLayer(in_dim, out_dim; feature_first = false)
```

For example
```julia 
in_d, out_d = 4, 3 # feature dimensions
N = 10 # batch size

# feature_first = true
l = P4ML.LinearLayer(in_d, out_d; feature_first = true)
ps, st = LuxCore.setup(MersenneTwister(1234), l)
x = randn(in_d, N) # feature-first
out, st = l(x, ps, st)
println(out == W * x) # true

# feature_first = false
l2 = P4ML.LinearLayer(in_d, out_d; feature_first = true)
ps2, st2 = LuxCore.setup(MersenneTwister(1234), l2)
x = randn(N, in_d) # batch-first
out, st = l(x, ps, st)
println(out == x * transpose(W))) # true
```
"""
struct LinearLayer{FEATFIRST} <: AbstractExplicitLayer
   in_dim::Integer
   out_dim::Integer
   use_cache::Bool
   @reqfields()
end

LinearLayer(in_dim::Int, out_dim::Int; feature_first = false, use_cache = true) = LinearLayer{feature_first}(in_dim, out_dim, use_cache, _make_reqfields()...)


(l::LinearLayer)(x::AbstractVector, ps, st) = begin
   out = acquire!(st.pool, :A, (l.out_dim, ), eltype(x)); 
   mul!(unwrap(out), ps.W, unwrap(x)); release!(x); 
   return out, st
end

(l::LinearLayer{true})(x::AbstractMatrix, ps, st) = begin 
   out = acquire!(st.pool, :bA, (l.out_dim, size(x, 2)), eltype(x)); 
   mul!(unwrap(out), ps.W, unwrap(x)); release!(x); 
   return out, st
end

(l::LinearLayer{false})(x::AbstractMatrix, ps, st) = begin
   out = acquire!(st.pool, :bA, (size(x, 1), l.out_dim), eltype(x)); 
   mul!(unwrap(out), unwrap(x), transpose(ps.W)); release!(x);
   return out, st
end
 
# Jerry: Maybe we should use Glorot Uniform if we have no idea about what we should use?
LuxCore.initialparameters(rng::AbstractRNG, l::LinearLayer) = ( W = randn(rng, l.out_dim, l.in_dim), )
LuxCore.initialstates(rng::AbstractRNG, l::LinearLayer) = ( l.use_cache ?  (pool =  ArrayPool(FlexArrayCache), ) : (pool =  ArrayPool(FlexArray), ))
 
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
