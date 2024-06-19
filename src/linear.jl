using LinearAlgebra: mul!

export LinearLayer

"""
`struct LinearLayer` : This lux layer returns `W * x` if `feature_first` is true, otherwise it returns `x * transpose(W)`, where `W` is the weight matrix`.

* `x::AbstractMatrix` of size `(in_dim, N)` or `(N, in_dim)`, where `in_dim = feature dimension`, `N = batch size`
* `W::AbstractMatrix` of size `(out_dim, in_dim)`

### Constructor 
```julia 
LinearLayer(in_dim, out_dim; feature_first = false)
```

### Example
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
   @reqfields()
end

LinearLayer(in_dim::Int, out_dim::Int; feature_first = false) = LinearLayer{feature_first}(in_dim, out_dim, _make_reqfields()...)

LuxCore.initialparameters(rng::AbstractRNG, l::LinearLayer) = ( W = randn(rng, l.out_dim, l.in_dim), )
LuxCore.initialstates(rng::AbstractRNG, l::LinearLayer) = NamedTuple()

# ----------------------- evaluation and allocation interfaces 

_valtype(l::LinearLayer, x::AbstractArray, ps, st)  = promote_type(eltype(x), eltype(ps.W))
_gradtype(l::LinearLayer, x, ps, st) = promote_type(eltype(x), eltype(ps.W))

_out_size(l::LinearLayer, x::AbstractVector, ps, st) = (l.out_dim, )
_out_size(l::LinearLayer{true}, x::AbstractMatrix, ps, st) = (l.out_dim, size(x, 2))
_out_size(l::LinearLayer{false}, x::AbstractMatrix, ps, st) = (size(x, 1), l.out_dim)

(l::LinearLayer)(args...) = evaluate(l, args...)
evaluate(l::LinearLayer, args...) = _with_safe_alloc(evaluate!, l, args...) 

function whatalloc(::typeof(evaluate!), l::LinearLayer, x::AbstractArray, ps, st)
   TV = _valtype(l, x, ps, st)
   sz = _out_size(l, x, ps, st)
   return (TV, sz...)
end

# -------------- kernels

function evaluate!(out, l::LinearLayer, x::AbstractVecOrMat, ps, st)
   mul!(out, ps.W, x)
   return out, st
end

function evaluate!(out, l::LinearLayer{false}, x::AbstractMatrix, ps, st)
   mul!(out, x, transpose(PtrArray(ps.W)))
   return out, st
end

# -------------------- reverse mode gradient

function pullback(∂A, l::LinearLayer, x, ps)
   TA = promote_type(eltype(x), eltype(ps.W))
   ∂x, ∂W = zeros(TA, size(x)), zeros(TA, size(ps.W))
   pullback!(∂x, ∂W, ∂A, l, x, ps)
   return ∂x, (W = ∂W,)
end

function pullback!(∂x, ∂W, ∂A, l::LinearLayer, x, ps)
   mul!(∂x, ps.W', ∂A)
   mul!(∂W, ∂A, x')
   return ∂x, ∂W
end

function pullback!(∂x, ∂W, ∂A, l::LinearLayer{false}, x::AbstractMatrix, ps)
   mul!(∂x, ∂A, ps.W)
   mul!(∂W, transpose(PtrArray(∂A)), x)
   return ∂x, ∂W
end

# --------------------- connect with ChainRules 
# can this be generalized again? 
# TODO: check whether we can do this without multiple dispatch on vec/mat without loss of performance

import ChainRulesCore: rrule, NoTangent

function rrule(::typeof(evaluate), l::LinearLayer, x::AbstractVecOrMat, ps, st)
   val = l(x, ps, st)
   function pb(∂Ast)
      return NoTangent(), NoTangent(), pullback(∂Ast[1], l, x, ps)..., NoTangent()
   end
   return val, pb
end