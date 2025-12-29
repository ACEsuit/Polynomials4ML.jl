using StaticArrays: StaticArray, SVector, StaticVector, similar_type
using GPUArraysCore: AbstractGPUArray
using ChainRulesCore
import ChainRulesCore: rrule, frule 


# -------------------------------------------------------------------

# any number of SArray are interpreted as a "single" input 
# by contrast an AbstractVector OTHER THAN an SVector are interpreted as a 
# "batch" of inputs.
const SINGLE = Union{Number, SArray}

"""
`StaticBatch{N,T}` : an auxiliary StaticArray type that is distinct from 
`SVector{N,T}`. It can be used to create a batch of inputs of static size N. 
It is in used to convert function calls with single inputs to function calls 
with a batch of inputs. 
"""
struct StaticBatch{N, T} <: StaticVector{N, T}
	data::NTuple{N, T}
end 

StaticBatch(x::SINGLE) = StaticBatch((x,))

Base.getindex(b::StaticBatch, i::Int) = b.data[i]
Base.getindex(b::StaticBatch, i::Integer) = b.data[i]


# a "batch" of inputs
const BATCH = Union{AbstractVector{<: SINGLE}, StaticBatch{<: SINGLE}}


# ------------------------------------------------------------
# In-place CPU interface 


_reshape(A::AbstractArray, dims::NTuple{N, Int}) where N = 
      Base.ReshapedArray(A, dims, ())

# _reshape(A::AbstractArray, dims::NTuple{N, Int}) where N = 
#       reshape(A, dims) 

function evaluate!(P, basis::AbstractP4MLBasis, x::SINGLE, args...) 
	evaluate!(_reshape(P, (1, length(P))), basis, StaticBatch(x), args...)
	return P
end

function evaluate_ed!(P, dP, basis::AbstractP4MLBasis, x::SINGLE, args...) 
	evaluate_ed!(_reshape(P, (1, length(P))), _reshape(dP, (1, length(dP))), 
					 basis, StaticBatch(x), args...)
	return P, dP
end						 

function evaluate!(P, basis::AbstractP4MLBasis, x::BATCH, args...)
   @assert size(P, 1) >= length(x) 
   @assert size(P, 2) >= length(basis)
   _evaluate!(P, nothing, basis, x, args...)
   return P
end

function evaluate_ed!(P, dP, basis::AbstractP4MLBasis, x::BATCH, args...)
   @assert size(P, 1) >= length(x) 
   @assert size(P, 2) >= length(basis)
   @assert size(dP, 1) >= length(x) 
   @assert size(dP, 2) >= length(basis)
   _evaluate!(P, dP, basis, x, args...)
   return P, dP 
end

# default for inner kernel which assumes that the basis has no parameters 
# and no state (it can still store static parameters and state within 
# the basis object).
_evaluate!(P, dP, basis::AbstractP4MLBasis, X) = 
      _evaluate!(P, dP, basis, X, nothing, nothing)

# ------------------------------------------------------------
# In-place KA interface 
# evaluate! called with a GPUArray redirects to ka_evaluate!
# But ka_evaluate! can also be called with CPU arrays to enable testing 
# KA kernels also on the CPU. 

evaluate!(P::AbstractGPUArray, basis::AbstractP4MLBasis, x::BATCH, args...) = 
		ka_evaluate!(P, basis, x, args...)

evaluate_ed!(P::AbstractGPUArray, dP::AbstractGPUArray, 
             basis::AbstractP4MLBasis, x::BATCH, args...) = 
		ka_evaluate_ed!(P, dP, basis, x, args...)      

function ka_evaluate!(P, basis::AbstractP4MLBasis, x::BATCH, args...) 
	_ka_evaluate_launcher!(P, nothing, basis, x, args...)
	return P
end 

function ka_evaluate_ed!(P, dP, basis::AbstractP4MLBasis, x::BATCH, args...)
   _ka_evaluate_launcher!(P, dP, basis, x, args...)
   return P, dP 
end

function _ka_evaluate_launcher!(P, dP, basis::AbstractP4MLBasis, x, args...)
	nX = length(x) 
	len_basis = length(basis)
	
	@assert size(P, 1) >= nX 
	@assert size(P, 2) >= len_basis 
	if !isnothing(dP)
		@assert size(dP, 1) >= nX
		@assert size(dP, 2) >= len_basis
	end

	backend = KernelAbstractions.get_backend(P)

	kernel! = _ka_evaluate!(backend)
	kernel!(P, dP, basis, x; ndrange = (nX,))
	
	return nothing 
end
   


# -----------------------------------------------------------
# managing defaults for input-output types
# We deliberately provide no defaults for `valtype` but we try to guess 
# gradtype based on the valtype. 

"""
   _valtype(basis, x) 

If the intention is that `P = basis(x)` where `P` is a `Vector{T}` 
then `_valtype(basis, x)` should return `T`. 

Here, `x` can be a single input, a batch or a type. A new basis type `TB` only 
needs to implement `_valtype(::TB, x::Type)`. 
"""
function _valtype end 

"""
   _gradtype(basis, x)

If the intention is that `P, dP = evaluate_ed(basis, x)` then 
then `_gradtype(basis, x)` should return `etype(dP)`. 
"""
function _gradtype end 

# first redirect input to type 
_valtype(basis, x::SINGLE, args...) = _valtype(basis, typeof(x), args...)
_valtype(basis, x::BATCH, args...) = _valtype(basis, eltype(x), args...)
_gradtype(basis, x::SINGLE, args...) = _gradtype(basis, typeof(x), args...)
_gradtype(basis, x::BATCH, args...) = _gradtype(basis, eltype(x), args...)

_valtype(basis, TX::Type, ps, st) = _valtype(basis, TX)
_gradtype(basis, TX::Type, ps, st) = _gradtype(basis, TX)

# default grad types
_gradtype(basis::AbstractP4MLBasis, TX::Type{<:Number}, args...) = 
      _valtype(basis, TX, args...)

_gradtype(basis::AbstractP4MLBasis, Tx::Type{<: StaticArray}, args...) = 
      StaticArrays.similar_type(Tx, 
                     promote_type(eltype(Tx), _valtype(basis, Tx, args...)))

# ------------------------------------------------------------
# allocation interface & WithAlloc Interface 

_out_size(basis::AbstractP4MLBasis, x::SINGLE, args...) = (length(basis),)

_out_size(basis::AbstractP4MLBasis, X::BATCH, args...) = (length(X), length(basis))

function whatalloc(::typeof(evaluate!), basis::AbstractP4MLBasis, x, args...)
   T = _valtype(basis, x, args...)
   sz = _out_size(basis, x, args...)
   return (T, sz...) 
end

function whatalloc(::typeof(evaluate_ed!), basis::AbstractP4MLBasis, x, args...)
   TV = _valtype(basis, x, args...)
   TG = _gradtype(basis, x, args...)
   sz = _out_size(basis, x, args...)
   return (TV, sz...), (TG, sz...)
end

# a helper that converts all whatalloc outputs to tuple form 
function _tup_whatalloc(args...) 
   _to_tuple(wa::Tuple{Vararg{Tuple}}) = wa 
   _to_tuple(wa::Tuple{<: Type, Vararg{Integer}}) = (wa,)
   return _to_tuple(whatalloc(args...))
end


# _with_safe_alloc is a simple analogy of WithAlloc.@withalloc 
# that allocates standard arrays on the heap instead of using Bumper 
function _with_safe_alloc(fcall, args...) 
   allocinfo = _tup_whatalloc(fcall, args...)
   outputs = ntuple(i -> zeros(allocinfo[i]...), length(allocinfo))
   return fcall(outputs..., args...)
end

function _with_safe_alloc(fcall, basis::AbstractP4MLBasis, X::BATCH, args...) 
   _alczero(T, args...) = fill!( similar(X, T, args...), zero(T) )
      
   allocinfo = _tup_whatalloc(fcall, basis, X, args...)
   outputs = ntuple(i -> _alczero(allocinfo[i]...), length(allocinfo))
   return fcall(outputs..., basis, X, args...)
end

# --------------------------------------- 
# allocating evaluation interface 

(l::AbstractP4MLBasis)(args...) = 
      evaluate(l, args...)
            
evaluate(l::AbstractP4MLBasis, args...) = 
      _with_safe_alloc(evaluate!, l, args...) 

evaluate_ed(l::AbstractP4MLBasis, args...) = 
      _with_safe_alloc(evaluate_ed!, l, args...)

evaluate_d(l::AbstractP4MLBasis, args...) = 
      evaluate_ed(l, args...)[2] 


# ------------------------------------------------------------
# Lux interface


"""
a fall-back method for `initalparameters` that all AbstractP4MLBasis
should overload 
"""
_init_luxparams(rng::AbstractRNG, l::Any) = _init_luxparams(l)
_init_luxparams(l) = NamedTuple() 

_init_luxstate(rng::AbstractRNG, l::Any) = _init_luxstate(l)
_init_luxstate(l) = NamedTuple() 

initialparameters(rng::AbstractRNG, l::AbstractP4MLBasis) = 
      _init_luxparams(rng, l)

initialstates(rng::AbstractRNG, l::AbstractP4MLBasis) = 
      _init_luxstate(rng, l)

(l::AbstractP4MLBasis)(X, ps::NamedTuple, st::NamedTuple) = 
      evaluate(l, X, ps, st), st 
