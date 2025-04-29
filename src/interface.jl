using StaticArrays: StaticArray, SVector, StaticVector, similar_type
using GPUArraysCore: AbstractGPUArray
using ChainRulesCore
import ChainRulesCore: rrule, frule 



# ---------------------------------------------------------------------------
# some helpers to deal with the required fields: 
# TODO: now that there is only meta, should this be removed? 

using ACEbase: @def 

const META = Dict{String, Any} 
_makemeta() = Dict{String, Any}()

@def reqfields begin
   meta::META
end

_make_reqfields() = (_makemeta(), )


# -------------------------------------------------------------------

# a "single" input
const SINGLE = Union{Number, SArray}

"""
`StaticBatch{N,T}` : an auxiliary StaticArray type that is distinct from 
`SVector{N,T}`. It can be used to create a batch of inputs of static size N. 
It is in particular used to convert function calls with single inputs to 
function calls with a batch of inputs. 
"""
struct StaticBatch{N, T} <: StaticVector{N, T}
	data::NTuple{N, T}
end 

StaticBatch(x::SINGLE) = StaticBatch((x,))

Base.getindex(b::StaticBatch, i::Int) = b.data[i]
Base.getindex(b::StaticBatch, i::Integer) = b.data[i]


# a "batch" of inputs
const BATCH = Union{AbstractVector{<: SINGLE}, StaticBatch{<: SINGLE}}

# TODO: check that we can remove this 
# const TupVec = Tuple{Vararg{AbstractVector}}
# const TupMat = Tuple{Vararg{AbstractMatrix}}
# const TupVecMat = Union{TupVec, TupMat}



# ------------------------------------------------------------
# In-place CPU interface 

function evaluate!(P, basis::AbstractP4MLBasis, x::SINGLE) 
	evaluate!(reshape(P, 1, :), basis, StaticBatch(x))
	return P
end

function evaluate_ed!(P, dP, basis::AbstractP4MLBasis, x::SINGLE) 
	evaluate_ed!(reshape(P, 1, :), reshape(dP, 1, :), 
					 basis, StaticBatch(x))
	return P, dP
end					 

function evaluate!(P, basis::AbstractP4MLBasis, x::BATCH)
   @assert size(P, 1) >= length(x) 
   @assert size(P, 2) >= length(basis)
   _evaluate!(P, nothing, basis, x)
   return P
end

function evaluate_ed!(P, dP, basis::AbstractP4MLBasis, x::BATCH)
   @assert size(P, 1) >= length(x) 
   @assert size(P, 2) >= length(basis)
   @assert size(dP, 1) >= length(x) 
   @assert size(dP, 2) >= length(basis)
   _evaluate!(P, dP, basis, x)
   return P, dP 
end

# ------------------------------------------------------------
# In-place KA interface 
# evaluate! called with a GPUArray redirects to ka_evaluate!
# But ka_evaluate! can also be called with CPU arrays to enable testing 
# KA kernels also on the CPU. 

evaluate!(P::AbstractGPUArray, basis::AbstractP4MLBasis, x::BATCH) = 
		ka_evaluate!(P, basis, x)

evaluate_ed!(P::AbstractGPUArray, dP::AbstractGPUArray, basis::AbstractP4MLBasis, x::BATCH) = 
		ka_evaluate_ed!(P, dP, basis, x)      

function ka_evaluate!(P, basis::AbstractP4MLBasis, x::BATCH) 
	_ka_evaluate_launcher!(P, nothing, basis, x)
	return P
end 

function ka_evaluate_ed!(P, dP, basis::AbstractP4MLBasis, x::BATCH)
   _ka_evaluate_launcher!(P, dP, basis, x)
   return P, dP 
end

function _ka_evaluate_launcher!(P, dP, basis::AbstractP4MLBasis, x)
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
# gradtype, hesstype and laplacetype based on the valtype. 

function _valtype end 
function _gradtype end 

# first redirect input to type 
_valtype(basis, x::SINGLE) = _valtype(basis, typeof(x))
_valtype(basis, x::BATCH) = _valtype(basis, eltype(x))
_gradtype(basis, x::SINGLE) = _gradtype(basis, typeof(x))
_gradtype(basis, x::BATCH) = _gradtype(basis, eltype(x))

# default grad types
_gradtype(basis::AbstractP4MLBasis, TX::Type{<:Number}) = 
      _valtype(basis, TX)

_gradtype(basis::AbstractP4MLBasis, Tx::Type{<: StaticArray}) = 
      StaticArrays.similar_type(Tx, 
                     promote_type(eltype(Tx), _valtype(basis, Tx)))

# ------------------------------------------------------------
# allocation interface & WithAlloc Interface 

_out_size(basis::AbstractP4MLBasis, x::SINGLE) = (length(basis),)

_out_size(basis::AbstractP4MLBasis, X::BATCH) = (length(X), length(basis))

function whatalloc(::typeof(evaluate!), basis::AbstractP4MLBasis, x)
   T = _valtype(basis, x)
   sz = _out_size(basis, x)
   return (T, sz...) 
end

function whatalloc(::typeof(evaluate_ed!), basis::AbstractP4MLBasis, x)
   TV = _valtype(basis, x)
   TG = _gradtype(basis, x)
   sz = _out_size(basis, x)
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

