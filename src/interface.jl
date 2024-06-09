using StaticArrays: StaticArray, SVector, StaticVector, similar_type
using ChainRulesCore
import ChainRulesCore: rrule, frule 


abstract type AbstractP4MLLayer end 

"""
`abstract type AbstractP4MLBasis end`

Annotates types that map a low-dimensional input, scalar or `SVector`,
to a vector of scalars (feature vector, embedding, basis...). 
"""
abstract type AbstractP4MLBasis <: AbstractP4MLLayer end

"""
`abstract type AbstractP4MLTensor end`

Annotates layers that map a vector to a vector. Each of the vectors may 
represent a tensor (hence the name). Future interfaces may generalize the 
allowed dimensionality of inputs to allow tensorial shapes.
"""
abstract type AbstractP4MLTensor <: AbstractP4MLLayer end 

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

# SphericalCoords is defined here so it can be part of SINGLE 
# TODO: retire this as soon as we fully switched to SpheriCart 

"""
`struct SphericalCoords` : a simple datatype storing spherical coordinates
of a point (x,y,z) in the format `(r, cosφ, sinφ, cosθ, sinθ)`.
Use `spher2cart` and `cart2spher` to convert between cartesian and spherical
coordinates.
"""
struct SphericalCoords{T}
	r::T
	cosφ::T
	sinφ::T
	cosθ::T
	sinθ::T
end

# NOTE: Because we don't have a use-case for general arrays, we assume that a 
#       BATCH is always given as an AbstractVector of SINGLEs. But in principle 
#       we could allow for more generality here if there is demand for it. 
#       This is something to be explored in the future. 

const SINGLE = Union{Number, StaticArray, SphericalCoords}
const BATCH = AbstractVector{<: SINGLE}

const TupVec = Tuple{Vararg{AbstractVector}}
const TupMat = Tuple{Vararg{AbstractMatrix}}
const TupVecMat = Union{TupVec, TupMat}

# ---------------------------------------
# managing defaults for input-output types
# We deliberately provide no defaults for `valtype` but we try to guess 
# gradtype, hesstype and laplacetype based on the valtype. 

function _valtype end 
function _gradtype end 
function _hesstype end 
function _laplacetype end

# first redirect input to type 
_valtype(basis, x::SINGLE) = _valtype(basis, typeof(x))
_valtype(basis, x::BATCH) = _valtype(basis, eltype(x))
_gradtype(basis, x::SINGLE) = _gradtype(basis, typeof(x))
_gradtype(basis, x::BATCH) = _gradtype(basis, eltype(x))
_hesstype(basis, x::SINGLE) = _hesstype(basis, typeof(x))
_hesstype(basis, x::BATCH) = _hesstype(basis, eltype(x))
_laplacetype(basis, x::SINGLE) = _laplacetype(basis, typeof(x))
_laplacetype(basis, x::BATCH) = _laplacetype(basis, eltype(x))

# default grad types
_gradtype(basis::AbstractP4MLBasis, TX::Type{<:Number}) = 
      _valtype(basis, TX)

_gradtype(basis::AbstractP4MLBasis, Tx::Type{<: StaticArray}) = 
      StaticArrays.similar_type(Tx, 
                     promote_type(eltype(Tx), _valtype(basis, Tx)))

# default hessian types 
_hesstype(basis::AbstractP4MLBasis, TX::Type{<:Number}) = 
      _valtype(basis, TX)

_hesstype(basis::AbstractP4MLBasis, ::Type{SVector{N, T}}) where {N, T} = 
      SMatrix{N, N, promote_type(_valtype(basis, SVector{N, T}), T)}

# laplacian types       
_laplacetype(basis::AbstractP4MLBasis, TX::Type{<: Number}) = 
      eltype(_hesstype(basis, TX))

_laplacetype(basis::AbstractP4MLBasis, TX::Type{SVector{N, T}}) where {N, T} = 
      eltype(_hesstype(basis, TX))


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

function whatalloc(::typeof(evaluate_ed2!), basis::AbstractP4MLBasis, x)
   TV = _valtype(basis, x)
   TG = _gradtype(basis, x)
   TH = _hesstype(basis, x)
   sz = _out_size(basis, x)
   return (TV, sz...), (TG, sz...), (TH, sz...)
end

# a helper that converts all whatalloc outputs to tuple form 
function _tup_whatalloc(args...) 
   _to_tuple(wa::Tuple{Vararg{<: Tuple}}) = wa 
   _to_tuple(wa::Tuple{<: Type, Vararg{<: Integer}}) = (wa,)
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

(l::AbstractP4MLLayer)(args...) = 
      evaluate(l, args...)
            
evaluate(l::AbstractP4MLLayer, args...) = 
      _with_safe_alloc(evaluate!, l, args...) 

evaluate_ed(l::AbstractP4MLLayer, args...) = 
      _with_safe_alloc(evaluate_ed!, l, args...)

evaluate_ed2(l::AbstractP4MLLayer, args...) = 
      _with_safe_alloc(evaluate_ed2!, l, args...)

evaluate_d(l::AbstractP4MLLayer, args...) = 
      evaluate_ed(l, args...)[2] 

evaluate_dd(l::AbstractP4MLLayer, args...) = 
      evaluate_ed2(l, args...)[3] 

pullback_evaluate(∂X, l::AbstractP4MLLayer, args...) = 
      _with_safe_alloc(pullback_evaluate!, ∂X, l, args...)

pushforward_evaluate(l::AbstractP4MLLayer, args...) = 
      _with_safe_alloc(pushforward_evaluate!, l, args...)

pb_pb_evaluate(∂P, ∂X, l::AbstractP4MLLayer, args...) = 
      _with_safe_alloc(pb_pb_evaluate!, ∂P, ∂X, l, args...)


