using StaticArrays: StaticArray, SVector, StaticVector, similar_type
using ChainRulesCore

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

# ---------------------------------------
# some helpers to deal with the required fields: 
# TODO: now that there is only meta, should this be removed? 

using ACEbase: @def 

const META = Dict{String, Any} 
_makemeta() = Dict{String, Any}()

@def reqfields begin
   meta::META
end

_make_reqfields() = (_makemeta(), )


# ---------------------------------------

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
const EITHER = Union{SINGLE, BATCH}

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

(basis::AbstractP4MLLayer)(args...) = evaluate(basis, args...)
            
evaluate(args...) = _with_safe_alloc(evaluate!, args...) 

evaluate_ed(args...) = _with_safe_alloc(evaluate_ed!, args...)

evaluate_ed2(args...) = _with_safe_alloc(evaluate_ed2!, args...)

evaluate_d(basis, args...) = evaluate_ed(basis, args...)[2] 

evaluate_dd(basis, args...) = evaluate_ed2(basis, args...)[3] 

pullback_evaluate(args...) = _with_safe_alloc(pullback_evaluate!, args...)

pushforward_evaluate(args...) = _with_safe_alloc(pushforward_evaluate!, args...)

pb_pb_evaluate(args...) = _with_safe_alloc(pb_pb_evaluate!, args...)


# --------------------------------------- 
# general rrules and frules interface for ChainRulesCore

import ChainRulesCore: rrule

# ∂_xa ( ∂P : P ) = ∑_ij ∂_xa ( ∂P_ij * P_ij ) 
#                 = ∑_ij ∂P_ij * ∂_xa ( P_ij )
#                 = ∑_ij ∂P_ij * dP_ij δ_ia
function rrule(::typeof(evaluate), 
                  basis::AbstractP4MLBasis, 
                  R::AbstractVector{<: Real})
   P = evaluate(basis, R)
   return P, ∂P -> (NoTangent(), NoTangent(), pullback_evaluate(∂P, basis, R))
end

function pullback_evaluate(∂P, basis::AbstractP4MLBasis, X::AbstractVector{<: Real})
   P, dP = evaluate_ed(basis, X)
   @assert size(∂P) == (length(X), length(basis))
   T∂R = promote_type(eltype(∂P), eltype(dP))
   ∂X = zeros(T∂R, length(X))
   # manual loops to avoid any broadcasting of StrideArrays 
   for n = 1:size(dP, 2)
         @simd ivdep for a = 1:length(X)
             ∂X[a] += dP[a, n] * ∂P[a, n]
         end
   end
   return ∂X
end


function rrule(::typeof(pullback_evaluate),
               ∂P, basis::AbstractP4MLBasis, X::AbstractVector{<: Real})
   ∂X = pullback_evaluate(∂P, basis, X)
   function _pb(∂2)
      ∂∂P, ∂X = pb_pb_evaluate(∂2, ∂P, basis, X)
      return NoTangent(), ∂∂P, NoTangent(), ∂X             
   end
   return ∂X, _pb 
end


function pb_pb_evaluate(∂2, ∂P, basis::AbstractP4MLBasis, 
                        X::AbstractVector{<: Real})
   # @info("_evaluate_pb2")                        
   Nbasis = length(basis)
   Nx = length(X)                        
   P, dP, ddP = evaluate_ed2(basis, X)
   # ∂2 is the dual to ∂X ∈ ℝ^N, N = length(X) 
   @assert ∂2 isa AbstractVector 
   @assert length(∂2) == Nx
   @assert size(∂P) == (Nx, Nbasis)

   ∂2_∂ = zeros(size(∂P))
   ∂2_X = zeros(length(X))

   for n = 1:Nbasis 
      @simd ivdep for a = 1:Nx 
         ∂2_∂[a, n] = ∂2[a] * dP[a, n]
         ∂2_X[a] += ∂2[a] * ddP[a, n] * ∂P[a, n]
      end
   end
   return ∂2_∂, ∂2_X
end

