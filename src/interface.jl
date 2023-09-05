using StaticArrays: StaticArray, SVector, StaticVector, similar_type
using ChainRulesCore

abstract type AbstractPoly4MLBasis end

abstract type ScalarPoly4MLBasis <: AbstractPoly4MLBasis end

abstract type SVecPoly4MLBasis <: AbstractPoly4MLBasis end

# ---------------------------------------
# some helpers to deal with the three required arrays: 

using ACEbase: @def 

const POOL = TSafe{ArrayPool{FlexArrayCache}}
_makepool() = TSafe(ArrayPool(FlexArrayCache))

const TMP = TSafe{ArrayPool{FlexArray}}
_maketmp() = TSafe(ArrayPool(FlexArray))

const META = Dict{String, Any} 
_makemeta() = Dict{String, Any}()


@def reqfields begin
   pool::POOL 
   tmp::TMP
   meta::META
end

_make_reqfields() = _makepool(), _maketmp(), _makemeta()


# ---------------------------------------

# SphericalCoords is defined here so it can be part of SINGLE 

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

const SINGLE = Union{Number, StaticArray, SphericalCoords}
const BATCH = AbstractVector{<: SINGLE}

const TupVec = Tuple{Vararg{AbstractVector}}
const TupMat = Tuple{Vararg{AbstractMatrix}}
const TupVecMat = Union{TupVec, TupMat}
# ---------------------------------------
# managing defaults for input-output types

function _valtype end 
function _gradtype end 
function _hesstype end 
function _laplacetype end 

_valtype(basis::AbstractPoly4MLBasis, x::SINGLE) = 
      _valtype(basis, typeof(x)) 

_gradtype(basis::AbstractPoly4MLBasis, x::SINGLE) = 
      _gradtype(basis, typeof(x)) 

_hesstype(basis::AbstractPoly4MLBasis, x::SINGLE) = 
      _hesstype(basis, typeof(x)) 

      
_gradtype(basis::AbstractPoly4MLBasis, TX::Type{<:Number}) = 
      _valtype(basis, TX)

_gradtype(basis::AbstractPoly4MLBasis, Tx::Type{<: StaticArray}) = 
      StaticArrays.similar_type(Tx, 
                     promote_type(eltype(Tx), _valtype(basis, Tx)))

_hesstype(basis::AbstractPoly4MLBasis, TX::Type{<:Number}) = 
      _valtype(basis, TX)

_hesstype(basis::AbstractPoly4MLBasis, ::Type{SVector{N, T}}) where {N, T} = 
      SMatrix{N, N, promote_type(_valtype(basis, SVector{N, T}), T)}


_laplacetype(basis::AbstractPoly4MLBasis, TX::Type{<: Number}) = 
      eltype(_hesstype(basis, TX))

_laplacetype(basis::AbstractPoly4MLBasis, TX::Type{SVector{N, T}}) where {N, T} = 
      eltype(_hesstype(basis, T))


_valtype(basis::AbstractPoly4MLBasis, X::BATCH) = 
      _valtype(basis, eltype(X))

_gradtype(basis::AbstractPoly4MLBasis, X::BATCH) = 
      _gradtype(basis, eltype(X))

_hesstype(basis::AbstractPoly4MLBasis, X::BATCH) = 
      _hesstype(basis, eltype(X))

_laplacetype(basis::AbstractPoly4MLBasis, X::BATCH) = 
      _laplacetype(basis, eltype(X))

# --------------------------------------- 
# allocation interface interface 

_out_size(basis::AbstractPoly4MLBasis, x::SINGLE) = (length(basis),)
_out_size(basis::AbstractPoly4MLBasis, X::BATCH) = (length(X), length(basis))

_outsym(x::SINGLE) = :out 
_outsym(X::BATCH) = :outb

_alloc(basis::AbstractPoly4MLBasis, X) = 
      acquire!(basis.pool, _outsym(X), _out_size(basis, X), _valtype(basis, X) )

_alloc_d(basis::AbstractPoly4MLBasis, X) = 
      acquire!(basis.pool, _outsym(X), _out_size(basis, X), _gradtype(basis, X) )

_alloc_dd(basis::AbstractPoly4MLBasis, X) = 
      acquire!(basis.pool, _outsym(X), _out_size(basis, X), _gradtype(basis, X) )

_alloc_ed(basis::AbstractPoly4MLBasis, x) = 
      _alloc(basis, x), _alloc_d(basis, x)

_alloc_ed2(basis::AbstractPoly4MLBasis, x) = 
      _alloc(basis, x), _alloc_d(basis, x), _alloc_dd(basis, x)

# --------------------------------------- 
# evaluation interface 

(basis::AbstractPoly4MLBasis)(x) = evaluate(basis, x)
            
function evaluate(basis::AbstractPoly4MLBasis, x) 
   B = _alloc(basis, x)
   evaluate!(unwrap(B), basis, x)
   return B 
end

function evaluate_ed(basis::AbstractPoly4MLBasis, x) 
   B, dB = _alloc_ed(basis, x)
   evaluate_ed!(unwrap(B), unwrap(dB), basis, x)
   return B, dB
end 

function evaluate_ed2(basis::AbstractPoly4MLBasis, x)
   B, dB, ddB = _alloc_ed2(basis, x)
   evaluate_ed2!(unwrap(B), unwrap(dB), unwrap(ddB), basis, x)
   return B, dB, ddB
end

evaluate_d(basis::AbstractPoly4MLBasis, x) = evaluate_ed(basis, x)[2] 

evaluate_dd(basis::AbstractPoly4MLBasis, x) = evaluate_ed2(basis, x)[3] 


# the next set of interface functions are in-place but work a little 
# differently : by using a FlexArray as input the evaluation function 
# can extract the right output array from it and then return it. 

# this is experimental and we don't require semver-stability for now ... 

# _alloc(flex::FlexArray, basis::AbstractPoly4MLBasis, X) = 
#       acquire!(flex, _out_size(basis, X), _valtype(basis, X))

# _alloc_d(flex_d::FlexArray, basis::AbstractPoly4MLBasis, X) = 
#       acquire!(flex_d, _out_size(basis, X), _gradtype(basis, X))

# _alloc_dd(flex_dd::FlexArray, basis::AbstractPoly4MLBasis, X) = 
#       acquire!(flex_dd, _out_size(basis, X), _hesstype(basis, X))

# _alloc_ed(flex::FlexArray, flex_d::FlexArray, basis::AbstractPoly4MLBasis, x) = 
#       _alloc(flex, basis, x), _alloc_d(flex_d, basis, x)      

# _alloc_ed2(flex::FlexArray, flex_d::FlexArray, 
#            flex_dd::FlexArray, basis::AbstractPoly4MLBasis, x) = 
#       _alloc(flex, basis, x), _alloc_d(flex_d, basis, x), _alloc_dd(flex_dd, basis, x)      


# function evaluate!(flex_B::FlexArray, basis::AbstractPoly4MLBasis, x) 
#    B = _alloc(flex_B, basis, x)
#    evaluate!(B, basis, x)
#    return B 
# end

# function evaluate_ed!(flex_B::FlexArray, 
#                       flex_dB::FlexArray, 
#                       basis::AbstractPoly4MLBasis, x) 
#    B, dB = _alloc_ed(flex_B, flex_dB, basis, x)
#    evaluate_ed!(B, dB, basis, x)
#    return B, dB
# end 

# function evaluate_ed2!(flex_B::FlexArray, 
#                        flex_dB::FlexArray, 
#                        flex_ddB::FlexArray,
#                        basis::AbstractPoly4MLBasis, x)
#    B, dB, ddB = _alloc_ed2(flex_B, flex_dB, flex_ddB, basis, x)
#    evaluate_ed2!(B, dB, ddB, basis, x)
#    return B, dB, ddB
# end

# --------------------------------------- 
# general rrules and frules interface for ChainRulesCore

import ChainRulesCore: rrule

# ∂_xa ( ∂P : P ) = ∑_ij ∂_xa ( ∂P_ij * P_ij ) 
#                 = ∑_ij ∂P_ij * ∂_xa ( P_ij )
#                 = ∑_ij ∂P_ij * dP_ij δ_ia
function rrule(::typeof(evaluate), 
                  basis::ScalarPoly4MLBasis, 
                  R::AbstractVector{<: Real})
   P = evaluate(basis, R)
   return P, ∂ -> (NoTangent(), NoTangent(), _evaluate_pb(basis, ∂, R))
end

function _evaluate_pb(basis::ScalarPoly4MLBasis, ∂, X::AbstractVector{<: Real})
   P, dP = evaluate_ed(basis, X)
   @assert size(∂) == (length(X), length(basis))
   T∂R = promote_type(eltype(∂), eltype(dP))
   ∂X = zeros(T∂R, length(X))
   # manual loops to avoid any broadcasting of StrideArrays 
   for n = 1:size(dP, 2)
         @simd ivdep for a = 1:length(X)
             ∂X[a] += dP[a, n] * ∂[a, n]
         end
   end
   
   return ∂X
end


function rrule(::typeof(_evaluate_pb),
               basis::ScalarPoly4MLBasis, ∂, X::AbstractVector{<: Real})
   ∂X = _evaluate_pb(basis, ∂, X)
   return ∂X, ∂2 -> (NoTangent(), NoTangent(), 
                     _evaluate_pb2(basis, ∂2, ∂, X)...)
end


function _evaluate_pb2(basis::ScalarPoly4MLBasis, 
                        ∂2, ∂, X::AbstractVector{<: Real})
   @info("_evaluate_pb2")                        
   Nbasis = length(basis)
   Nx = length(X)                        
   P, dP, ddP = evaluate_ed2(basis, X)
   # ∂2 is the dual to ∂X ∈ ℝ^N, N = length(X) 
   @assert ∂2 isa AbstractVector 
   @assert length(∂2) == Nx
   @assert size(∂) == (Nx, Nbasis)

   ∂2_∂ = zeros(size(∂))
   ∂2_X = zeros(length(X))

   for n = 1:Nbasis 
      @simd ivdep for a = 1:Nx 
         ∂2_∂[a, n] = ∂2[a] * dP[a, n]
         ∂2_X[a] += ∂2[a] * ddP[a, n] * ∂[a, n]
      end
   end

   release!(P); release!(dP); release!(ddP)

   return ∂2_∂, ∂2_X
end