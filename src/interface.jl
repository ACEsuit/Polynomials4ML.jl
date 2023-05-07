using StaticArrays: StaticArray, SVector, StaticVector, similar_type

abstract type AbstractPoly4MLBasis end

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

# ---------------------------------------
# managing defaults for input-output types

function _valtype end 
function _gradtype end 
function _hesstype end 
function _laplacetype end 

_valtype(basis::AbstractPoly4MLBasis, x::Number) = 
      _valtype(basis, typeof(x))

_gradtype(basis::AbstractPoly4MLBasis, x::Number) = 
      promote_type(_valtype(basis, x), typeof(x))

_gradtype(basis::AbstractPoly4MLBasis, x::StaticArray) = 
      StaticArrays.similar_type(typeof(x), 
                                promote_type(eltype(x), _valtype(basis, x)))

_hesstype(basis::AbstractPoly4MLBasis, x::Number) = 
      promote_type(_valtype(basis, x), typeof(x))

_hesstype(basis::AbstractPoly4MLBasis, x::SVector{N}) where {N} = 
      SMatrix{N, N, promote_type(_valtype(basis, x), eltype(x))}

_laplacetype(basis::AbstractPoly4MLBasis, x::Number) = 
      _hesstype(basis, x)

_laplacetype(basis::AbstractPoly4MLBasis, x::StaticVector) = 
      eltype(_hesstype(basis, x))


# --------------------------------------- 
# allocation interface interface 

# TODO: here we should work with eltype(BATCH) instead of X[1]

_out_size(basis::AbstractPoly4MLBasis, x::SINGLE) = length(basis)
_out_size(basis::AbstractPoly4MLBasis, X::BATCH) = (length(X), length(basis))

_alloc(basis::AbstractPoly4MLBasis, X) = 
      Array{ _valtype(basis, X) }(undef, _out_size(basis, X))

_alloc_d(basis::AbstractPoly4MLBasis, X) = 
      Array{ _gradtype(basis, X[1]) }(undef, _out_size(basis, X))

_alloc_dd(basis::AbstractPoly4MLBasis, X) = 
      Array{ _hesstype(basis, X[1]) }(undef, _out_size(basis, X))

_alloc_ed(basis::AbstractPoly4MLBasis, x) = 
      _alloc(basis, x), _alloc_d(basis, x)

_alloc_ed2(basis::AbstractPoly4MLBasis, x) = 
   _alloc(basis, x), _alloc_d(basis, x), _alloc_dd(basis, x)

# --------------------------------------- 
# evaluation interface 

(basis::AbstractPoly4MLBasis)(x) = evaluate(basis, x)
            
function evaluate(basis::AbstractPoly4MLBasis, x) 
   B = _alloc(basis, x)
   evaluate!(B, basis, x)
   return B 
end

function evaluate_ed(basis::AbstractPoly4MLBasis, x) 
   B, dB = _alloc_ed(basis, x)
   evaluate_ed!(B, dB, basis, x)
   return B, dB
end 

evaluate_d(basis::AbstractPoly4MLBasis, x) = evaluate_ed(basis, x)[2] 

function evaluate_ed2(basis::AbstractPoly4MLBasis, x)
   B, dB, ddB = _alloc_ed2(basis, x)
   evaluate_ed2!(B, dB, ddB, basis, x)
   return B, dB, ddB
end

evaluate_dd(basis::AbstractPoly4MLBasis, x) = evaluate_ed2(basis, x)[3] 


# the next set of interface functions are in-place but work a little 
# differently : by using a FlexTempArray as input the evaluation function 
# can extract the right output array from it and then return it. 

_alloc(flex::FlexTempArray, basis::AbstractPoly4MLBasis, X) = 
      aquire!(flex, _out_size(basis, X), _valtype(basis, X))

_alloc_d(flex_d::FlexTempArray, basis::AbstractPoly4MLBasis, X) = 
      aquire!(flex_d, _out_size(basis, X), _gradtype(basis, X))

_alloc_dd(flex_dd::FlexTempArray, basis::AbstractPoly4MLBasis, X) = 
      aquire!(flex_dd, _out_size(basis, X), _hesstype(basis, X))

_alloc_ed(flex::FlexTempArray, flex_d::FlexTempArray, basis::AbstractPoly4MLBasis, x) = 
      _alloc(flex, basis, x), _alloc_d(flex_d, basis, x)      

_alloc_ed2(flex::FlexTempArray, flex_d::FlexTempArray, 
           flex_dd::FlexTempArray, basis::AbstractPoly4MLBasis, x) = 
      _alloc(flex, basis, x), _alloc_d(flex_d, basis, x), _alloc_dd(flex_dd, basis, x)      


function evaluate!(flex_B::FlexTempArray, basis::AbstractPoly4MLBasis, x) 
   B = _alloc(flex_B, basis, x)
   evaluate!(B, basis, x)
   return B 
end

function evaluate_ed!(flex_B::FlexTempArray, 
                      flex_dB::FlexTempArray, 
                      basis::AbstractPoly4MLBasis, x) 
   B, dB = _alloc_ed(flex_B, flex_dB, basis, x)
   evaluate_ed!(B, dB, basis, x)
   return B, dB
end 

function evaluate_ed2!(flex_B::FlexTempArray, 
                       flex_dB::FlexTempArray, 
                       flex_ddB::FlexTempArray,
                       basis::AbstractPoly4MLBasis, x)
   B, dB, ddB = _alloc_ed2(flex_B, flex_dB, flex_ddB, basis, x)
   evaluate_ed2!(B, dB, ddB, basis, x)
   return B, dB, ddB
end