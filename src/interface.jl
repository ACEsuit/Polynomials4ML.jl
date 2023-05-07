using StaticArrays: StaticArray, SVector, StaticVector, similar_type

abstract type AbstractPoly4MLBasis end

# NOTE: Because we don't have a use-case for general arrays, we assume that a 
#       BATCH is always given as an AbstractVector of SINGLEs. But in principle 
#       we could allow for more generality here if there is demand for it. 

const SINGLE = Union{Number, StaticArray}
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

_alloc(basis::AbstractPoly4MLBasis, x::SINGLE) = 
      Vector{ _valtype(basis, x) }(undef, length(basis))

_alloc(basis::AbstractPoly4MLBasis, X::BATCH) = 
      Array{ _valtype(basis, X[1]) }(undef, (length(X), length(basis)))

_alloc_d(basis::AbstractPoly4MLBasis, x::SINGLE) = 
      Vector{ _gradtype(basis, x) }(undef, length(basis))

_alloc_d(basis::AbstractPoly4MLBasis, X::BATCH) = 
      Array{ _gradtype(basis, X[1]) }(undef, (length(X), length(basis)))

_alloc_dd(basis::AbstractPoly4MLBasis, x::SINGLE) = 
      Vector{ _hesstype(basis, x) }(undef, length(basis))

_alloc_dd(basis::AbstractPoly4MLBasis, X::BATCH) = 
      Array{ _hesstype(basis, X[1]) }(undef, (length(X), length(basis)))

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

