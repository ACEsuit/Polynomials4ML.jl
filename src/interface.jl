
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

_gradtype(basis::AbstractPoly4MLBasis, x::Number) = 
      promote_type(_valtype(basis, x), typeof(x))

_gradtype(basis::AbstractPoly4MLBasis, x::StaticArray) = 
      StaticArrays.similar_type(typeof(x), promote_type(_valtype(basis, x)))

_hesstype(basis::AbstractPoly4MLBasis, x::Number) = 
      promote_type(_valtype(basis, x), typeof(x))

_hesstype(basis::AbstractPoly4MLBasis, x::StaticVector) = 
      StaticArrays.similar_type(typeof(x), promote_type(_valtype(basis, x)))

_laplacetype(basis::AbstractPoly4MLBasis, x::Number) = 
      _hesstype(basis, x)

_laplacetype(basis::AbstractPoly4MLBasis, x::Number) = 
      eltype(_hesstype(basis, x))


# --------------------------------------- 
# allocation interface interface 

_alloc(basis::AbstractPoly4MLBasis, x::SINGLE) = 
      Vector{ _valtype(basis, x) }(undef, length(basis))

_alloc(basis::AbstractPoly4MLBasis, X::BATCH) = 
      Array{ _valtype(basis, X[1]) }(undef, (length(basis), length(X)))

_alloc_d(basis::AbstractPoly4MLBasis, x::SINGLE) = 
      Vector{ _gradtype(basis, x) }(undef, length(basis))

_alloc_d(basis::AbstractPoly4MLBasis, X::BATCH) = 
      Array{ _gradtype(basis, X[1]) }(undef, (length(basis), length(X)))

_alloc_dd(basis::AbstractPoly4MLBasis, x::SINGLE) = 
      Vector{ _hesstype(basis, x) }(undef, length(basis))

_alloc_dd(basis::AbstractPoly4MLBasis, X::BATCH) = 
      Array{ _hesstype(basis, X[1]) }(undef, (length(basis), length(X)))

_alloc_ed(basis::AbstractPoly4MLBasis, x) = 
      _alloc(basis, x), _alloc_d(basis, x)

_alloc_ed2(basis::AbstractPoly4MLBasis, x) = 
   _alloc(basis, x), _alloc_d(basis, x), _alloc_dd(basis, x)



# --------------------------------------- 
# evaluation interface 

(basis::AbstractPoly4MLBasis)(x) = evaluate(basis, x)
            
evaluate(basis::AbstractPoly4MLBasis, x) = 
      evaluate!(_alloc(basis, x), basis, x)

evaluate_ed(basis::AbstractPoly4MLBasis, x) = 
      evaluate_ed!(_alloc(basis, x), _alloc_d(basis, x), basis, x)

evaluate_d(basis::AbstractPoly4MLBasis, x) = evaluate_ed(basis, x)[2] 

evaluate_ed2(basis::AbstractPoly4MLBasis, x) = 
      evaluate_ed2!(_alloc(basis, x), _alloc_d(basis, x), _alloc_dd(basis, x), 
                    basis, x)

evaluate_dd(basis::AbstractPoly4MLBasis, x) = evaluate_ed2(basis, x)[3] 

