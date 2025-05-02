
# Lux Integration


## Internals 

There is a default 
```julia
_evaluate!(P, dP, basis, X) = _evaluate!(P, dP, basis, X, nothing, nothing) 
```
This assumes that the basis has no parameters and no state other than 
frozen hyperparameters stored in `basis` itself. If `basis` does have parameters 
and or state, then it should overload this call. For example, 

```julia
struct LearnableCheb{N, SW} <: AbstractP4MLBasis 
   cheb::ChebBasis{N}
   W::SW   # store as SMatrix!!!
end

# the following "transfer" to a different method with explicit parameters 
# is zero-cost! 

_evaluate!(P, dP, basis, X) = 
      _evaluate!(P, dP, basis, X, (W = basis.W,), NamedTuple())

function _evaluate!(P, dP, basis, X, ps, st)
   T, dT = evaluate_ed(basis.cheb) 
   mul!(P, ps.W, T)
   isnothing(dP) || mul!(dP, ps.W, dT)   
   return nothing 
end
```
