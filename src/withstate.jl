

struct WithState{TB, TP, TS} <: AbstractP4MLBasis
   basis::TB 
   ps::TP 
   st::TS 
end

_valtype(basis::WithState, T::Type, args...) = 
      _valtype(basis.basis, T, args...)

Base.length(basis::WithState) = length(basis.basis)

_evaluate!(P, dP, basis::WithState, X) = 
      _evaluate!(P, dP, basis.basis, X, basis.ps, basis.st) 

