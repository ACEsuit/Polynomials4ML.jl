
abstract type AbstractPoly4MLBasis end

(basis::AbstractPoly4MLBasis)(x) = evaluate(basis, x)
            
evaluate(basis::AbstractPoly4MLBasis, x) = evaluate!(_alloc(basis, x), basis, x)

evaluate_ed(basis::AbstractPoly4MLBasis, x) = 
      evaluate_ed!(_alloc(basis, x), _alloc(basis, x), basis, x)

evaluate_d(basis::AbstractPoly4MLBasis, x) = evaluate_ed(basis, x)[2] 

evaluate_ed2(basis::AbstractPoly4MLBasis, x) = 
      evaluate_ed2!(_alloc(basis, x), _alloc(basis, x), _alloc(basis, x), 
                    basis, x)

evaluate_d2(basis::AbstractPoly4MLBasis, x) = evaluate_ed2(basis, x)[3] 

