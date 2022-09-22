
export natural_indices, 
       index, 
       evaluate, 
       evaluate_d, 
       evaluate_dd, 
       evaluate_ed, 
       evaluate_ed2, 
       evaluate!, 
       evaluate_ed!, 
       evaluate_ed2!, 
       orthpolybasis


function natural_indices end 
function index end
function evaluate end 
function evaluate_d end 
function evaluate_dd end 
function evaluate_ed end 
function evaluate_ed2 end 
function evaluate! end 
function evaluate_ed! end 
function evaluate_ed2! end 
function orthpolybasis end


abstract type  PolyBasis4ML end 

(basis::PolyBasis4ML)(x) = evaluate(basis, x)
            
evaluate(basis::PolyBasis4ML, x) = evaluate!(_alloc(basis, x), basis, x)

evaluate_ed(basis::PolyBasis4ML, x) = 
      evaluate_ed!(_alloc(basis, x), _alloc(basis, x), basis, x)

evaluate_d(basis::PolyBasis4ML, x) = evaluate_ed(basis, x)[2] 

evaluate_ed2(basis::PolyBasis4ML, x) = 
      evaluate_ed2!(_alloc(basis, x), _alloc(basis, x), _alloc(basis, x), 
                    basis, x)

evaluate_dd(basis::PolyBasis4ML, x) = evaluate_ed2(basis, x)[3] 

