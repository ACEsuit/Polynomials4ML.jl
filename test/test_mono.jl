using Polynomials4ML, Test
using Polynomials4ML: evaluate, evaluate_d, evaluate_dd
using Polynomials4ML.Testing: println_slim, print_tf, test_derivatives, 
                              test_withalloc, test_chainrules


##

@info("Testing Real Trigonometric Polynomials (RTrigBasis)")
N = 10
basis = MonoBasis(N) 

@info("      correctness")
mm = natural_indices(basis)
for ntest = 1:30
   local x 
   x = 2*π * rand()
   P = basis(x)
   P2 = [ x^m for m in mm ]
   print_tf(@test P ≈ P2)
end
println() 


##

test_evaluate_xx(basis)
test_withalloc(basis)
test_chainrules(basis)
