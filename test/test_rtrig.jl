using Polynomials4ML, Test
using Polynomials4ML: evaluate, evaluate_d, evaluate_dd
using Polynomials4ML.Testing: println_slim, print_tf, 
                              test_evaluate_xx, 
                              test_withalloc, 
                              test_chainrules


##

@info("Testing Real Trigonometric Polynomials (RTrigBasis)")
N = 10
basis = RTrigBasis(N) 

@info("      correctness")
mm = natural_indices(basis)
for ntest = 1:30
   local x 
   x = 2*π * rand()
   P = basis(x)
   P2 = [ (m >= 0 ? cos(m*x) : sin(abs(m)*x)) for m in mm ]
   print_tf(@test P ≈ P2)
end
println() 


##

test_evaluate_xx(basis)
test_withalloc(basis) 
test_chainrules(basis)
